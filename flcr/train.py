from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flcr.config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    DATASET_PATH,
    DEVICE,
    EPOCHS,
    EPOCH_CHECKPOINT_DIR,
    HIDDEN_DIM,
    LATEST_CHECKPOINT_PATH,
    LEARNING_RATE,
    NUM_WORKERS,
    WEIGHT_DECAY,
    ensure_directories,
    seed_everything,
)
from flcr.model import DualTowerRetriever, TextItemTrainDataset

try:
    import wandb
except ImportError:
    wandb = None


def move_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.to(DEVICE) for key, value in batch.items()}


def build_model(dataset: dict) -> DualTowerRetriever:
    return DualTowerRetriever(
        num_items=dataset["num_items"],
        query_embedding_dim=dataset["query_embedding_dim"],
        sentence_embedding_dim=dataset["sentence_embedding_dim"],
        item_title_embeddings=dataset["item_title_embeddings"],
        item_overview_embeddings=dataset["item_overview_embeddings"],
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)


def run_epoch(model: DualTowerRetriever, loader: DataLoader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = move_batch(batch)
        if is_train:
            optimizer.zero_grad()

        query_repr, item_repr = model(batch["query_embeddings"], batch["target_ids"])
        logits = query_repr @ item_repr.T
        targets = torch.arange(logits.shape[0], device=logits.device)
        loss = F.cross_entropy(logits, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = int(batch["target_ids"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="retrieval-system")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    seed_everything()
    ensure_directories()
    ensure_wandb_state(args)
    dataset = torch.load(DATASET_PATH, map_location="cpu")
    dataset_name = str(dataset.get("dataset_name", "msrd"))

    train_loader = DataLoader(
        TextItemTrainDataset(dataset["train_query_embeddings"], dataset["train_target_ids"]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_query_embeddings = dataset["val_loss_query_embeddings"] if "val_loss_query_embeddings" in dataset else dataset["val_query_embeddings"]
    val_target_ids = dataset["val_loss_target_ids"] if "val_loss_target_ids" in dataset else dataset["val_target_ids"]
    val_loader = DataLoader(
        TextItemTrainDataset(
            val_query_embeddings,
            val_target_ids,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = build_model(dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=WEIGHT_DECAY)

    start_epoch = 0
    latest_checkpoint = None
    if args.resume_from:
        start_epoch = load_training_state(args.resume_from, model, optimizer)
        print(
            f"Resumed from {args.resume_from} "
            f"(start_epoch={start_epoch + 1})"
        )
    run = init_wandb(args, dataset)
    train_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        train_loss = run_epoch(model, train_loader, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, optimizer=None)
        epoch_seconds = time.perf_counter() - epoch_start
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f} "
            f"epoch_seconds={epoch_seconds:.2f} device={DEVICE}"
        )
        latest_checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "num_items": dataset["num_items"],
                "query_embedding_dim": dataset["query_embedding_dim"],
                "sentence_embedding_dim": dataset["sentence_embedding_dim"],
                "dataset_name": dataset_name,
            },
            "val_loss": val_loss,
            "epoch": epoch + 1,
        }
        torch.save(latest_checkpoint, LATEST_CHECKPOINT_PATH)
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            EPOCH_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            epoch_checkpoint_path = (
                EPOCH_CHECKPOINT_DIR
                / f"msrd_text_retriever_epoch_{epoch + 1:03d}.pt"
            )
            torch.save(latest_checkpoint, epoch_checkpoint_path)
        if run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch_seconds": epoch_seconds,
                    "train/start_epoch": start_epoch + 1,
                    "data/dataset_name": dataset_name,
                }
            )

    torch.save(latest_checkpoint, CHECKPOINT_PATH)
    total_seconds = time.perf_counter() - train_start
    print(
        f"Saved checkpoint to {CHECKPOINT_PATH} "
        f"(dataset_name={dataset_name}, last_epoch={latest_checkpoint['epoch']}, "
        f"last_val_loss={latest_checkpoint['val_loss']:.4f}, total_seconds={total_seconds:.2f}, "
        f"latest={LATEST_CHECKPOINT_PATH})"
    )
    if run is not None:
        wandb.summary["last_epoch"] = latest_checkpoint["epoch"]
        wandb.summary["last_val_loss"] = latest_checkpoint["val_loss"]
        wandb.summary["train_total_seconds"] = total_seconds
        wandb.finish()


def ensure_wandb_state(args) -> None:
    if args.wandb and wandb is None:
        raise RuntimeError("wandb is not installed. Run `pip install wandb` or `pip install -r requirements.txt`.")


def init_wandb(args, dataset: dict):
    if not args.wandb:
        return None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": str(DEVICE),
            "num_items": dataset["num_items"],
            "train_examples": int(dataset["train_target_ids"].shape[0]),
            "val_examples": int(
                (dataset["val_loss_target_ids"] if "val_loss_target_ids" in dataset else dataset["val_target_ids"]).shape[0]
            ),
            "query_embedding_dim": dataset["query_embedding_dim"],
            "sentence_embedding_dim": dataset["sentence_embedding_dim"],
            "hidden_dim": HIDDEN_DIM,
            "weight_decay": WEIGHT_DECAY,
            "save_every": args.save_every,
        },
    )
    return run


def load_training_state(path: str, model: DualTowerRetriever, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    start_epoch = int(checkpoint.get("epoch", 0))
    return start_epoch


if __name__ == "__main__":
    main()
