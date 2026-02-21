from __future__ import annotations

from sentence_transformers import SentenceTransformer

from flcr.config import DEVICE, SENTENCE_MODEL_DIR, SENTENCE_MODEL_NAME, ensure_directories


def main():
    ensure_directories()
    model_device = "mps" if DEVICE.type == "mps" else "cpu"
    if SENTENCE_MODEL_DIR.exists() and any(SENTENCE_MODEL_DIR.iterdir()):
        print(f"Using existing sentence-transformer cache: {SENTENCE_MODEL_DIR}")
        return

    model = SentenceTransformer(SENTENCE_MODEL_NAME, device=model_device)
    model.save(str(SENTENCE_MODEL_DIR))
    print(f"Saved sentence-transformer model to {SENTENCE_MODEL_DIR}")


if __name__ == "__main__":
    main()
