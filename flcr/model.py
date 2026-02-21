from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class TextItemTrainDataset(Dataset):
    def __init__(self, query_embeddings: torch.Tensor, target_ids: torch.Tensor):
        self.query_embeddings = query_embeddings.float()
        self.target_ids = target_ids.long()

    def __len__(self) -> int:
        return int(self.target_ids.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "query_embeddings": self.query_embeddings[idx],
            "target_ids": self.target_ids[idx],
        }


class MLPProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(x), dim=-1)


class DualTowerRetriever(nn.Module):
    def __init__(
        self,
        num_items: int,
        query_embedding_dim: int,
        sentence_embedding_dim: int,
        item_title_embeddings: torch.Tensor,
        item_metadata_embeddings: torch.Tensor,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_items = num_items
        self.query_embedding_dim = query_embedding_dim
        self.sentence_embedding_dim = sentence_embedding_dim
        self.query_projection = MLPProjector(query_embedding_dim, hidden_dim, dropout=dropout)
        self.item_fusion = nn.Sequential(
            nn.Linear(sentence_embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.item_projection = MLPProjector(hidden_dim, hidden_dim, dropout=dropout)
        self.register_buffer("item_title_embeddings", item_title_embeddings.float(), persistent=False)
        self.register_buffer("item_metadata_embeddings", item_metadata_embeddings.float(), persistent=False)

    def encode_queries(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        return self.query_projection(query_embeddings)

    def encode_item_base(self, item_ids: torch.Tensor) -> torch.Tensor:
        title_repr = self.item_title_embeddings[item_ids]
        metadata_repr = self.item_metadata_embeddings[item_ids]
        fused = torch.cat([title_repr, metadata_repr], dim=-1)
        return self.item_fusion(fused)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        base_repr = self.encode_item_base(item_ids)
        return self.item_projection(base_repr)

    def forward(self, query_embeddings: torch.Tensor, target_ids: torch.Tensor):
        query_repr = self.encode_queries(query_embeddings)
        item_repr = self.encode_items(target_ids)
        return query_repr, item_repr
