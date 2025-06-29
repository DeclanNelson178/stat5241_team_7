from typing import List
import torch
import torch.nn as nn


class FeedForwardNet(nn.Module):
    def __init__(
        self, input_dim, with_batch_norm: bool, hidden_dims: List[int], dropout: float
    ):
        super().__init__()
        layers = []
        prev_hidden_layer = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_hidden_layer, h))
            if with_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_hidden_layer = h
        layers.append(
            nn.Linear(hidden_dims[-1], 1)
        )  # Output logits for BCEWithLogitsLoss
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()
