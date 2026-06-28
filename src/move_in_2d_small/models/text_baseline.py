from __future__ import annotations

import torch
from torch import nn


class TextMotionBaseline(nn.Module):
    """Tiny action/text-only baseline for background+text -> 2D motion.

    This intentionally ignores the image. It gives us a sanity-check baseline:
    can the model learn action-specific average 2D motion from labels alone?
    """

    def __init__(
        self,
        num_actions: int,
        num_frames: int = 64,
        num_joints: int = 13,
        hidden_dim: int = 256,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.embedding = nn.Embedding(num_actions, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_frames * num_joints * 2),
        )

    def forward(self, action_id: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(action_id)
        out = self.net(emb)
        return out.view(-1, self.num_frames, self.num_joints, 2)
