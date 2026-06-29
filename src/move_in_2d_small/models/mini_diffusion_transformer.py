from __future__ import annotations

import torch
from torch import nn

from move_in_2d_small.models.diffusion import sinusoidal_timestep_embedding


class AdaLNTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_model))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 6))

    def forward(self, tokens: torch.Tensor, timestep_embed: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(timestep_embed).chunk(6, dim=-1)
        x = modulate(self.norm1(tokens), shift1, scale1)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        tokens = tokens + gate1.unsqueeze(1) * attn_out
        x = modulate(self.norm2(tokens), shift2, scale2)
        tokens = tokens + gate2.unsqueeze(1) * self.mlp(x)
        return tokens


class MiniMoveIn2DDiffusion(nn.Module):
    """Paper-like mini denoiser: text token + scene tokens + noisy motion tokens."""

    def __init__(
        self,
        text_dim: int = 512,
        scene_dim: int = 384,
        num_frames: int = 64,
        num_joints: int = 13,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.motion_dim = num_joints * 2
        self.text_proj = nn.Linear(text_dim, d_model)
        self.scene_proj = nn.Linear(scene_dim, d_model)
        self.motion_proj = nn.Linear(self.motion_dim, d_model)
        self.time_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.motion_pos = nn.Parameter(torch.zeros(1, num_frames, d_model))
        self.text_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.scene_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList(
            [AdaLNTransformerBlock(d_model, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, self.motion_dim)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.motion_pos, std=0.02)
        nn.init.normal_(self.text_pos, std=0.02)
        nn.init.normal_(self.scene_pos, std=0.02)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        noisy_motion: torch.Tensor,
        timestep: torch.Tensor,
        text_embed: torch.Tensor,
        scene_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch = noisy_motion.shape[0]
        motion_tokens = noisy_motion.reshape(batch, self.num_frames, self.motion_dim)
        text_token = self.text_proj(text_embed).unsqueeze(1) + self.text_pos
        scene = self.scene_proj(scene_tokens) + self.scene_pos
        motion = self.motion_proj(motion_tokens) + self.motion_pos
        tokens = torch.cat([text_token, scene, motion], dim=1)
        time_embed = self.time_mlp(sinusoidal_timestep_embedding(timestep, tokens.shape[-1]))
        for block in self.blocks:
            tokens = block(tokens, time_embed)
        motion_out = tokens[:, -self.num_frames :]
        pred = self.out(self.final_norm(motion_out))
        return pred.view(batch, self.num_frames, self.num_joints, 2)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
