from __future__ import annotations

import math

import torch
from torch import nn


class DDPMMotionScheduler(nn.Module):
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.num_train_timesteps = num_train_timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x_start: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, timestep, x_start.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, timestep, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        text_embed: torch.Tensor,
        scene_tokens: torch.Tensor,
        shape: tuple[int, int, int, int],
        steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        device = text_embed.device
        x = torch.randn(shape, device=device)
        step_indices = torch.linspace(self.num_train_timesteps - 1, 0, steps, device=device).long()
        for i, timestep_value in enumerate(step_indices):
            timestep = torch.full((shape[0],), int(timestep_value), device=device, dtype=torch.long)
            pred_noise = model(x, timestep, text_embed, scene_tokens)
            alpha_t = extract(self.alphas_cumprod, timestep, x.shape)
            pred_x0 = (x - torch.sqrt(1.0 - alpha_t) * pred_noise) / torch.sqrt(alpha_t).clamp_min(1e-8)
            if i == len(step_indices) - 1:
                x = pred_x0
                continue
            prev_timestep = torch.full((shape[0],), int(step_indices[i + 1]), device=device, dtype=torch.long)
            alpha_prev = extract(self.alphas_cumprod, prev_timestep, x.shape)
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).clamp_min(0.0)
            direction = torch.sqrt((1.0 - alpha_prev - sigma**2).clamp_min(0.0)) * pred_noise
            x = torch.sqrt(alpha_prev) * pred_x0 + direction
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
        return x.clamp(0.0, 1.0)


def extract(values: torch.Tensor, timestep: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    out = values.gather(0, timestep)
    return out.reshape(timestep.shape[0], *([1] * (len(target_shape) - 1)))


def sinusoidal_timestep_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    exponent = -math.log(10000.0) * torch.arange(half, device=timestep.device, dtype=torch.float32) / max(half - 1, 1)
    freqs = torch.exp(exponent)
    args = timestep.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb
