#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from move_in_2d_small.models import DDPMMotionScheduler, MiniMoveIn2DDiffusion


def main() -> None:
    config = json.loads((PROJECT_ROOT / "configs/penn_action_bbox_lama_full.json").read_text(encoding="utf-8"))
    batch = 2
    model = MiniMoveIn2DDiffusion(
        text_dim=config["text_dim"],
        scene_dim=config["scene_dim"],
        num_frames=config["num_frames"],
        num_joints=config["num_joints"],
        d_model=64,
        num_layers=2,
        num_heads=4,
    )
    scheduler = DDPMMotionScheduler(num_train_timesteps=100)
    x = torch.rand(batch, config["num_frames"], config["num_joints"], 2)
    text = torch.rand(batch, config["text_dim"])
    scene = torch.rand(batch, config["scene_tokens"], config["scene_dim"])
    timestep = torch.randint(0, scheduler.num_train_timesteps, (batch,))
    noise = torch.randn_like(x)
    noisy = scheduler.q_sample(x, timestep, noise)
    pred = model(noisy, timestep, text, scene)
    sample = scheduler.ddim_sample(model, text, scene, x.shape, steps=4)
    print("pred_shape", tuple(pred.shape), "finite", bool(torch.isfinite(pred).all()))
    print("sample_shape", tuple(sample.shape), "finite", bool(torch.isfinite(sample).all()))
    assert pred.shape == x.shape
    assert sample.shape == x.shape
    assert torch.isfinite(pred).all()
    assert torch.isfinite(sample).all()
    print("smoke=ok")


if __name__ == "__main__":
    main()
