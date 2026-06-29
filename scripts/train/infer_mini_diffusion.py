#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from move_in_2d_small.data import load_manifest
from move_in_2d_small.data.condition_cache import cache_path_for_row, load_condition_cache
from move_in_2d_small.models import DDPMMotionScheduler, MiniMoveIn2DDiffusion
from move_in_2d_small.paths import resolve_project_path
from move_in_2d_small.viz import render_motion_contact_sheet, render_motion_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample motion from a trained mini diffusion checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ddim-steps", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = resolve_project_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=args.device)
    config = ckpt["config"]
    rows = load_manifest(config["manifest_path"])
    if args.sample_id:
        selected = [row for row in rows if row["sample_id"] == args.sample_id]
    else:
        selected = [row for row in rows if row.get("split") == args.split][: args.count]
    if not selected:
        raise SystemExit("No samples selected for inference.")
    device = torch.device(args.device)
    model = MiniMoveIn2DDiffusion(
        text_dim=config["text_dim"],
        scene_dim=config["scene_dim"],
        num_frames=config["num_frames"],
        num_joints=config["num_joints"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scheduler = DDPMMotionScheduler(config["diffusion_timesteps"]).to(device)
    steps = args.ddim_steps or config.get("ddim_steps", 50)
    out_dir = resolve_project_path(args.out_dir) if args.out_dir else ckpt_path.parent / "inference_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in selected:
        cond = load_condition_cache(cache_path_for_row(config["condition_cache_root"], row["sample_id"]))
        text_embed = cond["text_embed"].unsqueeze(0).to(device)
        scene_tokens = cond["scene_tokens"].unsqueeze(0).to(device)
        with torch.inference_mode():
            pred = scheduler.ddim_sample(
                model,
                text_embed,
                scene_tokens,
                (1, config["num_frames"], config["num_joints"], 2),
                steps=steps,
            )[0].cpu().numpy()
        image = cv2.imread(str(resolve_project_path(row["scene_image_path"])), cv2.IMREAD_COLOR)
        pred_px = norm_to_px(pred, row["image_width"], row["image_height"], image.shape[:2])
        visibility = np.ones((config["num_frames"], config["num_joints"]), dtype=bool)
        label = f"generated {row['sample_id']} {row['action_label']}"
        contact_path = out_dir / f"{row['sample_id']}_generated_contact_sheet.jpg"
        video_path = out_dir / f"{row['sample_id']}_generated.mp4"
        render_motion_contact_sheet(image, pred_px, visibility, contact_path, label)
        render_motion_video(image, pred_px, visibility, video_path, label)
        print(contact_path)
        print(video_path)


def norm_to_px(pred: np.ndarray, source_width: float, source_height: float, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    out = np.clip(pred.copy(), 0.0, 1.0)
    out[..., 0] *= float(source_width)
    out[..., 1] *= float(source_height)
    out[..., 0] *= width / max(float(source_width), 1.0)
    out[..., 1] *= height / max(float(source_height), 1.0)
    return out


if __name__ == "__main__":
    main()
