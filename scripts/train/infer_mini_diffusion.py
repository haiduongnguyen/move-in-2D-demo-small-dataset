#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from move_in_2d_small.data import load_manifest
from move_in_2d_small.data.condition_cache import cache_path_for_row, load_condition_cache
from move_in_2d_small.models import DDPMMotionScheduler, MiniMoveIn2DDiffusion
from move_in_2d_small.paths import PROJECT_DATA, resolve_project_path
from move_in_2d_small.viz import render_motion_contact_sheet, render_motion_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample motion from a trained mini diffusion checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--scene-image", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ddim-steps", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(PROJECT_DATA / "models" / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_DATA / "models" / "huggingface"))
    ckpt_path = resolve_project_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=args.device)
    config = ckpt["config"]
    if bool(args.scene_image) != bool(args.prompt):
        raise SystemExit("Use --scene-image and --prompt together for custom inference.")
    rows = load_manifest(config["manifest_path"]) if not args.scene_image else []
    if args.scene_image:
        selected = []
    elif args.sample_id:
        selected = [row for row in rows if row["sample_id"] == args.sample_id]
    else:
        selected = [row for row in rows if row.get("split") == args.split][: args.count]
    if not selected and not args.scene_image:
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
    if args.scene_image:
        run_custom_prompt(args, config, model, scheduler, device, out_dir, steps)
        return
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


def run_custom_prompt(args, config, model, scheduler, device: torch.device, out_dir: Path, steps: int) -> None:
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

    scene_path = resolve_project_path(args.scene_image)
    image = cv2.imread(str(scene_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read scene image: {scene_path}")
    clip_processor = CLIPProcessor.from_pretrained(config["clip_model"])
    clip_model = CLIPModel.from_pretrained(config["clip_model"]).to(device).eval()
    dino_processor = AutoImageProcessor.from_pretrained(config["dinov2_model"])
    dino_model = AutoModel.from_pretrained(config["dinov2_model"]).to(device).eval()
    with torch.inference_mode():
        text_embed = encode_text(clip_processor, clip_model, args.prompt, device).unsqueeze(0)
        scene_tokens = encode_scene(dino_processor, dino_model, scene_path, config["scene_tokens"], device).unsqueeze(0)
        pred = scheduler.ddim_sample(
            model,
            text_embed,
            scene_tokens,
            (1, config["num_frames"], config["num_joints"], 2),
            steps=steps,
        )[0].cpu().numpy()
    pred_px = norm_to_px(pred, image.shape[1], image.shape[0], image.shape[:2])
    visibility = np.ones((config["num_frames"], config["num_joints"]), dtype=bool)
    safe_name = safe_slug(args.prompt)
    label = f"generated custom {safe_name}"
    contact_path = out_dir / f"custom_{safe_name}_contact_sheet.jpg"
    video_path = out_dir / f"custom_{safe_name}.mp4"
    render_motion_contact_sheet(image, pred_px, visibility, contact_path, label)
    render_motion_video(image, pred_px, visibility, video_path, label)
    print(contact_path)
    print(video_path)


def encode_text(processor, model, prompt: str, device: torch.device) -> torch.Tensor:
    inputs = processor(text=[prompt], return_tensors="pt", padding=True, truncation=True).to(device)
    embed = model.get_text_features(**inputs)
    if not torch.is_tensor(embed):
        embed = embed.pooler_output
    return torch.nn.functional.normalize(embed, dim=-1)[0]


def encode_scene(processor, model, image_path: Path, scene_tokens: int, device: torch.device) -> torch.Tensor:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
    hidden = model(**inputs).last_hidden_state[0]
    patches = hidden[1:] if hidden.shape[0] > scene_tokens else hidden
    pooled = pool_tokens(patches, scene_tokens)
    return torch.nn.functional.normalize(pooled, dim=-1)


def pool_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if tokens.shape[0] == target_tokens:
        return tokens
    chunks = torch.chunk(tokens, target_tokens, dim=0)
    pooled = [chunk.mean(dim=0) for chunk in chunks]
    if len(pooled) < target_tokens:
        pooled.extend([pooled[-1]] * (target_tokens - len(pooled)))
    return torch.stack(pooled[:target_tokens], dim=0)


def safe_slug(text: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:80] or "prompt"


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
