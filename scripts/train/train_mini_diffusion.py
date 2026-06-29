#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from move_in_2d_small.data import build_action_vocab, load_manifest
from move_in_2d_small.data.condition_cache import CachedConditionMotionDataset
from move_in_2d_small.data.penn_action_dataset import collate_motion_batch
from move_in_2d_small.models import DDPMMotionScheduler, MiniMoveIn2DDiffusion
from move_in_2d_small.paths import resolve_project_path
from move_in_2d_small.viz import render_motion_contact_sheet, render_motion_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mini Move-in-2D diffusion transformer.")
    parser.add_argument("--config", default="configs/penn_action_bbox_lama_full.json")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--sample-ddim-steps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    rows = load_manifest(config["manifest_path"])
    action_vocab = build_action_vocab(rows)
    run_name = args.run_name or datetime.now().strftime("mini_diffusion_%Y%m%d_%H%M%S")
    run_dir = resolve_project_path(config["output_root"]) / run_name
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps({"script_args": vars(args), "dataset_config": config, "action_vocab": action_vocab}, indent=2),
        encoding="utf-8",
    )
    train_ds = make_dataset(config, rows, action_vocab, "train", args.max_train_samples)
    val_ds = make_dataset(config, rows, action_vocab, "val", args.max_val_samples)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_motion_batch
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_motion_batch
    )
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
    scheduler = DDPMMotionScheduler(config["diffusion_timesteps"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, scheduler, train_loader, optimizer, device)
        val_loss = run_epoch(model, scheduler, val_loader, None, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row))
        save_checkpoint(run_dir / "latest.pt", model, scheduler, config, args, history)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(run_dir / "best.pt", model, scheduler, config, args, history)
        if args.sample_count > 0:
            render_epoch_samples(model, scheduler, val_loader, device, sample_dir / f"epoch_{epoch:03d}", args.sample_count, args.sample_ddim_steps)
        (run_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"run_dir={run_dir}")


def make_dataset(config, rows, action_vocab, split: str, max_samples: int | None):
    dataset = CachedConditionMotionDataset(
        config["manifest_path"],
        split=split,
        image_size=config["image_size"],
        action_vocab=action_vocab,
        rows=rows,
        condition_cache_root=config["condition_cache_root"],
    )
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset


def run_epoch(model, scheduler, loader, optimizer, device: torch.device) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    count = 0
    for batch in loader:
        x_start = batch["motion"].to(device)
        visibility = batch["visibility"].to(device).unsqueeze(-1)
        text_embed = batch["text_embed"].to(device)
        scene_tokens = batch["scene_tokens"].to(device)
        timestep = torch.randint(0, scheduler.num_train_timesteps, (x_start.shape[0],), device=device)
        noise = torch.randn_like(x_start)
        noisy = scheduler.q_sample(x_start, timestep, noise)
        pred = model(noisy, timestep, text_embed, scene_tokens)
        loss_raw = (pred - noise).pow(2)
        loss = (loss_raw * visibility).sum() / (visibility.sum() * x_start.shape[-1]).clamp_min(1.0)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.detach().cpu()) * x_start.shape[0]
        count += x_start.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def render_epoch_samples(model, scheduler, loader, device: torch.device, out_dir: Path, count: int, ddim_steps: int) -> None:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for batch in loader:
        text_embed = batch["text_embed"].to(device)
        scene_tokens = batch["scene_tokens"].to(device)
        pred = scheduler.ddim_sample(
            model,
            text_embed,
            scene_tokens,
            (text_embed.shape[0], 64, 13, 2),
            steps=ddim_steps,
        ).detach().cpu().numpy()
        for idx in range(text_embed.shape[0]):
            sample_id = batch["sample_id"][idx]
            image = cv2.imread(str(resolve_project_path(batch["scene_image_path"][idx])), cv2.IMREAD_COLOR)
            pred_px = norm_to_px(pred[idx], float(batch["image_width"][idx]), float(batch["image_height"][idx]), image.shape[:2])
            visibility = np.ones((pred_px.shape[0], pred_px.shape[1]), dtype=bool)
            label = f"generated {sample_id} {batch['action_label'][idx]}"
            render_motion_contact_sheet(image, pred_px, visibility, out_dir / f"{sample_id}_generated_contact_sheet.jpg", label)
            render_motion_video(image, pred_px, visibility, out_dir / f"{sample_id}_generated.mp4", label)
            rendered += 1
            if rendered >= count:
                return


def norm_to_px(pred: np.ndarray, source_width: float, source_height: float, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    out = np.clip(pred.copy(), 0.0, 1.0)
    out[..., 0] *= source_width
    out[..., 1] *= source_height
    out[..., 0] *= width / max(source_width, 1.0)
    out[..., 1] *= height / max(source_height, 1.0)
    return out


def save_checkpoint(path: Path, model, scheduler, config, args, history) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": config,
            "script_args": vars(args),
            "history": history,
        },
        path,
    )


if __name__ == "__main__":
    main()
