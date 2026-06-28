#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from move_in_2d_small.data import PennActionMotionDataset, build_action_vocab, load_manifest
from move_in_2d_small.data.penn_action_dataset import collate_motion_batch
from move_in_2d_small.models import TextMotionBaseline
from move_in_2d_small.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny text/action-only 2D motion baseline.")
    parser.add_argument("--config", default="configs/penn_action_bbox_lama_full.json")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    rows = load_manifest(config["manifest_path"])
    action_vocab = build_action_vocab(rows)
    run_name = args.run_name or datetime.now().strftime("text_baseline_%Y%m%d_%H%M%S")
    run_dir = resolve_project_path(config["output_root"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps({"script_args": vars(args), "dataset_config": config, "action_vocab": action_vocab}, indent=2),
        encoding="utf-8",
    )

    train_ds = PennActionMotionDataset(config["manifest_path"], "train", config["image_size"], action_vocab, rows)
    val_ds = PennActionMotionDataset(config["manifest_path"], "val", config["image_size"], action_vocab, rows)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_motion_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_motion_batch,
    )
    device = torch.device(args.device)
    model = TextMotionBaseline(
        num_actions=len(action_vocab),
        num_frames=config["num_frames"],
        num_joints=config["num_joints"],
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_loss = run_epoch(model, val_loader, None, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row))
    torch.save(
        {
            "model_state": model.state_dict(),
            "action_vocab": action_vocab,
            "config": config,
            "script_args": vars(args),
            "history": history,
        },
        run_dir / "checkpoint.pt",
    )
    (run_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"run_dir={run_dir}")


def run_epoch(
    model: TextMotionBaseline,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    count = 0
    loss_fn = nn.SmoothL1Loss(reduction="none")
    for batch in loader:
        action_id = batch["action_id"].to(device)
        target = batch["motion"].to(device)
        visibility = batch["visibility"].to(device).unsqueeze(-1)
        pred = model(action_id)
        loss_raw = loss_fn(pred, target)
        loss = (loss_raw * visibility).sum() / (visibility.sum() * target.shape[-1]).clamp_min(1.0)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += float(loss.detach().cpu()) * len(action_id)
        count += len(action_id)
    return total / max(count, 1)


if __name__ == "__main__":
    main()
