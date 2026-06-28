#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from move_in_2d_small.data import PennActionMotionDataset, build_action_vocab, load_manifest
from move_in_2d_small.data.penn_action_dataset import collate_motion_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the Penn Action bbox+LaMa dataset loader.")
    parser.add_argument("--config", default="configs/penn_action_bbox_lama_full.json")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    rows = load_manifest(config["manifest_path"])
    action_vocab = build_action_vocab(rows)
    dataset = PennActionMotionDataset(
        config["manifest_path"],
        split=args.split,
        image_size=config["image_size"],
        action_vocab=action_vocab,
        rows=rows,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_motion_batch)
    batch = next(iter(loader))
    print(f"dataset={config['name']}")
    print(f"split={args.split} samples={len(dataset)} actions={len(action_vocab)}")
    print(f"image={tuple(batch['image'].shape)} dtype={batch['image'].dtype}")
    print(f"motion={tuple(batch['motion'].shape)} dtype={batch['motion'].dtype}")
    print(f"visibility={tuple(batch['visibility'].shape)} dtype={batch['visibility'].dtype}")
    print(f"action_id={tuple(batch['action_id'].shape)}")
    print(f"sample_ids={batch['sample_id']}")
    print(f"text_prompt[0]={batch['text_prompt'][0]}")
    assert batch["image"].shape == (args.batch_size, 3, config["image_size"], config["image_size"])
    assert batch["motion"].shape[1:] == (config["num_frames"], config["num_joints"], 2)
    assert torch.isfinite(batch["motion"]).all()
    assert torch.isfinite(batch["image"]).all()
    print("smoke=ok")


if __name__ == "__main__":
    main()

