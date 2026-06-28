#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from move_in_2d_small.data import load_manifest
from move_in_2d_small.paths import resolve_project_path
from move_in_2d_small.viz import render_motion_contact_sheet, render_motion_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render skeleton previews from dataset rows.")
    parser.add_argument("--config", default="configs/penn_action_bbox_lama_full.json")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-root-displacement", type=float, default=40.0)
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument("--out-dir", default="project_data/debug_runs/dataset_loader_previews")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    rows = [row for row in load_manifest(config["manifest_path"]) if row.get("split") == args.split]
    candidates = [
        row for row in rows if row.get("motion_stats", {}).get("root_displacement_px", 0.0) >= args.min_root_displacement
    ]
    if len(candidates) < args.count:
        candidates = rows
    rng = random.Random(args.seed)
    chosen = rng.sample(candidates, min(args.count, len(candidates)))
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in chosen:
        image = cv2.imread(str(resolve_project_path(row["scene_image_path"])), cv2.IMREAD_COLOR)
        motion = np.load(resolve_project_path(row["motion_path"]))
        keypoints = motion["keypoints_2d_px"].astype(np.float32)
        visibility = motion["visibility"].astype(bool)
        keypoints = scale_keypoints(keypoints, image.shape[:2], row["image_width"], row["image_height"])
        label = f"{row['sample_id']} {row['action_label']}"
        contact_path = out_dir / f"{row['sample_id']}_contact_sheet.jpg"
        render_motion_contact_sheet(image, keypoints, visibility, contact_path, label)
        print(contact_path)
        if args.write_video:
            video_path = out_dir / f"{row['sample_id']}_motion.mp4"
            render_motion_video(image, keypoints, visibility, video_path, label)
            print(video_path)


def scale_keypoints(
    keypoints: np.ndarray, image_shape: tuple[int, int], source_width: float, source_height: float
) -> np.ndarray:
    height, width = image_shape
    scaled = keypoints.copy()
    scaled[..., 0] *= width / max(float(source_width), 1.0)
    scaled[..., 1] *= height / max(float(source_height), 1.0)
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == "__main__":
    main()

