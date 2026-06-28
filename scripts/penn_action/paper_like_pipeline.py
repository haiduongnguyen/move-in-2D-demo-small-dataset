#!/usr/bin/env python3
"""Paper-like preprocessing branch for Penn Action.

This script adds a separate branch beside the GT dataset branch. It starts with
steps that can run locally now (frame quality and an explicitly labelled
GT-mask inpainting baseline) and defines the data contracts used by later
off-the-shelf detector/pose/segmentation models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from prepare_dataset import (
    DATASET_ROOT,
    PROJECT_DATA,
    PROJECT_ROOT,
    find_dataset_dirs,
    frame_files_for,
    read_jsonl,
    write_jsonl,
)


PAPER_ROOT = DATASET_ROOT / "paper_like_processed"
PAPER_MINI_ROOT = DATASET_ROOT / "paper_like_mini_move_in_2d"
COMPARISON_ROOT = DATASET_ROOT / "comparison_gt_vs_pseudo"
GT_MANIFEST = DATASET_ROOT / "mini_move_in_2d" / "manifest_filtered.jsonl"
RAW_MANIFEST = DATASET_ROOT / "processed" / "index" / "manifest_raw.jsonl"
OPENCV_MASK_RCNN_ROOT = PROJECT_DATA / "models" / "opencv" / "mask_rcnn_inception_v2_coco"
OPENCV_MASK_RCNN_GRAPH = OPENCV_MASK_RCNN_ROOT / "frozen_inference_graph.pb"
OPENCV_MASK_RCNN_CONFIG = OPENCV_MASK_RCNN_ROOT / "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
PENN_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (7, 10),
    (10, 11),
    (11, 12),
]


def ensure_layout() -> None:
    for path in [
        PAPER_ROOT / "frame_quality",
        PAPER_ROOT / "person_detection",
        PAPER_ROOT / "pseudo_pose_raw",
        PAPER_ROOT / "person_masks",
        PAPER_ROOT / "backgrounds_inpainted",
        PAPER_ROOT / "background_previews",
        PAPER_ROOT / "motions",
        PAPER_ROOT / "scenes_inpainted",
        PAPER_ROOT / "model_outputs",
        PAPER_MINI_ROOT,
        COMPARISON_ROOT / "previews",
        PROJECT_DATA / "models",
        OPENCV_MASK_RCNN_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def raw_rows() -> list[dict[str, Any]]:
    if RAW_MANIFEST.exists():
        return read_jsonl(RAW_MANIFEST)
    raise SystemExit(
        f"Missing raw manifest: {RAW_MANIFEST}. Run: python3 scripts/penn_action/prepare_dataset.py build-raw-manifest"
    )


def selected_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = raw_rows()
    if args.video_ids:
        wanted = set(args.video_ids)
        rows = [row for row in rows if row["video_id"] in wanted]
    if args.max_videos:
        rows = rows[: args.max_videos]
    return rows


def image_quality(frame_path: Path) -> dict[str, Any]:
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "readable": False,
            "width": None,
            "height": None,
            "brightness_mean": None,
            "blur_score": None,
            "quality_flags": ["unreadable"],
        }
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    flags = []
    if width < 128 or height < 128:
        flags.append("low_resolution")
    if brightness < 20:
        flags.append("too_dark")
    if brightness > 235:
        flags.append("too_bright")
    if blur < 20:
        flags.append("blurry")
    return {
        "readable": True,
        "width": int(width),
        "height": int(height),
        "brightness_mean": brightness,
        "blur_score": blur,
        "quality_flags": flags,
    }


def frame_quality(args: argparse.Namespace) -> None:
    ensure_layout()
    _, frames_root = find_dataset_dirs()
    rows = selected_rows(args)
    frame_out = PAPER_ROOT / "frame_quality" / "frame_quality_manifest.jsonl"
    video_out = PAPER_ROOT / "frame_quality" / "video_quality_summary.jsonl"
    frame_records: list[dict[str, Any]] = []
    video_records: list[dict[str, Any]] = []

    for row in rows:
        video_id = row["video_id"]
        frames = frame_files_for(frames_root, video_id)
        bad = 0
        unreadable = 0
        brightness_values = []
        blur_values = []
        for idx, frame_path in enumerate(frames):
            q = image_quality(frame_path)
            rel = str(frame_path.relative_to(PROJECT_ROOT))
            record = {
                "video_id": video_id,
                "frame_index": idx,
                "frame_path": rel,
                **q,
            }
            frame_records.append(record)
            if not q["readable"]:
                unreadable += 1
            if q["quality_flags"]:
                bad += 1
            if q["brightness_mean"] is not None:
                brightness_values.append(q["brightness_mean"])
            if q["blur_score"] is not None:
                blur_values.append(q["blur_score"])
        total = len(frames)
        video_records.append(
            {
                "video_id": video_id,
                "action_label": row["action_label"],
                "num_frames": total,
                "bad_frame_ratio": bad / max(total, 1),
                "unreadable_frame_ratio": unreadable / max(total, 1),
                "brightness_mean": float(np.mean(brightness_values)) if brightness_values else None,
                "blur_score_median": float(np.median(blur_values)) if blur_values else None,
                "quality_flags": video_flags(total, bad, unreadable),
            }
        )

    write_jsonl(frame_out, frame_records)
    write_jsonl(video_out, video_records)
    print(f"Wrote {len(frame_records)} frame quality rows to {frame_out}")
    print(f"Wrote {len(video_records)} video quality rows to {video_out}")


def video_flags(total: int, bad: int, unreadable: int) -> list[str]:
    flags = []
    if total == 0:
        flags.append("no_frames")
    if unreadable / max(total, 1) > 0.01:
        flags.append("too_many_unreadable_frames")
    if bad / max(total, 1) > 0.5:
        flags.append("too_many_low_quality_frames")
    return flags


def gt_mask_inpaint_baseline(args: argparse.Namespace) -> None:
    """Create an explicitly labelled baseline using GT keypoints to make masks.

    This is not the final paper-like segmentation step. It exists to validate the
    downstream mask -> inpainting -> preview wiring before Mask R-CNN/SAM is added.
    """
    ensure_layout()
    if not GT_MANIFEST.exists():
        raise SystemExit(f"Missing GT manifest: {GT_MANIFEST}")
    rows = read_jsonl(GT_MANIFEST)
    if args.max_samples:
        rows = rows[: args.max_samples]
    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = PAPER_ROOT / "experiments" / "gt_mask_inpaint_baseline" / run_id
    mask_root = run_root / "person_masks"
    bg_root = run_root / "backgrounds_inpainted"
    preview_root = run_root / "background_previews"
    run_root.mkdir(parents=True, exist_ok=False)
    run_config = {
        "run_id": run_id,
        "method": f"gt_bbox_mask_plus_opencv_{args.method}",
        "note": "Debug baseline only. Uses GT bbox rectangle, not paper-like segmentation.",
        "max_samples": args.max_samples,
        "mask_pad": args.mask_pad,
        "inpaint_radius": args.inpaint_radius,
        "opencv_method": args.method,
    }
    (run_root / "config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
    manifest_rows = []
    written = 0
    for row in rows:
        scene_path = PROJECT_ROOT / row["scene_source_path"]
        motion_path = PROJECT_ROOT / row["motion_path"]
        if not scene_path.exists() or not motion_path.exists():
            continue
        data = np.load(motion_path)
        bboxes = data["bbox_xyxy"]
        frame_idx = len(bboxes) // 2
        bbox = bboxes[frame_idx]
        if not np.isfinite(bbox).all():
            continue
        mask_path = mask_root / f"{row['sample_id']}_gt_bbox_baseline.png"
        bg_path = bg_root / f"{row['sample_id']}_opencv_{args.method}_gt_bbox_baseline.jpg"
        preview_path = preview_root / f"{row['sample_id']}_opencv_{args.method}_gt_bbox_baseline.jpg"
        inpaint_with_bbox(
            scene_path,
            bbox,
            mask_path,
            bg_path,
            preview_path,
            args.mask_pad,
            args.inpaint_radius,
            args.method,
        )
        manifest_rows.append(
            {
                "run_id": run_id,
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "action_label": row["action_label"],
                "scene_source_path": row["scene_source_path"],
                "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
                "background_path": str(bg_path.relative_to(PROJECT_ROOT)),
                "preview_path": str(preview_path.relative_to(PROJECT_ROOT)),
                "method": f"gt_bbox_mask_plus_opencv_{args.method}",
            }
        )
        written += 1
    write_jsonl(run_root / "manifest.jsonl", manifest_rows)
    print(f"Wrote {written} GT-mask inpainting baseline samples under {run_root}")


def mask_rcnn_inpaint(args: argparse.Namespace) -> None:
    """Use OpenCV DNN Mask R-CNN person masks, then inpaint.

    This is the first actual Mask R-CNN experiment. GT bbox is only used to
    select the matching person among detections for an apples-to-apples 5-sample
    comparison; the mask itself comes from Mask R-CNN.
    """
    ensure_layout()
    if not OPENCV_MASK_RCNN_GRAPH.exists() or not OPENCV_MASK_RCNN_CONFIG.exists():
        raise SystemExit(
            "Missing OpenCV Mask R-CNN files. Expected:\n"
            f"- {OPENCV_MASK_RCNN_GRAPH}\n"
            f"- {OPENCV_MASK_RCNN_CONFIG}\n"
            "Download them before running mask-rcnn-inpaint."
        )
    if not GT_MANIFEST.exists():
        raise SystemExit(f"Missing GT manifest: {GT_MANIFEST}")
    net = cv2.dnn.readNetFromTensorflow(str(OPENCV_MASK_RCNN_GRAPH), str(OPENCV_MASK_RCNN_CONFIG))
    rows = read_jsonl(GT_MANIFEST)
    if args.max_samples:
        rows = rows[: args.max_samples]
    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = PAPER_ROOT / "experiments" / "mask_rcnn_inpaint" / run_id
    mask_root = run_root / "person_masks"
    bg_root = run_root / "backgrounds_inpainted"
    preview_root = run_root / "background_previews"
    run_root.mkdir(parents=True, exist_ok=False)
    config = {
        "run_id": run_id,
        "method": f"opencv_mask_rcnn_person_mask_plus_opencv_{args.method}",
        "model_graph": str(OPENCV_MASK_RCNN_GRAPH.relative_to(PROJECT_ROOT)),
        "model_config": str(OPENCV_MASK_RCNN_CONFIG.relative_to(PROJECT_ROOT)),
        "max_samples": args.max_samples,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "dilate_px": args.dilate_px,
        "inpaint_radius": args.inpaint_radius,
        "opencv_method": args.method,
        "note": "GT bbox is used only to select the target person detection for controlled comparison.",
    }
    (run_root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    manifest = []
    for row in rows:
        scene_path = PROJECT_ROOT / row["scene_source_path"]
        motion_path = PROJECT_ROOT / row["motion_path"]
        if not scene_path.exists() or not motion_path.exists():
            continue
        data = np.load(motion_path)
        gt_bbox = data["bbox_xyxy"][len(data["bbox_xyxy"]) // 2]
        if not np.isfinite(gt_bbox).all():
            continue
        image = cv2.imread(str(scene_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        mask, det = mask_rcnn_person_mask(net, image, gt_bbox, args.score_threshold, args.mask_threshold)
        if mask is None:
            manifest.append(
                {
                    "run_id": run_id,
                    "sample_id": row["sample_id"],
                    "video_id": row["video_id"],
                    "action_label": row["action_label"],
                    "scene_source_path": row["scene_source_path"],
                    "status": "no_person_mask",
                }
            )
            continue
        if args.dilate_px > 0:
            kernel = np.ones((args.dilate_px, args.dilate_px), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        mask_path = mask_root / f"{row['sample_id']}_mask_rcnn_person.png"
        bg_path = bg_root / f"{row['sample_id']}_mask_rcnn_opencv_{args.method}.jpg"
        preview_path = preview_root / f"{row['sample_id']}_mask_rcnn_opencv_{args.method}.jpg"
        inpaint_with_mask(image, mask, mask_path, bg_path, preview_path, args.inpaint_radius, args.method)
        manifest.append(
            {
                "run_id": run_id,
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "action_label": row["action_label"],
                "scene_source_path": row["scene_source_path"],
                "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
                "background_path": str(bg_path.relative_to(PROJECT_ROOT)),
                "preview_path": str(preview_path.relative_to(PROJECT_ROOT)),
                "status": "ok",
                "detection": det,
                "method": f"opencv_mask_rcnn_person_mask_plus_opencv_{args.method}",
            }
        )
    write_jsonl(run_root / "manifest.jsonl", manifest)
    ok_count = sum(1 for row in manifest if row["status"] == "ok")
    print(f"Wrote {ok_count}/{len(manifest)} Mask R-CNN inpaint samples under {run_root}")


def mask_rcnn_temporal_fill(args: argparse.Namespace) -> None:
    """Fill the target person region from nearby frames, then finish with OpenCV.

    This is useful for mostly static-camera videos. Mask R-CNN identifies the
    person in the target and nearby frames. Pixels in the target person mask are
    copied from nearby frames where those pixels are not covered by a person.
    Remaining holes are completed with OpenCV inpainting.
    """
    ensure_layout()
    if not OPENCV_MASK_RCNN_GRAPH.exists() or not OPENCV_MASK_RCNN_CONFIG.exists():
        raise SystemExit(
            "Missing OpenCV Mask R-CNN files. Expected:\n"
            f"- {OPENCV_MASK_RCNN_GRAPH}\n"
            f"- {OPENCV_MASK_RCNN_CONFIG}"
        )
    if not GT_MANIFEST.exists():
        raise SystemExit(f"Missing GT manifest: {GT_MANIFEST}")
    _, frames_root = find_dataset_dirs()
    net = cv2.dnn.readNetFromTensorflow(str(OPENCV_MASK_RCNN_GRAPH), str(OPENCV_MASK_RCNN_CONFIG))
    rows = read_jsonl(GT_MANIFEST)
    if args.max_samples:
        rows = rows[: args.max_samples]
    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = PAPER_ROOT / "experiments" / "mask_rcnn_temporal_fill" / run_id
    mask_root = run_root / "person_masks"
    bg_root = run_root / "backgrounds_filled"
    preview_root = run_root / "background_previews"
    run_root.mkdir(parents=True, exist_ok=False)
    config = {
        "run_id": run_id,
        "method": "mask_rcnn_temporal_fill_plus_opencv_finish",
        "max_samples": args.max_samples,
        "neighbor_radius": args.neighbor_radius,
        "neighbor_stride": args.neighbor_stride,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "dilate_px": args.dilate_px,
        "inpaint_radius": args.inpaint_radius,
        "opencv_method": args.method,
        "note": "Uses neighboring frames to fill masked target pixels before OpenCV inpaint cleanup.",
    }
    (run_root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    manifest = []
    for row in rows:
        scene_path = PROJECT_ROOT / row["scene_source_path"]
        motion_path = PROJECT_ROOT / row["motion_path"]
        if not scene_path.exists() or not motion_path.exists():
            continue
        image = cv2.imread(str(scene_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        data = np.load(motion_path)
        gt_bbox = data["bbox_xyxy"][len(data["bbox_xyxy"]) // 2]
        if not np.isfinite(gt_bbox).all():
            continue
        target_mask, det = mask_rcnn_person_mask(net, image, gt_bbox, args.score_threshold, args.mask_threshold)
        if target_mask is None:
            manifest.append(
                {
                    "run_id": run_id,
                    "sample_id": row["sample_id"],
                    "video_id": row["video_id"],
                    "action_label": row["action_label"],
                    "scene_source_path": row["scene_source_path"],
                    "status": "no_target_mask",
                }
            )
            continue
        if args.dilate_px > 0:
            kernel = np.ones((args.dilate_px, args.dilate_px), np.uint8)
            target_mask = cv2.dilate(target_mask, kernel, iterations=1)
        video_frames = frame_files_for(frames_root, row["video_id"])
        source_images, source_masks = neighboring_sources(
            net,
            video_frames,
            row["scene_frame_index"],
            gt_bbox,
            args.neighbor_radius,
            args.neighbor_stride,
            args.score_threshold,
            args.mask_threshold,
            args.dilate_px,
        )
        if not source_images:
            manifest.append(
                {
                    "run_id": run_id,
                    "sample_id": row["sample_id"],
                    "video_id": row["video_id"],
                    "action_label": row["action_label"],
                    "scene_source_path": row["scene_source_path"],
                    "status": "no_neighbor_sources",
                    "detection": det,
                }
            )
            continue
        filled, coverage = temporal_fill_image(image, target_mask, source_images, source_masks, args.inpaint_radius, args.method)
        mask_path = mask_root / f"{row['sample_id']}_mask_rcnn_person.png"
        bg_path = bg_root / f"{row['sample_id']}_temporal_fill.jpg"
        preview_path = preview_root / f"{row['sample_id']}_temporal_fill.jpg"
        write_temporal_preview(image, target_mask, filled, mask_path, bg_path, preview_path)
        manifest.append(
            {
                "run_id": run_id,
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "action_label": row["action_label"],
                "scene_source_path": row["scene_source_path"],
                "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
                "background_path": str(bg_path.relative_to(PROJECT_ROOT)),
                "preview_path": str(preview_path.relative_to(PROJECT_ROOT)),
                "status": "ok",
                "neighbor_frames_used": len(source_images),
                "target_mask_temporal_coverage": coverage,
                "detection": det,
                "method": "mask_rcnn_temporal_fill_plus_opencv_finish",
            }
        )
    write_jsonl(run_root / "manifest.jsonl", manifest)
    ok_count = sum(1 for row in manifest if row["status"] == "ok")
    print(f"Wrote {ok_count}/{len(manifest)} Mask R-CNN temporal-fill samples under {run_root}")


def mask_rcnn_lama(args: argparse.Namespace) -> None:
    """Use Mask R-CNN person masks, then LaMa to inpaint the background."""
    ensure_layout()
    if not OPENCV_MASK_RCNN_GRAPH.exists() or not OPENCV_MASK_RCNN_CONFIG.exists():
        raise SystemExit(
            "Missing OpenCV Mask R-CNN files. Expected:\n"
            f"- {OPENCV_MASK_RCNN_GRAPH}\n"
            f"- {OPENCV_MASK_RCNN_CONFIG}"
        )
    if not GT_MANIFEST.exists():
        raise SystemExit(f"Missing GT manifest: {GT_MANIFEST}")
    try:
        import torch
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:
        raise SystemExit(
            "Missing LaMa dependencies. Install simple-lama-inpainting, torch, and torchvision first."
        ) from exc

    net = cv2.dnn.readNetFromTensorflow(str(OPENCV_MASK_RCNN_GRAPH), str(OPENCV_MASK_RCNN_CONFIG))
    device = torch.device(args.device)
    lama = SimpleLama(device=device)
    rows = read_jsonl(GT_MANIFEST)
    if args.max_samples:
        rows = rows[: args.max_samples]
    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = PAPER_ROOT / "experiments" / "mask_rcnn_lama" / run_id
    mask_root = run_root / "person_masks"
    bg_root = run_root / "backgrounds_inpainted"
    preview_root = run_root / "background_previews"
    run_root.mkdir(parents=True, exist_ok=False)
    config = {
        "run_id": run_id,
        "method": "opencv_mask_rcnn_person_mask_plus_lama",
        "model_graph": str(OPENCV_MASK_RCNN_GRAPH.relative_to(PROJECT_ROOT)),
        "model_config": str(OPENCV_MASK_RCNN_CONFIG.relative_to(PROJECT_ROOT)),
        "lama_package": "simple-lama-inpainting",
        "torch_home": os.environ.get("TORCH_HOME"),
        "max_samples": args.max_samples,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "dilate_px": args.dilate_px,
        "device": args.device,
        "note": "GT bbox is used only to select the target person detection; mask comes from Mask R-CNN and pixels are filled by LaMa.",
    }
    (run_root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    manifest = []
    for row in rows:
        scene_path = PROJECT_ROOT / row["scene_source_path"]
        motion_path = PROJECT_ROOT / row["motion_path"]
        if not scene_path.exists() or not motion_path.exists():
            continue
        data = np.load(motion_path)
        gt_bbox = data["bbox_xyxy"][len(data["bbox_xyxy"]) // 2]
        if not np.isfinite(gt_bbox).all():
            continue
        image = cv2.imread(str(scene_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        mask, det = mask_rcnn_person_mask(net, image, gt_bbox, args.score_threshold, args.mask_threshold)
        if mask is None:
            manifest.append(
                {
                    "run_id": run_id,
                    "sample_id": row["sample_id"],
                    "video_id": row["video_id"],
                    "action_label": row["action_label"],
                    "scene_source_path": row["scene_source_path"],
                    "status": "no_person_mask",
                }
            )
            continue
        if args.dilate_px > 0:
            kernel = np.ones((args.dilate_px, args.dilate_px), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        mask_path = mask_root / f"{row['sample_id']}_mask_rcnn_person.png"
        bg_path = bg_root / f"{row['sample_id']}_mask_rcnn_lama.jpg"
        preview_path = preview_root / f"{row['sample_id']}_mask_rcnn_lama.jpg"
        inpaint_with_lama(image, mask, lama, mask_path, bg_path, preview_path)
        manifest.append(
            {
                "run_id": run_id,
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "action_label": row["action_label"],
                "scene_source_path": row["scene_source_path"],
                "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
                "background_path": str(bg_path.relative_to(PROJECT_ROOT)),
                "preview_path": str(preview_path.relative_to(PROJECT_ROOT)),
                "status": "ok",
                "detection": det,
                "method": "opencv_mask_rcnn_person_mask_plus_lama",
            }
        )
    write_jsonl(run_root / "manifest.jsonl", manifest)
    ok_count = sum(1 for row in manifest if row["status"] == "ok")
    print(f"Wrote {ok_count}/{len(manifest)} Mask R-CNN + LaMa samples under {run_root}")


def bbox_lama(args: argparse.Namespace) -> None:
    """Use a padded GT bbox mask, then LaMa to inpaint the background."""
    ensure_layout()
    if not GT_MANIFEST.exists():
        raise SystemExit(f"Missing GT manifest: {GT_MANIFEST}")
    try:
        import torch
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:
        raise SystemExit(
            "Missing LaMa dependencies. Install simple-lama-inpainting, torch, and torchvision first."
        ) from exc

    device = torch.device(args.device)
    lama = SimpleLama(device=device)
    rows = read_jsonl(GT_MANIFEST)
    if args.max_samples:
        rows = rows[: args.max_samples]
    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = PAPER_ROOT / "experiments" / "bbox_lama" / run_id
    mask_root = run_root / "person_masks"
    bg_root = run_root / "backgrounds_inpainted"
    preview_root = run_root / "background_previews"
    run_root.mkdir(parents=True, exist_ok=False)
    config = {
        "run_id": run_id,
        "method": "gt_bbox_mask_plus_lama",
        "lama_package": "simple-lama-inpainting",
        "torch_home": os.environ.get("TORCH_HOME"),
        "max_samples": args.max_samples,
        "mask_pad": args.mask_pad,
        "device": args.device,
        "note": "Uses padded Penn Action/GT motion bbox as a coarse erase region, then fills pixels with LaMa.",
    }
    (run_root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    manifest = []
    for row in rows:
        scene_path = PROJECT_ROOT / row["scene_source_path"]
        motion_path = PROJECT_ROOT / row["motion_path"]
        if not scene_path.exists() or not motion_path.exists():
            continue
        image = cv2.imread(str(scene_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        data = np.load(motion_path)
        bbox = data["bbox_xyxy"][len(data["bbox_xyxy"]) // 2]
        if not np.isfinite(bbox).all():
            continue
        mask = bbox_mask(image.shape[:2], bbox, args.mask_pad)
        mask_path = mask_root / f"{row['sample_id']}_gt_bbox_mask.png"
        bg_path = bg_root / f"{row['sample_id']}_bbox_lama.jpg"
        preview_path = preview_root / f"{row['sample_id']}_bbox_lama.jpg"
        inpaint_with_lama(image, mask, lama, mask_path, bg_path, preview_path)
        manifest.append(
            {
                "run_id": run_id,
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "action_label": row["action_label"],
                "scene_source_path": row["scene_source_path"],
                "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
                "background_path": str(bg_path.relative_to(PROJECT_ROOT)),
                "preview_path": str(preview_path.relative_to(PROJECT_ROOT)),
                "status": "ok",
                "bbox_xyxy": [float(v) for v in bbox],
                "method": "gt_bbox_mask_plus_lama",
            }
        )
    write_jsonl(run_root / "manifest.jsonl", manifest)
    ok_count = sum(1 for row in manifest if row["status"] == "ok")
    print(f"Wrote {ok_count}/{len(manifest)} bbox + LaMa samples under {run_root}")


def build_bbox_lama_dataset(args: argparse.Namespace) -> None:
    """Build a paper-like mini dataset using padded bbox masks and LaMa scenes."""
    ensure_layout()
    if not GT_MANIFEST.exists():
        raise SystemExit(f"Missing GT manifest: {GT_MANIFEST}")
    try:
        import torch
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:
        raise SystemExit(
            "Missing LaMa dependencies. Install simple-lama-inpainting, torch, and torchvision first."
        ) from exc

    rows = read_jsonl(GT_MANIFEST)
    rows = select_dataset_rows(rows, args.max_samples, args.sample_mode)
    run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root = PAPER_MINI_ROOT / run_id
    scene_root = dataset_root / "scenes"
    mask_root = dataset_root / "bbox_masks"
    preview_root = dataset_root / "previews"
    dataset_root.mkdir(parents=True, exist_ok=False)

    device = torch.device(args.device)
    lama = SimpleLama(device=device)
    config = {
        "run_id": run_id,
        "source_manifest": str(GT_MANIFEST.relative_to(PROJECT_ROOT)),
        "method": "gt_bbox_mask_plus_lama_dataset_v1",
        "lama_package": "simple-lama-inpainting",
        "torch_home": os.environ.get("TORCH_HOME"),
        "max_samples": args.max_samples,
        "sample_mode": args.sample_mode,
        "mask_pad": args.mask_pad,
        "device": args.device,
        "note": "Dataset branch v1. Motion/text/split come from the existing Penn Action mini dataset; scene_image_path is replaced by bbox+LaMa background.",
    }
    (dataset_root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    manifest = []
    failures = []
    for row in rows:
        scene_path = PROJECT_ROOT / row["scene_source_path"]
        motion_path = PROJECT_ROOT / row["motion_path"]
        if not scene_path.exists() or not motion_path.exists():
            failures.append({"sample_id": row.get("sample_id"), "reason": "missing_scene_or_motion"})
            continue
        image = cv2.imread(str(scene_path), cv2.IMREAD_COLOR)
        if image is None:
            failures.append({"sample_id": row.get("sample_id"), "reason": "unreadable_scene"})
            continue
        data = np.load(motion_path)
        bbox = data["bbox_xyxy"][len(data["bbox_xyxy"]) // 2]
        if not np.isfinite(bbox).all():
            failures.append({"sample_id": row.get("sample_id"), "reason": "invalid_bbox"})
            continue
        mask = bbox_mask(image.shape[:2], bbox, args.mask_pad)
        sample_id = row["sample_id"]
        mask_path = mask_root / f"{sample_id}_bbox_mask.png"
        scene_out = scene_root / f"{sample_id}_bbox_lama.jpg"
        preview_path = preview_root / f"{sample_id}_bbox_lama.jpg"
        inpaint_with_lama(image, mask, lama, mask_path, scene_out, preview_path)

        out_row = dict(row)
        out_row["original_scene_image_path"] = row["scene_image_path"]
        out_row["scene_image_path"] = str(scene_out.relative_to(PROJECT_ROOT))
        out_row["bbox_mask_path"] = str(mask_path.relative_to(PROJECT_ROOT))
        out_row["background_preview_path"] = str(preview_path.relative_to(PROJECT_ROOT))
        out_row["background_method"] = "gt_bbox_mask_plus_lama"
        out_row["background_config"] = {"mask_pad": args.mask_pad}
        out_row["paper_like_branch"] = "bbox_lama_v1"
        manifest.append(out_row)

    write_jsonl(dataset_root / "manifest_filtered.jsonl", manifest)
    write_jsonl(dataset_root / "failures.jsonl", failures)
    splits = build_split_index(manifest)
    (dataset_root / "splits.json").write_text(json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8")
    write_bbox_lama_dataset_report(dataset_root, manifest, failures, config)
    print(f"Wrote {len(manifest)} bbox+LaMa dataset samples under {dataset_root}")
    if failures:
        print(f"Skipped {len(failures)} samples; see {dataset_root / 'failures.jsonl'}")


def select_dataset_rows(
    rows: list[dict[str, Any]], max_samples: int | None, sample_mode: str
) -> list[dict[str, Any]]:
    if max_samples is None:
        return rows
    if sample_mode == "first":
        return rows[:max_samples]
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_action[row.get("action_label", "unknown")].append(row)
    selected = []
    actions = sorted(by_action)
    cursor = 0
    while len(selected) < max_samples and actions:
        action = actions[cursor % len(actions)]
        bucket = by_action[action]
        if bucket:
            selected.append(bucket.pop(0))
        actions = [name for name in actions if by_action[name]]
        cursor += 1
    return selected


def neighboring_sources(
    net: cv2.dnn_Net,
    frame_paths: list[Path],
    center_index: int,
    target_bbox: np.ndarray,
    radius: int,
    stride: int,
    score_threshold: float,
    mask_threshold: float,
    dilate_px: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    source_images = []
    source_masks = []
    if not frame_paths:
        return source_images, source_masks
    indices = []
    for offset in range(-radius, radius + 1, max(stride, 1)):
        if offset == 0:
            continue
        idx = center_index + offset
        if 0 <= idx < len(frame_paths):
            indices.append(idx)
    for idx in indices:
        image = cv2.imread(str(frame_paths[idx]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        mask, _ = mask_rcnn_person_mask(net, image, target_bbox, score_threshold, mask_threshold)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        elif dilate_px > 0:
            kernel = np.ones((dilate_px, dilate_px), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        source_images.append(image)
        source_masks.append(mask)
    return source_images, source_masks


def temporal_fill_image(
    target: np.ndarray,
    target_mask: np.ndarray,
    source_images: list[np.ndarray],
    source_masks: list[np.ndarray],
    radius: float,
    method: str,
) -> tuple[np.ndarray, float]:
    hole = target_mask > 0
    filled = target.copy()
    values = []
    valid_masks = []
    for image, mask in zip(source_images, source_masks):
        if image.shape != target.shape:
            continue
        valid = (mask == 0) & hole
        values.append(image.astype(np.float32))
        valid_masks.append(valid)
    if not values:
        return inpaint_array(target, target_mask, radius, method), 0.0
    stack = np.stack(values, axis=0)
    valid_stack = np.stack(valid_masks, axis=0)
    coverage_pixels = valid_stack.any(axis=0)
    for channel in range(3):
        channel_values = stack[:, :, :, channel]
        channel_values = np.where(valid_stack, channel_values, np.nan)
        median = np.nanmedian(channel_values, axis=0)
        use = hole & np.isfinite(median)
        filled[:, :, channel][use] = median[use].astype(np.uint8)
    remaining_mask = np.zeros(target_mask.shape, dtype=np.uint8)
    remaining_mask[hole & ~coverage_pixels] = 255
    if remaining_mask.any():
        filled = inpaint_array(filled, remaining_mask, radius, method)
    coverage = float(coverage_pixels[hole].mean()) if hole.any() else 1.0
    return filled, coverage


def inpaint_array(image: np.ndarray, mask: np.ndarray, radius: float, method: str) -> np.ndarray:
    cv2_method = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(image, mask, radius, cv2_method)


def write_temporal_preview(
    image: np.ndarray,
    mask: np.ndarray,
    filled: np.ndarray,
    mask_path: Path,
    bg_path: Path,
    preview_path: Path,
) -> None:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    bg_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(bg_path), filled)
    preview = np.concatenate([image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), filled], axis=1)
    cv2.imwrite(str(preview_path), preview)


def inpaint_with_lama(
    image: np.ndarray,
    mask: np.ndarray,
    lama: Any,
    mask_path: Path,
    bg_path: Path,
    preview_path: Path,
) -> None:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    bg_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inpainted_rgb = np.array(lama(Image.fromarray(image_rgb), Image.fromarray(mask)))
    inpainted_rgb = inpainted_rgb[: image.shape[0], : image.shape[1]]
    inpainted = cv2.cvtColor(inpainted_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(bg_path), inpainted)
    preview = np.concatenate([image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), inpainted], axis=1)
    cv2.imwrite(str(preview_path), preview)


def bbox_mask(shape: tuple[int, int], bbox: np.ndarray, pad: int) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = bbox.astype(float)
    x1 = max(0, int(math.floor(x1)) - pad)
    y1 = max(0, int(math.floor(y1)) - pad)
    x2 = min(width - 1, int(math.ceil(x2)) + pad)
    y2 = min(height - 1, int(math.ceil(y2)) + pad)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1 : y2 + 1, x1 : x2 + 1] = 255
    return mask


def build_split_index(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split = row.get("split", "train")
        splits.setdefault(split, []).append(row["sample_id"])
    return splits


def write_bbox_lama_dataset_report(
    dataset_root: Path,
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    split_counts = Counter(row.get("split", "unknown") for row in rows)
    action_counts = Counter(row.get("action_label", "unknown") for row in rows)
    video_count = len({row.get("video_id") for row in rows})
    mask_pad = config["mask_pad"]
    lines = [
        "# Paper-like Mini Dataset: bbox + LaMa",
        "",
        "## Summary",
        "",
        f"- samples: {len(rows)}",
        f"- videos: {video_count}",
        f"- failures: {len(failures)}",
        f"- scene method: padded Penn Action bbox mask + LaMa",
        f"- bbox mask pad: {mask_pad}px",
        f"- motion source: existing Penn Action 2D keypoint motion files",
        f"- text source: existing action-label prompt templates",
        "",
        "## Splits",
        "",
    ]
    for split, count in sorted(split_counts.items()):
        lines.append(f"- {split}: {count}")
    lines.extend(["", "## Actions", ""])
    for action, count in sorted(action_counts.items()):
        lines.append(f"- {action}: {count}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- manifest: `manifest_filtered.jsonl`",
            "- splits: `splits.json`",
            "- scenes: `scenes/`",
            "- bbox masks: `bbox_masks/`",
            "- previews: `previews/`",
            "- failures: `failures.jsonl`",
            "",
            "## Notes",
            "",
            "- This is the selected background-removal branch for dataset v1.",
            "- It intentionally keeps GT/Penn Action motion for the first trainable dataset, while replacing original scene frames with human-removed LaMa backgrounds.",
            "- Preview images are three panels: original frame, bbox mask, LaMa background.",
        ]
    )
    (dataset_root / "dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def mask_rcnn_person_mask(
    net: cv2.dnn_Net,
    image: np.ndarray,
    target_bbox: np.ndarray,
    score_threshold: float,
    mask_threshold: float,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    best = None
    target = target_bbox.astype(float)
    detections = boxes[0, 0]
    for i, det in enumerate(detections):
        class_id = int(det[1])
        score = float(det[2])
        # TensorFlow COCO Mask R-CNN as loaded by OpenCV commonly reports
        # person as class 0. Some exported graphs use the COCO id 1. Accept
        # both and let bbox IoU with the target person choose the instance.
        if class_id not in {0, 1} or score < score_threshold:
            continue
        box = np.array(
            [det[3] * width, det[4] * height, det[5] * width, det[6] * height],
            dtype=np.float32,
        )
        box = clip_bbox(box, width, height)
        if box_area(box) <= 0:
            continue
        iou = bbox_iou(box, target)
        rank = (iou, score)
        if best is None or rank > best["rank"]:
            best = {
                "rank": rank,
                "box": box,
                "score": score,
                "mask_index": i,
                "class_id": class_id,
                "iou_with_gt_bbox": iou,
            }
    if best is None:
        return None, None
    box = best["box"].astype(int)
    x1, y1, x2, y2 = box.tolist()
    mask_channels = masks.shape[1]
    mask_class = min(int(best["class_id"]), mask_channels - 1)
    mask = masks[best["mask_index"], mask_class]
    mask = cv2.resize(mask, (max(x2 - x1 + 1, 1), max(y2 - y1 + 1, 1)))
    mask = (mask > mask_threshold).astype(np.uint8) * 255
    full = np.zeros((height, width), dtype=np.uint8)
    full[y1 : y2 + 1, x1 : x2 + 1] = mask[: y2 - y1 + 1, : x2 - x1 + 1]
    return full, {
        "bbox_xyxy": [float(v) for v in best["box"]],
        "class_id": int(best["class_id"]),
        "score": best["score"],
        "iou_with_gt_bbox": best["iou_with_gt_bbox"],
    }


def clip_bbox(box: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = box.copy()
    clipped[0] = np.clip(clipped[0], 0, width - 1)
    clipped[1] = np.clip(clipped[1], 0, height - 1)
    clipped[2] = np.clip(clipped[2], 0, width - 1)
    clipped[3] = np.clip(clipped[3], 0, height - 1)
    return clipped


def box_area(box: np.ndarray) -> float:
    return max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def inpaint_with_bbox(
    scene_path: Path,
    bbox: np.ndarray,
    mask_path: Path,
    bg_path: Path,
    preview_path: Path,
    pad: int,
    radius: float,
    method: str,
) -> None:
    image = cv2.imread(str(scene_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox.astype(float)
    x1 = max(0, int(math.floor(x1)) - pad)
    y1 = max(0, int(math.floor(y1)) - pad)
    x2 = min(width - 1, int(math.ceil(x2)) + pad)
    y2 = min(height - 1, int(math.ceil(y2)) + pad)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1 : y2 + 1, x1 : x2 + 1] = 255
    cv2_method = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    inpainted = cv2.inpaint(image, mask, radius, cv2_method)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    bg_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(bg_path), inpainted)
    preview = np.concatenate([image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), inpainted], axis=1)
    cv2.imwrite(str(preview_path), preview)


def inpaint_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    mask_path: Path,
    bg_path: Path,
    preview_path: Path,
    radius: float,
    method: str,
) -> None:
    cv2_method = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    inpainted = cv2.inpaint(image, mask, radius, cv2_method)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    bg_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(bg_path), inpainted)
    preview = np.concatenate([image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), inpainted], axis=1)
    cv2.imwrite(str(preview_path), preview)


def compare_gt_vs_pseudo(args: argparse.Namespace) -> None:
    ensure_layout()
    gt_path = Path(args.gt_manifest)
    pseudo_path = Path(args.pseudo_manifest)
    if not gt_path.exists():
        raise SystemExit(f"Missing GT manifest: {gt_path}")
    if not pseudo_path.exists():
        raise SystemExit(
            f"Missing pseudo manifest: {pseudo_path}. Create a paper-like pseudo dataset first."
        )
    gt_rows = read_jsonl(gt_path)
    pseudo_rows = read_jsonl(pseudo_path)
    gt_by_key = {(r["video_id"], r["frame_start"], r["frame_end"]): r for r in gt_rows}
    pseudo_by_key = {(r["video_id"], r["frame_start"], r["frame_end"]): r for r in pseudo_rows}
    keys = sorted(set(gt_by_key) & set(pseudo_by_key))
    metrics = []
    for key in keys:
        gt = gt_by_key[key]
        pseudo = pseudo_by_key[key]
        gt_motion = np.load(PROJECT_ROOT / gt["motion_path"])
        pseudo_motion = np.load(PROJECT_ROOT / pseudo["motion_path"])
        metrics.append(sample_pose_metrics(gt, pseudo, gt_motion, pseudo_motion))
    write_comparison_report(metrics, len(gt_rows), len(pseudo_rows), len(keys))
    if args.previews:
        render_comparison_previews(metrics[: args.previews], gt_by_key, pseudo_by_key)


def sample_pose_metrics(
    gt_row: dict[str, Any],
    pseudo_row: dict[str, Any],
    gt_motion: np.lib.npyio.NpzFile,
    pseudo_motion: np.lib.npyio.NpzFile,
) -> dict[str, Any]:
    gt = gt_motion["keypoints_2d_px"].astype(np.float32)
    pseudo = pseudo_motion["keypoints_2d_px"].astype(np.float32)
    gt_vis = gt_motion["visibility"].astype(bool)
    pseudo_vis = pseudo_motion["visibility"].astype(bool)
    joint_count = min(gt.shape[1], pseudo.shape[1])
    frame_count = min(gt.shape[0], pseudo.shape[0])
    gt = gt[:frame_count, :joint_count]
    pseudo = pseudo[:frame_count, :joint_count]
    vis = gt_vis[:frame_count, :joint_count] & pseudo_vis[:frame_count, :joint_count]
    diff = np.linalg.norm(gt - pseudo, axis=-1)
    valid_diff = diff[vis]
    bbox = gt_motion["bbox_xyxy"][:frame_count]
    bbox_wh = np.maximum(bbox[:, 2:4] - bbox[:, 0:2], 1.0)
    norm = np.maximum(bbox_wh[:, 0], bbox_wh[:, 1])[:, None]
    pck = float(((diff / norm) < 0.2)[vis].mean()) if valid_diff.size else 0.0
    gt_root = gt_motion["root_xy"][:frame_count]
    pseudo_root = pseudo_motion["root_xy"][:frame_count]
    root_valid = np.isfinite(gt_root).all(axis=1) & np.isfinite(pseudo_root).all(axis=1)
    return {
        "sample_id": gt_row["sample_id"],
        "video_id": gt_row["video_id"],
        "action_label": gt_row["action_label"],
        "split": gt_row.get("split"),
        "mean_keypoint_l2_px": float(valid_diff.mean()) if valid_diff.size else None,
        "median_keypoint_l2_px": float(np.median(valid_diff)) if valid_diff.size else None,
        "pck_0_2_bbox": pck,
        "gt_missing_ratio": 1.0 - float(gt_vis[:frame_count, :joint_count].mean()),
        "pseudo_missing_ratio": 1.0 - float(pseudo_vis[:frame_count, :joint_count].mean()),
        "root_l2_px": float(np.linalg.norm(gt_root[root_valid] - pseudo_root[root_valid], axis=1).mean())
        if root_valid.any()
        else None,
    }


def write_comparison_report(metrics: list[dict[str, Any]], gt_total: int, pseudo_total: int, matched: int) -> None:
    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    metrics_path = COMPARISON_ROOT / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    by_action = defaultdict(list)
    for row in metrics:
        by_action[row["action_label"]].append(row)
    lines = [
        "# GT vs Pseudo Comparison",
        "",
        f"- GT samples: {gt_total}",
        f"- pseudo samples: {pseudo_total}",
        f"- matched samples: {matched}",
        "",
        "## Overall",
        "",
        metric_summary(metrics),
        "",
        "## By Action",
        "",
    ]
    for action, rows in sorted(by_action.items()):
        lines.append(f"### {action}")
        lines.append("")
        lines.append(metric_summary(rows))
        lines.append("")
    report_path = COMPARISON_ROOT / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {report_path}")


def metric_summary(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "- no matched rows"
    keys = ["mean_keypoint_l2_px", "median_keypoint_l2_px", "pck_0_2_bbox", "root_l2_px"]
    out = [f"- samples: {len(rows)}"]
    for key in keys:
        values = [r[key] for r in rows if r[key] is not None]
        if values:
            out.append(f"- {key}: {float(np.mean(values)):.4f}")
    return "\n".join(out)


def render_comparison_previews(
    metric_rows: list[dict[str, Any]],
    gt_by_key: dict[tuple[str, int, int], dict[str, Any]],
    pseudo_by_key: dict[tuple[str, int, int], dict[str, Any]],
) -> None:
    # This function intentionally stays conservative. The preview contract is
    # finalized once a real pseudo pose extractor is selected.
    (COMPARISON_ROOT / "previews").mkdir(parents=True, exist_ok=True)
    print("Comparison preview rendering is reserved for the selected pseudo pose joint format.")


def status(args: argparse.Namespace) -> None:
    ensure_layout()
    paths = {
        "frame_quality": PAPER_ROOT / "frame_quality" / "frame_quality_manifest.jsonl",
        "video_quality": PAPER_ROOT / "frame_quality" / "video_quality_summary.jsonl",
        "gt_manifest": GT_MANIFEST,
        "paper_like_manifest": PAPER_MINI_ROOT / "manifest_filtered.jsonl",
        "comparison_report": COMPARISON_ROOT / "report.md",
    }
    for name, path in paths.items():
        exists = path.exists()
        suffix = ""
        if exists and path.is_file():
            suffix = f" ({path.stat().st_size} bytes)"
        print(f"{name}: {'ok' if exists else 'missing'} {path}{suffix}")


def tool_check(args: argparse.Namespace) -> None:
    import importlib.util

    tools = ["cv2", "PIL", "numpy", "scipy", "torch", "torchvision", "ultralytics", "mediapipe", "detectron2"]
    for name in tools:
        print(f"{name}: {bool(importlib.util.find_spec(name))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ensure-layout")
    sub.add_parser("status")
    sub.add_parser("tool-check")
    fq = sub.add_parser("frame-quality")
    add_selection_args(fq)
    baseline = sub.add_parser("gt-mask-inpaint-baseline")
    baseline.add_argument("--max-samples", type=int, default=32)
    baseline.add_argument("--mask-pad", type=int, default=16)
    baseline.add_argument("--inpaint-radius", type=float, default=5.0)
    baseline.add_argument("--run-name", default=None)
    baseline.add_argument("--method", choices=["telea", "ns"], default="telea")
    mrcnn = sub.add_parser("mask-rcnn-inpaint")
    mrcnn.add_argument("--max-samples", type=int, default=5)
    mrcnn.add_argument("--run-name", default=None)
    mrcnn.add_argument("--score-threshold", type=float, default=0.5)
    mrcnn.add_argument("--mask-threshold", type=float, default=0.5)
    mrcnn.add_argument("--dilate-px", type=int, default=7)
    mrcnn.add_argument("--inpaint-radius", type=float, default=5.0)
    mrcnn.add_argument("--method", choices=["telea", "ns"], default="telea")
    temporal = sub.add_parser("mask-rcnn-temporal-fill")
    temporal.add_argument("--max-samples", type=int, default=5)
    temporal.add_argument("--run-name", default=None)
    temporal.add_argument("--neighbor-radius", type=int, default=24)
    temporal.add_argument("--neighbor-stride", type=int, default=4)
    temporal.add_argument("--score-threshold", type=float, default=0.5)
    temporal.add_argument("--mask-threshold", type=float, default=0.5)
    temporal.add_argument("--dilate-px", type=int, default=7)
    temporal.add_argument("--inpaint-radius", type=float, default=5.0)
    temporal.add_argument("--method", choices=["telea", "ns"], default="telea")
    lama = sub.add_parser("mask-rcnn-lama")
    lama.add_argument("--max-samples", type=int, default=5)
    lama.add_argument("--run-name", default=None)
    lama.add_argument("--score-threshold", type=float, default=0.5)
    lama.add_argument("--mask-threshold", type=float, default=0.5)
    lama.add_argument("--dilate-px", type=int, default=7)
    lama.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    blama = sub.add_parser("bbox-lama")
    blama.add_argument("--max-samples", type=int, default=5)
    blama.add_argument("--run-name", default=None)
    blama.add_argument("--mask-pad", type=int, default=32)
    blama.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ds = sub.add_parser("build-bbox-lama-dataset")
    ds.add_argument("--max-samples", type=int, default=None)
    ds.add_argument("--run-name", default=None)
    ds.add_argument("--mask-pad", type=int, default=32)
    ds.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ds.add_argument("--sample-mode", choices=["first", "balanced"], default="first")
    cmp_parser = sub.add_parser("compare-gt-pseudo")
    cmp_parser.add_argument("--gt-manifest", default=str(GT_MANIFEST))
    cmp_parser.add_argument("--pseudo-manifest", default=str(PAPER_MINI_ROOT / "manifest_filtered.jsonl"))
    cmp_parser.add_argument("--previews", type=int, default=0)
    return parser.parse_args()


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--video-ids", nargs="*", default=None)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", str(PROJECT_DATA / "models" / "torch"))
    os.environ.setdefault("HF_HOME", str(PROJECT_DATA / "models" / "huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DATA / "models" / "cache"))
    ensure_layout()
    commands = {
        "ensure-layout": lambda _: print(PAPER_ROOT),
        "status": status,
        "tool-check": tool_check,
        "frame-quality": frame_quality,
        "gt-mask-inpaint-baseline": gt_mask_inpaint_baseline,
        "mask-rcnn-inpaint": mask_rcnn_inpaint,
        "mask-rcnn-temporal-fill": mask_rcnn_temporal_fill,
        "mask-rcnn-lama": mask_rcnn_lama,
        "bbox-lama": bbox_lama,
        "build-bbox-lama-dataset": build_bbox_lama_dataset,
        "compare-gt-pseudo": compare_gt_vs_pseudo,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
