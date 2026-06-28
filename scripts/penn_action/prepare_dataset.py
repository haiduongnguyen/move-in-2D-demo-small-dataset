#!/usr/bin/env python3
"""Prepare a mini Move-in-2D style dataset from Penn Action.

The pipeline intentionally mirrors the paper's dataset flow at a smaller scale:
index all raw sequences first, compute quality signals, then filter into a clean
training dataset. Heavy artifacts live under project_data/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DATA = PROJECT_ROOT / "project_data"
DATASET_ROOT = PROJECT_DATA / "datasets" / "penn_action"
RAW_ROOT = DATASET_ROOT / "raw"
ARCHIVE_PATH = RAW_ROOT / "Penn_Action.tar.gz"
EXTRACT_ROOT = RAW_ROOT / "Penn_Action"
PROCESSED_ROOT = DATASET_ROOT / "processed"
MINI_ROOT = DATASET_ROOT / "mini_move_in_2d"
LOG_ROOT = PROJECT_DATA / "logs"
EXPECTED_ARCHIVE_BYTES = 3_235_203_923
DOWNLOAD_URL = "https://www.cis.upenn.edu/~kostas/Penn_Action.tar.gz"

ACTION_TEMPLATES = {
    "baseball_pitch": "a person throws a baseball pitch",
    "baseball_swing": "a person swings a baseball bat",
    "bench_press": "a person performs a bench press",
    "bowl": "a person bowls a ball",
    "clean_and_jerk": "a person performs a clean and jerk",
    "golf_swing": "a person swings a golf club",
    "jumping_jacks": "a person does jumping jacks",
    "jump_rope": "a person jumps rope",
    "pullup": "a person does pull-ups",
    "pushup": "a person does push-ups",
    "situp": "a person does sit-ups",
    "squat": "a person performs squats",
    "strum_guitar": "a person strums a guitar",
    "tennis_forehand": "a person hits a tennis forehand",
    "tennis_serve": "a person serves a tennis ball",
}

PENN_ACTION_EDGES = [
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


def loadmat(path: Path) -> dict[str, Any]:
    try:
        from scipy.io import loadmat as scipy_loadmat
    except ImportError as exc:  # pragma: no cover - exercised by environment setup
        raise SystemExit(
            "Missing scipy. Install dependencies with: python3 -m pip install -r requirements.txt"
        ) from exc
    return scipy_loadmat(path)


def ensure_layout() -> None:
    for path in [
        RAW_ROOT,
        PROCESSED_ROOT / "index",
        PROCESSED_ROOT / "scenes",
        PROCESSED_ROOT / "motions",
        PROCESSED_ROOT / "previews",
        MINI_ROOT,
        LOG_ROOT / "download",
        LOG_ROOT / "preprocess",
        LOG_ROOT / "dataset_report",
        PROJECT_DATA / "models" / "torch",
        PROJECT_DATA / "models" / "huggingface",
        PROJECT_DATA / "models" / "opencv",
        PROJECT_DATA / "models" / "move_in_2d_aux",
        PROJECT_DATA / "models" / "cache",
        DATASET_ROOT / "debug_runs",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mat_scalar(value: Any, default: Any = None) -> Any:
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    item = arr.squeeze()
    if item.shape == ():
        item = item.item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    if isinstance(item, np.ndarray):
        if item.dtype.kind in {"U", "S"}:
            return "".join(item.astype(str).ravel()).strip()
        if item.size == 1:
            return item.item()
    return item


def normalize_action(action: Any) -> str:
    text = str(action).strip().lower()
    text = text.replace(" ", "_").replace("-", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def prompt_for_action(action_label: str) -> str:
    return ACTION_TEMPLATES.get(action_label, f"a person performs {action_label.replace('_', ' ')}")


def find_dataset_dirs() -> tuple[Path, Path]:
    candidates = []
    if EXTRACT_ROOT.exists():
        candidates.append(EXTRACT_ROOT)
    candidates.extend(RAW_ROOT.glob("*/"))
    for root in candidates:
        labels = root / "labels"
        frames = root / "frames"
        if labels.is_dir() and frames.is_dir():
            return labels, frames
    raise FileNotFoundError(
        f"Could not find Penn Action labels/frames under {RAW_ROOT}. "
        "Run the extract command after the archive finishes downloading."
    )


def frame_files_for(frames_root: Path, video_id: str) -> list[Path]:
    frame_dir = frames_root / video_id
    files: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(frame_dir.glob(pattern))
    return sorted(files)


def image_size(frame_files: list[Path], mat: dict[str, Any]) -> tuple[int | None, int | None]:
    dims = mat.get("dimensions")
    if dims is not None:
        arr = np.asarray(dims).astype(float).squeeze()
        if arr.size >= 2:
            height, width = int(arr[0]), int(arr[1])
            return width, height
    if frame_files:
        with Image.open(frame_files[0]) as img:
            return img.size
    return None, None


def verify_archive(args: argparse.Namespace) -> None:
    ensure_layout()
    if not ARCHIVE_PATH.exists():
        raise SystemExit(f"Archive not found: {ARCHIVE_PATH}")
    size = ARCHIVE_PATH.stat().st_size
    status = "complete" if size == EXPECTED_ARCHIVE_BYTES else "incomplete"
    print(f"{ARCHIVE_PATH}")
    print(f"size={size} expected={EXPECTED_ARCHIVE_BYTES} status={status}")
    if size > EXPECTED_ARCHIVE_BYTES:
        raise SystemExit("Archive is larger than expected; inspect before extracting.")
    if args.require_complete and size != EXPECTED_ARCHIVE_BYTES:
        raise SystemExit("Archive is not complete yet.")


def download(args: argparse.Namespace) -> None:
    ensure_layout()
    log_path = LOG_ROOT / "download" / "download.log"
    pid_path = LOG_ROOT / "download" / "download.pid"
    cmd = [
        "wget",
        "-c",
        "--progress=dot:giga",
        "-O",
        str(ARCHIVE_PATH),
        DOWNLOAD_URL,
        "-o",
        str(log_path),
    ]
    if args.foreground:
        subprocess.run(cmd, check=True)
        return
    proc = subprocess.Popen(cmd, start_new_session=True)
    pid_path.write_text(str(proc.pid) + "\n", encoding="utf-8")
    print(f"Started download pid={proc.pid}")
    print(f"log={log_path}")


def extract(args: argparse.Namespace) -> None:
    ensure_layout()
    verify_archive(argparse.Namespace(require_complete=True))
    if EXTRACT_ROOT.exists() and any(EXTRACT_ROOT.iterdir()) and not args.force:
        print(f"Extracted dataset already exists: {EXTRACT_ROOT}")
        return
    tmp_dir = RAW_ROOT / "_extract_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        safe_extract(tar, tmp_dir)
    roots = [p for p in tmp_dir.iterdir() if p.is_dir()]
    if len(roots) == 1:
        if EXTRACT_ROOT.exists():
            shutil.rmtree(EXTRACT_ROOT)
        roots[0].rename(EXTRACT_ROOT)
        shutil.rmtree(tmp_dir)
    else:
        if EXTRACT_ROOT.exists():
            shutil.rmtree(EXTRACT_ROOT)
        tmp_dir.rename(EXTRACT_ROOT)
    print(f"Extracted to {EXTRACT_ROOT}")


def safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    root = path.resolve()
    for member in tar.getmembers():
        target = (path / member.name).resolve()
        if root not in [target, *target.parents]:
            raise RuntimeError(f"Unsafe path in archive: {member.name}")
    tar.extractall(path)


def build_raw_manifest(args: argparse.Namespace) -> None:
    ensure_layout()
    labels_root, frames_root = find_dataset_dirs()
    rows = []
    for label_path in sorted(labels_root.glob("*.mat")):
        video_id = label_path.stem
        mat = loadmat(label_path)
        frames = frame_files_for(frames_root, video_id)
        action = normalize_action(mat_scalar(mat.get("action"), "unknown"))
        nframes = int(mat_scalar(mat.get("nframes"), len(frames)) or len(frames))
        width, height = image_size(frames, mat)
        train_value = mat_scalar(mat.get("train"), None)
        rows.append(
            {
                "video_id": video_id,
                "label_path": str(label_path.relative_to(PROJECT_ROOT)),
                "frames_dir": str((frames_root / video_id).relative_to(PROJECT_ROOT)),
                "num_frame_files": len(frames),
                "nframes": nframes,
                "image_width": width,
                "image_height": height,
                "action_label": action,
                "text_prompt": prompt_for_action(action),
                "penn_train_flag": None if train_value is None else int(train_value),
            }
        )
    out = PROCESSED_ROOT / "index" / "manifest_raw.jsonl"
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} raw rows to {out}")


def load_motion(mat: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "x" not in mat or "y" not in mat:
        raise ValueError("Annotation is missing x/y joint arrays.")
    x = np.asarray(mat["x"], dtype=np.float32)
    y = np.asarray(mat["y"], dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected 2D x/y arrays, got {x.shape} and {y.shape}.")
    keypoints = np.stack([x, y], axis=-1)
    vis = mat.get("visibility")
    if vis is None:
        visibility = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    else:
        visibility = np.asarray(vis).astype(bool)
    if visibility.shape != x.shape:
        visibility = np.broadcast_to(visibility, x.shape).copy()
    return keypoints, visibility


def bbox_from_keypoints(kpts: np.ndarray, vis: np.ndarray) -> np.ndarray:
    bboxes = np.zeros((kpts.shape[0], 4), dtype=np.float32)
    for i in range(kpts.shape[0]):
        pts = kpts[i][vis[i]]
        if len(pts) == 0:
            bboxes[i] = np.nan
        else:
            bboxes[i] = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
    return bboxes


def visibility_masked_keypoints(kpts: np.ndarray, vis: np.ndarray) -> np.ndarray:
    masked = kpts.astype(np.float32).copy()
    masked[~vis] = np.nan
    return masked


def root_from_visible_keypoints(kpts: np.ndarray, vis: np.ndarray) -> np.ndarray:
    coords = kpts[:, :, :2].astype(np.float32)
    visible = vis[:, :, None].astype(np.float32)
    counts = visible.sum(axis=1)
    sums = (coords * visible).sum(axis=1)
    root = np.full((kpts.shape[0], 2), np.nan, dtype=np.float32)
    valid = counts[:, 0] > 0
    root[valid] = sums[valid] / counts[valid]
    return root


def quality_stats(kpts: np.ndarray, vis: np.ndarray, bboxes: np.ndarray, width: int, height: int) -> dict[str, Any]:
    missing_ratio = 1.0 - float(vis.mean()) if vis.size else 1.0
    root = root_from_visible_keypoints(kpts, vis)
    valid_root = np.isfinite(root).all(axis=1)
    if valid_root.any():
        displacement = float(np.linalg.norm(root[valid_root][-1] - root[valid_root][0]))
    else:
        displacement = 0.0
    bbox_wh = bboxes[:, 2:4] - bboxes[:, 0:2]
    bbox_area = bbox_wh[:, 0] * bbox_wh[:, 1]
    median_bbox_area = float(np.nanmedian(bbox_area)) if np.isfinite(bbox_area).any() else 0.0
    frame_area = max(float(width * height), 1.0)
    out = (
        (kpts[..., 0] < 0)
        | (kpts[..., 0] >= width)
        | (kpts[..., 1] < 0)
        | (kpts[..., 1] >= height)
    ) & vis
    out_of_frame_ratio = float(out.sum() / max(vis.sum(), 1))
    adjacent_valid = valid_root[1:] & valid_root[:-1]
    jumps = np.linalg.norm(np.diff(root, axis=0), axis=1)
    jumps = jumps[adjacent_valid]
    pose_jump_px_p95 = float(np.percentile(jumps, 95)) if jumps.size else 0.0
    return {
        "missing_joint_ratio": missing_ratio,
        "root_displacement_px": displacement,
        "median_bbox_area_ratio": median_bbox_area / frame_area,
        "out_of_frame_ratio": out_of_frame_ratio,
        "pose_jump_px_p95": pose_jump_px_p95,
    }


def resize_frame_indices(start: int, length: int, target_len: int) -> np.ndarray:
    return np.linspace(start, start + length - 1, target_len).round().astype(int)


def copy_scene(frame_path: Path, out_path: Path, size: int | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(frame_path) as img:
        img = img.convert("RGB")
        if size:
            img.thumbnail((size, size))
        img.save(out_path, quality=92)


def process(args: argparse.Namespace) -> None:
    ensure_layout()
    labels_root, frames_root = find_dataset_dirs()
    raw_manifest = PROCESSED_ROOT / "index" / "manifest_raw.jsonl"
    if not raw_manifest.exists():
        build_raw_manifest(args)
    raw_rows = read_jsonl(raw_manifest)
    process_rows(raw_rows, args, PROCESSED_ROOT, Path("project_data") / "datasets" / "penn_action" / "processed")


def process_rows(
    raw_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    output_root: Path,
    output_rel_root: Path,
) -> list[dict[str, Any]]:
    labels_root, frames_root = find_dataset_dirs()
    processed_rows = []
    action_to_id = {a: i for i, a in enumerate(sorted({r["action_label"] for r in raw_rows}))}
    stride = args.stride or args.window_size

    for row in raw_rows:
        video_id = row["video_id"]
        label_path = PROJECT_ROOT / row["label_path"]
        frames = frame_files_for(frames_root, video_id)
        if not frames:
            processed_rows.append({**row, "processed": False, "reject_reason": "missing_frames"})
            continue
        try:
            mat = loadmat(label_path)
            kpts, vis = load_motion(mat)
        except Exception as exc:
            processed_rows.append({**row, "processed": False, "reject_reason": f"bad_annotation:{exc}"})
            continue

        n = min(len(kpts), len(frames), int(row["nframes"] or len(kpts)))
        if n < args.min_source_frames:
            processed_rows.append({**row, "processed": False, "reject_reason": "too_short"})
            continue
        width = int(row["image_width"] or 1)
        height = int(row["image_height"] or 1)
        source_window = min(args.window_size, n)
        window_starts = list(range(0, n - source_window + 1, stride))
        if (n - source_window) not in window_starts:
            window_starts.append(n - source_window)

        for window_idx, start in enumerate(window_starts):
            indices = resize_frame_indices(start, source_window, args.target_frames)
            win_kpts = kpts[indices]
            win_vis = vis[indices]
            bboxes = bbox_from_keypoints(win_kpts, win_vis)
            stats = quality_stats(win_kpts, win_vis, bboxes, width, height)
            norm = win_kpts.copy()
            norm[..., 0] = norm[..., 0] / max(width, 1)
            norm[..., 1] = norm[..., 1] / max(height, 1)
            root = root_from_visible_keypoints(win_kpts, win_vis)
            sample_id = f"{video_id}_{window_idx:03d}"
            motion_rel = output_rel_root / "motions" / f"{sample_id}.npz"
            motion_path = PROJECT_ROOT / motion_rel
            motion_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                motion_path,
                keypoints_2d_norm=norm.astype(np.float32),
                keypoints_2d_px=win_kpts.astype(np.float32),
                visibility=win_vis.astype(np.uint8),
                bbox_xyxy=bboxes.astype(np.float32),
                root_xy=root,
                action_id=np.array(action_to_id[row["action_label"]], dtype=np.int64),
            )
            scene_src = frames[int(indices[len(indices) // 2])]
            scene_rel = output_rel_root / "scenes" / f"{sample_id}.jpg"
            copy_scene(scene_src, PROJECT_ROOT / scene_rel, args.scene_size)
            quality_flags = {
                "large_motion": stats["root_displacement_px"] >= args.large_motion_px,
                "low_missing_joints": stats["missing_joint_ratio"] <= args.max_missing_ratio,
                "bbox_large_enough": stats["median_bbox_area_ratio"] >= args.min_bbox_area_ratio,
                "in_frame": stats["out_of_frame_ratio"] <= args.max_out_of_frame_ratio,
                "stable_pose": stats["pose_jump_px_p95"] <= args.max_pose_jump_px,
            }
            processed_rows.append(
                {
                    "sample_id": sample_id,
                    "video_id": video_id,
                    "window_index": window_idx,
                    "frame_start": start,
                    "frame_end": start + source_window - 1,
                    "source_window_frames": source_window,
                    "scene_frame_index": int(indices[len(indices) // 2]),
                    "scene_source_path": str(scene_src.relative_to(PROJECT_ROOT)),
                    "action_label": row["action_label"],
                    "text_prompt": row["text_prompt"],
                    "scene_image_path": str(scene_rel),
                    "motion_path": str(motion_rel),
                    "num_frames": args.target_frames,
                    "image_width": width,
                    "image_height": height,
                    "bbox_stats": {
                        "median_bbox_area_ratio": stats["median_bbox_area_ratio"],
                    },
                    "motion_stats": {
                        "root_displacement_px": stats["root_displacement_px"],
                        "pose_jump_px_p95": stats["pose_jump_px_p95"],
                    },
                    "quality_flags": quality_flags,
                    "quality_stats": stats,
                    "processed": True,
                    "reject_reason": None,
                    "penn_train_flag": row.get("penn_train_flag"),
                }
            )
    out = output_root / "index" / "manifest_processed.jsonl"
    write_jsonl(out, processed_rows)
    (output_root / "index" / "action_to_id.json").write_text(
        json.dumps(action_to_id, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {len(processed_rows)} processed rows to {out}")
    return processed_rows


def split_and_filter(args: argparse.Namespace) -> None:
    ensure_layout()
    rows = read_jsonl(PROCESSED_ROOT / "index" / "manifest_processed.jsonl")
    split_and_filter_rows(rows, args, MINI_ROOT)


def split_and_filter_rows(rows: list[dict[str, Any]], args: argparse.Namespace, output_root: Path) -> list[dict[str, Any]]:
    good = []
    rejected = []
    for row in rows:
        if not row.get("processed"):
            rejected.append({**row, "filter_reject_reason": row.get("reject_reason", "not_processed")})
            continue
        flags = row["quality_flags"]
        ok = all(
            [
                flags["low_missing_joints"],
                flags["bbox_large_enough"],
                flags["in_frame"],
                flags["stable_pose"],
            ]
        )
        if ok:
            good.append(row)
        else:
            reasons = [k for k, v in flags.items() if not v and k != "large_motion"]
            rejected.append({**row, "filter_reject_reason": ",".join(reasons)})

    video_ids = sorted({r["video_id"] for r in good})
    rng = random.Random(args.seed)
    rng.shuffle(video_ids)
    n = len(video_ids)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    split_by_video = {}
    for vid in video_ids[:n_train]:
        split_by_video[vid] = "train"
    for vid in video_ids[n_train : n_train + n_val]:
        split_by_video[vid] = "val"
    for vid in video_ids[n_train + n_val :]:
        split_by_video[vid] = "test"
    for row in good:
        row["split"] = split_by_video[row["video_id"]]

    out = output_root / "manifest_filtered.jsonl"
    rej = output_root / "manifest_rejected.jsonl"
    write_jsonl(out, good)
    write_jsonl(rej, rejected)
    splits = defaultdict(list)
    for vid, split in split_by_video.items():
        splits[split].append(vid)
    (output_root / "splits.json").write_text(json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(good)} filtered samples to {out}")
    print(f"Wrote {len(rejected)} rejected rows to {rej}")
    return good


def draw_preview(sample: dict[str, Any], out_path: Path) -> None:
    scene_path = PROJECT_ROOT / sample["scene_image_path"]
    motion_path = PROJECT_ROOT / sample["motion_path"]
    with Image.open(scene_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        data = np.load(motion_path)
        kpts = data["keypoints_2d_px"]
        vis = data["visibility"].astype(bool)
        frame_idx = len(kpts) // 2
        pts = kpts[frame_idx]
        visible = vis[frame_idx]
        scale_x = img.width / max(float(sample["image_width"]), 1.0)
        scale_y = img.height / max(float(sample["image_height"]), 1.0)
        pts_scaled = pts.copy()
        pts_scaled[:, 0] *= scale_x
        pts_scaled[:, 1] *= scale_y
        for a, b in PENN_ACTION_EDGES:
            if a < len(visible) and b < len(visible) and visible[a] and visible[b]:
                draw.line([tuple(pts_scaled[a]), tuple(pts_scaled[b])], fill=(0, 255, 80), width=3)
        for i, p in enumerate(pts_scaled):
            if i < len(visible) and visible[i]:
                x, y = float(p[0]), float(p[1])
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 40, 40))
        draw.text((8, 8), sample["text_prompt"], fill=(255, 255, 255))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)


def render_previews(args: argparse.Namespace) -> None:
    rows = read_jsonl(MINI_ROOT / "manifest_filtered.jsonl")
    render_preview_rows(rows, args, PROCESSED_ROOT / "previews")


def render_preview_rows(rows: list[dict[str, Any]], args: argparse.Namespace, output_root: Path) -> None:
    if not rows:
        print("No filtered rows to preview.")
        return
    rng = random.Random(args.seed)
    chosen = []
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    for split, split_rows in by_split.items():
        chosen.extend(rng.sample(split_rows, min(args.count_per_split, len(split_rows))))
    for row in chosen:
        out = output_root / f"{row['sample_id']}.jpg"
        draw_preview(row, out)
    print(f"Wrote {len(chosen)} previews to {output_root}")


def report(args: argparse.Namespace) -> None:
    processed_path = PROCESSED_ROOT / "index" / "manifest_processed.jsonl"
    filtered_path = MINI_ROOT / "manifest_filtered.jsonl"
    rejected_path = MINI_ROOT / "manifest_rejected.jsonl"
    processed = read_jsonl(processed_path) if processed_path.exists() else []
    filtered = read_jsonl(filtered_path) if filtered_path.exists() else []
    rejected = read_jsonl(rejected_path) if rejected_path.exists() else []
    action_counts = Counter(r["action_label"] for r in filtered)
    split_counts = Counter(r.get("split", "none") for r in filtered)
    rejection_counts = Counter(r.get("filter_reject_reason") or r.get("reject_reason") or "unknown" for r in rejected)
    displacements = [r["motion_stats"]["root_displacement_px"] for r in filtered if "motion_stats" in r]
    missing = [r["quality_stats"]["missing_joint_ratio"] for r in filtered if "quality_stats" in r]
    lines = [
        "# Penn Action Mini Move-in-2D Dataset Report",
        "",
        f"- processed rows: {len(processed)}",
        f"- filtered samples: {len(filtered)}",
        f"- rejected rows: {len(rejected)}",
        "",
        "## Split Counts",
        "",
        *[f"- {k}: {v}" for k, v in sorted(split_counts.items())],
        "",
        "## Action Counts",
        "",
        *[f"- {k}: {v}" for k, v in sorted(action_counts.items())],
        "",
        "## Quality Summary",
        "",
        f"- root displacement px mean: {float(np.mean(displacements)) if displacements else 0:.2f}",
        f"- missing joint ratio mean: {float(np.mean(missing)) if missing else 0:.4f}",
        "",
        "## Rejection Reasons",
        "",
        *[f"- {k}: {v}" for k, v in sorted(rejection_counts.items())],
    ]
    out = MINI_ROOT / "dataset_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


def write_report(processed: list[dict[str, Any]], filtered: list[dict[str, Any]], rejected: list[dict[str, Any]], out: Path) -> None:
    action_counts = Counter(r["action_label"] for r in filtered)
    split_counts = Counter(r.get("split", "none") for r in filtered)
    rejection_counts = Counter(r.get("filter_reject_reason") or r.get("reject_reason") or "unknown" for r in rejected)
    displacements = [r["motion_stats"]["root_displacement_px"] for r in filtered if "motion_stats" in r]
    missing = [r["quality_stats"]["missing_joint_ratio"] for r in filtered if "quality_stats" in r]
    lines = [
        "# Penn Action Debug Dataset Report",
        "",
        f"- processed rows: {len(processed)}",
        f"- filtered samples: {len(filtered)}",
        f"- rejected rows: {len(rejected)}",
        "",
        "## Split Counts",
        "",
        *[f"- {k}: {v}" for k, v in sorted(split_counts.items())],
        "",
        "## Action Counts",
        "",
        *[f"- {k}: {v}" for k, v in sorted(action_counts.items())],
        "",
        "## Quality Summary",
        "",
        f"- root displacement px mean: {float(np.mean(displacements)) if displacements else 0:.2f}",
        f"- missing joint ratio mean: {float(np.mean(missing)) if missing else 0:.4f}",
        "",
        "## Rejection Reasons",
        "",
        *[f"- {k}: {v}" for k, v in sorted(rejection_counts.items())],
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


def smoke(args: argparse.Namespace) -> None:
    rows = read_jsonl(MINI_ROOT / "manifest_filtered.jsonl")
    smoke_rows(rows, args)


def smoke_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not rows:
        raise SystemExit("No filtered samples found.")
    rng = random.Random(args.seed)
    sample_rows = rng.sample(rows, min(args.count, len(rows)))
    for row in sample_rows:
        scene = PROJECT_ROOT / row["scene_image_path"]
        motion = PROJECT_ROOT / row["motion_path"]
        if not scene.exists():
            raise SystemExit(f"Missing scene image: {scene}")
        if not motion.exists():
            raise SystemExit(f"Missing motion file: {motion}")
        data = np.load(motion)
        for key in ["keypoints_2d_norm", "keypoints_2d_px", "visibility", "bbox_xyxy", "root_xy", "action_id"]:
            if key not in data:
                raise SystemExit(f"{motion} missing key {key}")
        kpts = data["keypoints_2d_norm"]
        if kpts.shape[0] != row["num_frames"] or kpts.shape[-1] != 2:
            raise SystemExit(f"Unexpected keypoint shape for {motion}: {kpts.shape}")
        if not np.isfinite(kpts).all():
            raise SystemExit(f"Non-finite keypoints in {motion}")
    print(f"Smoke checked {len(sample_rows)} samples.")


def debug_subset(args: argparse.Namespace) -> None:
    ensure_layout()
    raw_manifest = PROCESSED_ROOT / "index" / "manifest_raw.jsonl"
    if not raw_manifest.exists():
        build_raw_manifest(args)
    raw_rows = read_jsonl(raw_manifest)[: args.video_count]
    debug_root = DATASET_ROOT / "debug_runs" / args.name
    processed_root = debug_root / "processed"
    mini_root = debug_root / "mini_move_in_2d"
    output_rel_root = Path("project_data") / "datasets" / "penn_action" / "debug_runs" / args.name / "processed"
    for path in [
        processed_root / "index",
        processed_root / "scenes",
        processed_root / "motions",
        processed_root / "previews",
        mini_root,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    write_jsonl(processed_root / "index" / "manifest_raw_subset.jsonl", raw_rows)
    processed = process_rows(raw_rows, args, processed_root, output_rel_root)
    filtered = split_and_filter_rows(processed, args, mini_root)
    rejected_path = mini_root / "manifest_rejected.jsonl"
    rejected = read_jsonl(rejected_path) if rejected_path.exists() else []
    render_preview_rows(filtered, args, processed_root / "previews")
    write_report(processed, filtered, rejected, mini_root / "dataset_report.md")
    smoke_rows(filtered, args)
    print(f"Debug dataset written to {debug_root}")


def all_steps(args: argparse.Namespace) -> None:
    verify_archive(argparse.Namespace(require_complete=True))
    extract(argparse.Namespace(force=args.force_extract))
    build_raw_manifest(args)
    process(args)
    split_and_filter(args)
    render_previews(args)
    report(args)
    smoke(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ensure-layout")
    d = sub.add_parser("download")
    d.add_argument("--foreground", action="store_true")
    v = sub.add_parser("verify-archive")
    v.add_argument("--require-complete", action="store_true")
    e = sub.add_parser("extract")
    e.add_argument("--force", action="store_true")
    sub.add_parser("build-raw-manifest")
    p = sub.add_parser("process")
    add_processing_args(p)
    f = sub.add_parser("filter")
    add_filter_args(f)
    r = sub.add_parser("render-previews")
    r.add_argument("--count-per-split", type=int, default=8)
    r.add_argument("--seed", type=int, default=7)
    sub.add_parser("report")
    s = sub.add_parser("smoke")
    s.add_argument("--count", type=int, default=24)
    s.add_argument("--seed", type=int, default=7)
    dbg = sub.add_parser("debug-subset")
    add_processing_args(dbg)
    add_filter_args(dbg)
    dbg.add_argument("--video-count", type=int, default=3)
    dbg.add_argument("--name", default="first_3_videos")
    dbg.add_argument("--count-per-split", type=int, default=8)
    dbg.add_argument("--count", type=int, default=24)
    a = sub.add_parser("all")
    add_processing_args(a)
    add_filter_args(a)
    a.add_argument("--count-per-split", type=int, default=8)
    a.add_argument("--count", type=int, default=24)
    a.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def add_processing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--min-source-frames", type=int, default=32)
    parser.add_argument("--target-frames", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--scene-size", type=int, default=512)
    parser.add_argument("--large-motion-px", type=float, default=200.0)
    parser.add_argument("--max-missing-ratio", type=float, default=0.35)
    parser.add_argument("--min-bbox-area-ratio", type=float, default=0.005)
    parser.add_argument("--max-out-of-frame-ratio", type=float, default=0.15)
    parser.add_argument("--max-pose-jump-px", type=float, default=160.0)


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", str(PROJECT_DATA / "models" / "torch"))
    os.environ.setdefault("HF_HOME", str(PROJECT_DATA / "models" / "huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DATA / "models" / "cache"))
    ensure_layout()
    commands = {
        "ensure-layout": lambda a: print(PROJECT_DATA),
        "download": download,
        "verify-archive": verify_archive,
        "extract": extract,
        "build-raw-manifest": build_raw_manifest,
        "process": process,
        "filter": split_and_filter,
        "render-previews": render_previews,
        "report": report,
        "smoke": smoke,
        "debug-subset": debug_subset,
        "all": all_steps,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
