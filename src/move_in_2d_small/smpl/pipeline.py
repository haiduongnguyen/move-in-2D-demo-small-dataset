from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from move_in_2d_small.paths import PROJECT_DATA, PROJECT_ROOT, resolve_project_path


DATASET_ROOT = PROJECT_DATA / "datasets" / "penn_action"
RAW_FRAMES_ROOT = DATASET_ROOT / "raw" / "Penn_Action" / "frames"
PAPER_LIKE_MANIFEST = (
    DATASET_ROOT
    / "paper_like_mini_move_in_2d"
    / "bbox_lama_dataset_full_001"
    / "manifest_filtered.jsonl"
)
SMPL_PROCESSED_ROOT = DATASET_ROOT / "smpl_processed"
SMPL_DATASET_ROOT = DATASET_ROOT / "paper_like_smpl_mini_move_in_2d"
DEFAULT_4D_HUMANS_ROOT = PROJECT_DATA / "models" / "4d_humans" / "4D-Humans"
DEFAULT_SMPL_MODEL_ROOT = PROJECT_DATA / "models" / "smpl"
FOURD_HUMANS_HOME = PROJECT_DATA / "models" / "4d_humans" / "home"
LOG_ROOT = PROJECT_DATA / "logs" / "smpl_extract"

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

SMPL_PREVIEW_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (9, 12),
    (12, 15),
    (12, 16),
    (12, 17),
    (16, 18),
    (18, 20),
    (20, 22),
    (17, 19),
    (19, 21),
    (21, 23),
]


def ensure_layout() -> None:
    for path in [
        SMPL_PROCESSED_ROOT / "raw_outputs",
        SMPL_PROCESSED_ROOT / "motions",
        SMPL_PROCESSED_ROOT / "index",
        SMPL_PROCESSED_ROOT / "previews",
        SMPL_PROCESSED_ROOT / "comparison_with_2d",
        SMPL_PROCESSED_ROOT / "test_runs",
        SMPL_DATASET_ROOT,
        DEFAULT_4D_HUMANS_ROOT.parent,
        DEFAULT_SMPL_MODEL_ROOT,
        FOURD_HUMANS_HOME,
        LOG_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rel(path: str | Path) -> str:
    resolved = resolve_project_path(path)
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def frame_files_for(video_id: str) -> list[Path]:
    frame_dir = RAW_FRAMES_ROOT / video_id
    files: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(frame_dir.glob(pattern))
    return sorted(files)


def load_manifest(path: str | Path = PAPER_LIKE_MANIFEST) -> list[dict[str, Any]]:
    if not resolve_project_path(path).exists():
        raise FileNotFoundError(f"Missing paper-like manifest: {resolve_project_path(path)}")
    return read_jsonl(path)


def choose_video_rows(rows: list[dict[str, Any]], num_videos: int, video_ids: list[str] | None = None) -> list[dict[str, Any]]:
    by_video: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_id = row["video_id"]
        current = by_video.get(video_id)
        if current is None or int(row.get("frame_end", 0)) > int(current.get("frame_end", 0)):
            by_video[video_id] = row
    if video_ids:
        missing = [video_id for video_id in video_ids if video_id not in by_video]
        if missing:
            raise ValueError(f"Video ids not present in manifest: {missing}")
        return [by_video[video_id] for video_id in video_ids]

    selected: list[dict[str, Any]] = []
    seen_actions: set[str] = set()
    eligible_rows = [
        row
        for row in by_video.values()
        if len(frame_files_for(row["video_id"])) >= int(row.get("frame_end", row.get("num_frames", 64) - 1)) + 1
    ]
    if not eligible_rows:
        eligible_rows = list(by_video.values())

    for row in eligible_rows:
        action = row["action_label"]
        if action in seen_actions:
            continue
        selected.append(row)
        seen_actions.add(action)
        if len(selected) >= num_videos:
            return selected
    for row in eligible_rows:
        if row["video_id"] not in {item["video_id"] for item in selected}:
            selected.append(row)
            if len(selected) >= num_videos:
                break
    return selected


@dataclass(frozen=True)
class SetupCheck:
    ok: bool
    report: dict[str, Any]


def check_setup(
    fourd_humans_root: str | Path = DEFAULT_4D_HUMANS_ROOT,
    smpl_model_root: str | Path = DEFAULT_SMPL_MODEL_ROOT,
) -> SetupCheck:
    ensure_layout()
    fourd_root = resolve_project_path(fourd_humans_root)
    smpl_root = resolve_project_path(smpl_model_root)
    track_py = fourd_root / "track.py"
    demo_py = fourd_root / "demo.py"
    phalp_smpl_cache = FOURD_HUMANS_HOME / ".cache" / "phalp" / "3D" / "models" / "smpl" / "SMPL_NEUTRAL.pkl"
    hmr2_smpl_cache = FOURD_HUMANS_HOME / ".cache" / "4DHumans" / "data" / "smpl" / "SMPL_NEUTRAL.pkl"
    smpl_candidates = [
        smpl_root / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
        smpl_root / "SMPL_NEUTRAL.pkl",
        smpl_root / "SMPL_NEUTRAL.npz",
        fourd_root / "data" / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
        fourd_root / "data" / "smpl" / "SMPL_NEUTRAL.pkl",
        fourd_root / "data" / "smpl" / "SMPL_NEUTRAL.npz",
        phalp_smpl_cache,
        hmr2_smpl_cache,
    ]
    env = {
        "python": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "cwd": str(PROJECT_ROOT),
        "fourd_humans_root": str(fourd_root),
        "smpl_model_root": str(smpl_root),
        "track_py_exists": track_py.exists(),
        "demo_py_exists": demo_py.exists(),
        "smpl_model_candidates": [str(path) for path in smpl_candidates],
        "smpl_model_exists": any(path.exists() for path in smpl_candidates),
        "project_cache_home": str(FOURD_HUMANS_HOME),
        "phalp_required_smpl_cache": str(phalp_smpl_cache),
        "hmr2_required_smpl_cache": str(hmr2_smpl_cache),
        "notes": [
            "4D-Humans is intentionally kept outside the project package.",
            "If this check fails, clone/setup 4D-Humans under project_data/models/4d_humans/4D-Humans or pass --fourd-humans-root.",
            "SMPL neutral model files are license-gated and must be downloaded manually from the official SMPL/SMPLify access flow.",
            "The original 4D-Humans README asks for basicModel_neutral_lbs_10_207_0_v1.0.0.pkl in 4D-Humans/data/.",
            "PHALP expects a converted SMPL_NEUTRAL.pkl under HOME/.cache/phalp/3D/models/smpl/.",
        ],
    }
    env["ok"] = bool(env["track_py_exists"] and env["smpl_model_exists"])
    report_path = LOG_ROOT / "setup_check.json"
    write_json(report_path, env)
    return SetupCheck(ok=bool(env["ok"]), report=env)


def prepare_test_run(
    run_name: str,
    num_videos: int = 5,
    video_ids: list[str] | None = None,
    manifest_path: str | Path = PAPER_LIKE_MANIFEST,
) -> Path:
    ensure_layout()
    rows = load_manifest(manifest_path)
    selected = choose_video_rows(rows, num_videos=num_videos, video_ids=video_ids)
    run_root = SMPL_PROCESSED_ROOT / "test_runs" / run_name
    run_root.mkdir(parents=True, exist_ok=False)
    video_rows = []
    for row in selected:
        video_id = row["video_id"]
        frames = frame_files_for(video_id)
        video_rows.append(
            {
                "video_id": video_id,
                "action_label": row["action_label"],
                "first_sample_id": row["sample_id"],
                "frame_dir": rel(RAW_FRAMES_ROOT / video_id),
                "num_frames": len(frames),
                "first_frame": rel(frames[0]) if frames else None,
                "paper_like_manifest": rel(manifest_path),
            }
        )
    write_jsonl(run_root / "selected_videos.jsonl", video_rows)
    write_json(
        run_root / "run_config.json",
        {
            "run_name": run_name,
            "num_videos": len(video_rows),
            "mode": "test",
            "selected_video_ids": [row["video_id"] for row in video_rows],
        },
    )
    return run_root


def load_selected_video_rows(run_name: str) -> list[dict[str, Any]]:
    path = SMPL_PROCESSED_ROOT / "test_runs" / run_name / "selected_videos.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected videos: {path}. Run prepare-test first.")
    return read_jsonl(path)


def run_4dhumans_extract(
    run_name: str,
    fourd_humans_root: str | Path = DEFAULT_4D_HUMANS_ROOT,
    python_bin: str = sys.executable,
    extra_args: list[str] | None = None,
    dry_run: bool = False,
) -> Path:
    ensure_layout()
    setup = check_setup(fourd_humans_root=fourd_humans_root)
    if not setup.report["track_py_exists"] and not dry_run:
        raise FileNotFoundError(f"Missing 4D-Humans track.py under {resolve_project_path(fourd_humans_root)}")
    rows = load_selected_video_rows(run_name)
    raw_run_root = SMPL_PROCESSED_ROOT / "raw_outputs" / run_name
    raw_run_root.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict[str, Any]] = []
    for row in rows:
        video_id = row["video_id"]
        frame_dir = resolve_project_path(row["frame_dir"])
        output_dir = raw_run_root / video_id
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            python_bin,
            str(resolve_project_path(fourd_humans_root) / "track.py"),
            f"video.source={frame_dir}",
            f"video.output_dir={output_dir}",
        ]
        cmd.extend(extra_args or [])
        record = {
            "video_id": video_id,
            "frame_dir": str(frame_dir),
            "output_dir": str(output_dir),
            "command": cmd,
            "dry_run": dry_run,
        }
        if dry_run:
            record["returncode"] = None
        else:
            env = os.environ.copy()
            env["HOME"] = str(FOURD_HUMANS_HOME)
            env["XDG_CACHE_HOME"] = str(FOURD_HUMANS_HOME / ".cache")
            proc = subprocess.run(
                cmd,
                cwd=resolve_project_path(fourd_humans_root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
            record["returncode"] = proc.returncode
            log_path = LOG_ROOT / f"{run_name}_{video_id}.log"
            log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
            record["log_path"] = rel(log_path)
            record["success"] = proc.returncode == 0
        log_rows.append(record)
    write_jsonl(LOG_ROOT / f"{run_name}_extract_commands.jsonl", log_rows)
    return raw_run_root


def to_numpy(value: Any) -> np.ndarray | None:
    try:
        import torch
    except Exception:  # pragma: no cover - torch may not be importable in a converter-only environment
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (float, int, bool, np.number)):
        return np.asarray(value)
    if isinstance(value, (list, tuple)) and value:
        converted = [to_numpy(item) for item in value]
        converted = [item for item in converted if item is not None]
        if not converted:
            return None
        try:
            return np.stack(converted)
        except ValueError:
            return None
    return None


def collect_arrays(obj: Any, prefix: str = "") -> list[tuple[str, np.ndarray]]:
    arrays: list[tuple[str, np.ndarray]] = []
    arr = to_numpy(obj)
    if arr is not None and arr.dtype != object:
        arrays.append((prefix, arr))
        return arrays
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            arrays.extend(collect_arrays(value, path))
    elif isinstance(obj, (list, tuple)):
        # Walk a small prefix of large lists to avoid exploding deeply nested track dumps.
        for idx, value in enumerate(obj[:10]):
            path = f"{prefix}[{idx}]"
            arrays.extend(collect_arrays(value, path))
    return arrays


FIELD_ALIASES = {
    "global_orient": ("global_orient", "global_orientation", "root_orient", "root_pose"),
    "body_pose": ("body_pose", "pose_body", "body_rotvec", "pred_pose"),
    "betas": ("betas", "shape", "pred_shape"),
    "transl": ("transl", "translation", "pred_cam_t", "cam_t"),
    "camera": ("camera", "pred_cam", "cam", "camera_bbox"),
    "joints_3d": ("joints_3d", "pred_joints", "smpl_joints", "joints"),
    "joints_2d": ("joints_2d", "keypoints_2d", "projected_keypoints", "pred_keypoints_2d"),
    "joint_confidence": ("joint_confidence", "keypoint_scores", "scores", "conf"),
    "bbox_xyxy": ("bbox_xyxy", "bbox", "boxes", "person_bbox"),
}


def score_candidate(field: str, key_path: str, arr: np.ndarray) -> tuple[int, int, int]:
    aliases = FIELD_ALIASES[field]
    key = key_path.lower().replace("-", "_")
    alias_score = max((10 if alias in key else 0) for alias in aliases)
    shape_score = 0
    if field == "global_orient" and arr.ndim >= 2 and arr.shape[-1] in {3, 6, 9}:
        shape_score += 5
    elif field == "body_pose" and arr.ndim >= 2 and arr.shape[-1] in {3, 6, 69, 72, 135, 144}:
        shape_score += 5
    elif field == "betas" and arr.shape[-1] in {10, 16}:
        shape_score += 5
    elif field in {"joints_3d", "joints_2d"} and arr.ndim >= 3 and arr.shape[-2] >= 13:
        shape_score += 5
    elif field == "bbox_xyxy" and arr.shape[-1] == 4:
        shape_score += 5
    elif field in {"transl", "camera"} and arr.ndim >= 2:
        shape_score += 3
    time_score = int(arr.shape[0]) if arr.ndim > 0 else 0
    return (alias_score, shape_score, time_score)


def choose_field(arrays: list[tuple[str, np.ndarray]], field: str) -> np.ndarray | None:
    candidates = []
    for key_path, arr in arrays:
        score = score_candidate(field, key_path, arr)
        if score[0] > 0 or score[1] > 0:
            candidates.append((score, key_path, arr))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    arr = np.asarray(candidates[0][2])
    return np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)


def first_pickle(root: Path) -> Path | None:
    pickles = sorted(root.rglob("*.pkl")) + sorted(root.rglob("*.pickle"))
    return pickles[0] if pickles else None


def convert_raw_outputs(run_name: str) -> Path:
    ensure_layout()
    raw_run_root = SMPL_PROCESSED_ROOT / "raw_outputs" / run_name
    if not raw_run_root.exists():
        raise FileNotFoundError(f"Missing raw output run: {raw_run_root}")
    converted_root = SMPL_PROCESSED_ROOT / "motions" / run_name
    converted_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    video_dirs = [path for path in sorted(raw_run_root.iterdir()) if path.is_dir()]
    for video_dir in video_dirs:
        video_id = video_dir.name
        pkl_path = first_pickle(video_dir)
        if pkl_path is None:
            failures.append({"video_id": video_id, "reason": "missing_pickle", "raw_output_dir": str(video_dir)})
            continue
        try:
            with pkl_path.open("rb") as f:
                obj = pickle.load(f)
        except Exception as exc:
            failures.append({"video_id": video_id, "reason": f"pickle_load_failed: {exc}", "pickle_path": str(pkl_path)})
            continue
        arrays = collect_arrays(obj)
        payload: dict[str, np.ndarray] = {}
        for field in FIELD_ALIASES:
            arr = choose_field(arrays, field)
            if arr is not None:
                payload[field] = arr
        if not any(key in payload for key in ("global_orient", "body_pose", "joints_3d", "joints_2d")):
            failures.append(
                {
                    "video_id": video_id,
                    "reason": "no_recognized_motion_fields",
                    "pickle_path": str(pkl_path),
                    "array_keys_seen": [key for key, _ in arrays[:50]],
                }
            )
            continue
        time_length = infer_time_length(payload)
        payload["valid"] = np.ones((time_length,), dtype=np.uint8)
        payload["frame_indices"] = np.arange(time_length, dtype=np.int32)
        out_path = converted_root / f"{video_id}_smpl.npz"
        np.savez_compressed(out_path, **payload)
        manifest_rows.append(
            {
                "video_id": video_id,
                "smpl_video_motion_path": rel(out_path),
                "raw_pickle_path": rel(pkl_path),
                "num_frames": int(time_length),
                "fields": sorted(payload.keys()),
            }
        )
    manifest_path = SMPL_PROCESSED_ROOT / "index" / f"manifest_smpl_raw_{run_name}.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    write_jsonl(SMPL_PROCESSED_ROOT / "index" / f"manifest_smpl_failures_{run_name}.jsonl", failures)
    return manifest_path


def infer_time_length(payload: dict[str, np.ndarray]) -> int:
    lengths = []
    for key, arr in payload.items():
        if key == "betas" and arr.ndim == 1:
            continue
        if arr.ndim > 0:
            lengths.append(int(arr.shape[0]))
    return max(lengths) if lengths else 0


def slice_field(arr: np.ndarray, start: int, end: int, target_len: int) -> np.ndarray | None:
    if arr.ndim == 0:
        return arr
    if arr.ndim == 1 and arr.shape[0] in {10, 16}:
        return arr
    if arr.shape[0] <= start:
        return None
    clamped_end = min(end, arr.shape[0] - 1)
    sliced = arr[start : clamped_end + 1]
    if sliced.shape[0] == 0:
        return None
    if sliced.shape[0] != target_len:
        sliced = resample_time(sliced, target_len)
    return sliced


def resample_time(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[0] == target_len:
        return arr
    if arr.shape[0] == 1:
        return np.repeat(arr, target_len, axis=0)
    src_x = np.linspace(0.0, 1.0, arr.shape[0])
    dst_x = np.linspace(0.0, 1.0, target_len)
    if arr.dtype.kind in {"b", "i", "u"}:
        indices = np.rint(dst_x * (arr.shape[0] - 1)).astype(np.int64)
        return arr[indices]
    flat = arr.reshape(arr.shape[0], -1)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(dst_x, src_x, flat[:, col])
    return out.reshape((target_len, *arr.shape[1:]))


def build_smpl_dataset(run_name: str, output_run_name: str | None = None, manifest_path: str | Path = PAPER_LIKE_MANIFEST) -> Path:
    ensure_layout()
    output_run_name = output_run_name or run_name
    raw_manifest_path = SMPL_PROCESSED_ROOT / "index" / f"manifest_smpl_raw_{run_name}.jsonl"
    if not raw_manifest_path.exists():
        raise FileNotFoundError(f"Missing converted SMPL raw manifest: {raw_manifest_path}")
    raw_rows = read_jsonl(raw_manifest_path)
    smpl_by_video = {row["video_id"]: row for row in raw_rows}
    paper_rows = load_manifest(manifest_path)
    out_root = SMPL_DATASET_ROOT / output_run_name
    motion_root = out_root / "motions"
    motion_root.mkdir(parents=True, exist_ok=True)
    final_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in paper_rows:
        video_id = row["video_id"]
        smpl_row = smpl_by_video.get(video_id)
        if smpl_row is None:
            rejected.append({"sample_id": row["sample_id"], "video_id": video_id, "reason": "missing_video_smpl"})
            continue
        data = np.load(resolve_project_path(smpl_row["smpl_video_motion_path"]))
        start = int(row["frame_start"])
        end = int(row["frame_end"])
        target_len = int(row["num_frames"])
        sample_payload: dict[str, np.ndarray] = {}
        failed_field = None
        for key in data.files:
            sliced = slice_field(data[key], start, end, target_len)
            if sliced is None:
                if key in {"global_orient", "body_pose", "joints_3d", "joints_2d", "valid"}:
                    failed_field = key
                    break
                continue
            sample_payload[key] = sliced
        if failed_field is not None:
            rejected.append(
                {
                    "sample_id": row["sample_id"],
                    "video_id": video_id,
                    "reason": f"cannot_slice_required_field:{failed_field}",
                    "frame_start": start,
                    "frame_end": end,
                    "smpl_num_frames": int(smpl_row["num_frames"]),
                }
            )
            continue
        if not any(key in sample_payload for key in ("global_orient", "body_pose", "joints_3d", "joints_2d")):
            rejected.append({"sample_id": row["sample_id"], "video_id": video_id, "reason": "no_motion_fields_after_slice"})
            continue
        out_motion = motion_root / f"{row['sample_id']}_smpl.npz"
        np.savez_compressed(out_motion, **sample_payload)
        valid = sample_payload.get("valid", np.ones((target_len,), dtype=np.uint8))
        final_row = {
            "sample_id": row["sample_id"],
            "video_id": video_id,
            "action_label": row["action_label"],
            "text_prompt": row["text_prompt"],
            "scene_image_path": row["scene_image_path"],
            "smpl_motion_path": rel(out_motion),
            "source_2d_motion_path": row["motion_path"],
            "num_frames": target_len,
            "image_width": row["image_width"],
            "image_height": row["image_height"],
            "frame_start": start,
            "frame_end": end,
            "split": row["split"],
            "smpl_stats": {
                "valid_ratio": float(np.asarray(valid).mean()) if valid.size else 0.0,
                "fields": sorted(sample_payload.keys()),
            },
            "quality_flags": {
                "smpl_available": True,
                "valid_enough": float(np.asarray(valid).mean()) >= 0.8 if valid.size else False,
            },
        }
        if "original_scene_image_path" in row:
            final_row["original_scene_image_path"] = row["original_scene_image_path"]
        final_rows.append(final_row)
    write_jsonl(out_root / "manifest_smpl.jsonl", final_rows)
    write_jsonl(out_root / "rejected_smpl.jsonl", rejected)
    write_json(out_root / "splits.json", split_summary(final_rows))
    write_dataset_report(out_root, final_rows, rejected, run_name)
    return out_root / "manifest_smpl.jsonl"


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row["video_id"])
    return {
        split: {
            "num_samples": sum(1 for row in rows if row["split"] == split),
            "num_videos": len(set(video_ids)),
            "video_ids": sorted(set(video_ids)),
        }
        for split, video_ids in sorted(by_split.items())
    }


def write_dataset_report(out_root: Path, rows: list[dict[str, Any]], rejected: list[dict[str, Any]], source_run_name: str) -> None:
    action_counts = Counter(row["action_label"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    reject_counts = Counter(row["reason"] for row in rejected)
    lines = [
        "# SMPL Dataset Report",
        "",
        f"- source SMPL run: `{source_run_name}`",
        f"- accepted samples: {len(rows)}",
        f"- rejected samples: {len(rejected)}",
        f"- accepted videos: {len({row['video_id'] for row in rows})}",
        "",
        "## Splits",
        "",
    ]
    for split, count in sorted(split_counts.items()):
        lines.append(f"- {split}: {count}")
    lines.extend(["", "## Actions", ""])
    for action, count in sorted(action_counts.items()):
        lines.append(f"- {action}: {count}")
    lines.extend(["", "## Rejection Reasons", ""])
    if reject_counts:
        for reason, count in sorted(reject_counts.items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This branch preserves the earlier 2D keypoint dataset and stores SMPL motions separately.",
            "- Preview quality should be checked before training on this manifest.",
        ]
    )
    (out_root / "dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_previews(dataset_run_name: str, max_samples: int = 8) -> Path:
    ensure_layout()
    out_root = SMPL_DATASET_ROOT / dataset_run_name
    manifest_path = out_root / "manifest_smpl.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing SMPL dataset manifest: {manifest_path}")
    rows = read_jsonl(manifest_path)[:max_samples]
    preview_root = out_root / "previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        image_path = resolve_project_path(row.get("original_scene_image_path") or row["scene_image_path"])
        image = Image.open(image_path).convert("RGB")
        frames = []
        motion_2d = np.load(resolve_project_path(row["source_2d_motion_path"]))
        smpl = np.load(resolve_project_path(row["smpl_motion_path"]))
        keypoints_2d = motion_2d["keypoints_2d_px"]
        visibility = motion_2d["visibility"]
        smpl_joints_2d = smpl["joints_2d"] if "joints_2d" in smpl.files else None
        for frame_idx in np.linspace(0, row["num_frames"] - 1, num=min(6, row["num_frames"]), dtype=int):
            canvas = image.copy()
            draw = ImageDraw.Draw(canvas)
            draw_skeleton(draw, keypoints_2d[frame_idx], visibility[frame_idx], PENN_EDGES, color=(0, 210, 255), radius=3)
            if smpl_joints_2d is not None and smpl_joints_2d.ndim >= 3:
                conf = np.ones((smpl_joints_2d.shape[1],), dtype=np.float32)
                draw_skeleton(draw, smpl_joints_2d[frame_idx], conf, SMPL_PREVIEW_EDGES, color=(255, 72, 72), radius=2)
            draw.rectangle((0, 0, 220, 34), fill=(0, 0, 0))
            draw.text((8, 8), f"{row['sample_id']} frame {int(frame_idx)}", fill=(255, 255, 255))
            frames.append(canvas)
        sheet = contact_sheet(frames)
        sheet.save(preview_root / f"{row['sample_id']}_smpl_preview.jpg", quality=92)
    return preview_root


def draw_skeleton(
    draw: ImageDraw.ImageDraw,
    joints: np.ndarray,
    visibility: np.ndarray,
    edges: list[tuple[int, int]],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    joints = np.asarray(joints)
    visibility = np.asarray(visibility)
    for a, b in edges:
        if a >= len(joints) or b >= len(joints) or a >= len(visibility) or b >= len(visibility):
            continue
        if visibility[a] <= 0 or visibility[b] <= 0:
            continue
        ax, ay = joints[a][:2]
        bx, by = joints[b][:2]
        if not np.all(np.isfinite([ax, ay, bx, by])):
            continue
        draw.line((float(ax), float(ay), float(bx), float(by)), fill=color, width=max(1, radius))
    for idx, xy in enumerate(joints):
        if idx >= len(visibility) or visibility[idx] <= 0:
            continue
        x, y = xy[:2]
        if not np.all(np.isfinite([x, y])):
            continue
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def contact_sheet(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("No images to render.")
    widths, heights = zip(*(img.size for img in images), strict=False)
    sheet = Image.new("RGB", (sum(widths), max(heights)), color=(25, 25, 25))
    x = 0
    for image in images:
        sheet.paste(image, (x, 0))
        x += image.size[0]
    return sheet


def run_test_pipeline(
    run_name: str,
    num_videos: int,
    video_ids: list[str] | None,
    fourd_humans_root: str | Path,
    python_bin: str,
    extra_args: list[str] | None,
    dry_run: bool,
) -> dict[str, Any]:
    run_root = prepare_test_run(run_name=run_name, num_videos=num_videos, video_ids=video_ids)
    raw_root = run_4dhumans_extract(
        run_name=run_name,
        fourd_humans_root=fourd_humans_root,
        python_bin=python_bin,
        extra_args=extra_args,
        dry_run=dry_run,
    )
    result: dict[str, Any] = {"run_root": rel(run_root), "raw_output_root": rel(raw_root)}
    if not dry_run:
        manifest = convert_raw_outputs(run_name)
        result["converted_manifest"] = rel(manifest)
        dataset_manifest = build_smpl_dataset(run_name=run_name)
        result["dataset_manifest"] = rel(dataset_manifest)
        result["preview_root"] = rel(render_previews(run_name))
    return result
