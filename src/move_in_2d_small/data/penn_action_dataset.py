from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from move_in_2d_small.paths import resolve_project_path


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = resolve_project_path(path)
    with manifest_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_action_vocab(rows: list[dict[str, Any]]) -> dict[str, int]:
    actions = sorted({row["action_label"] for row in rows})
    return {action: idx for idx, action in enumerate(actions)}


class PennActionMotionDataset(Dataset):
    """Dataset for the mini background+text -> 2D motion task.

    Each sample returns a clean background image, text/action condition, and a
    64-frame Penn Action 2D keypoint motion target.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        image_size: int = 224,
        action_vocab: dict[str, int] | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        all_rows = rows if rows is not None else load_manifest(manifest_path)
        self.rows = [row for row in all_rows if row.get("split") == split]
        if not self.rows:
            raise ValueError(f"No rows found for split={split!r} in {manifest_path}")
        self.manifest_path = resolve_project_path(manifest_path)
        self.split = split
        self.image_size = image_size
        self.action_vocab = action_vocab or build_action_vocab(all_rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = self._load_image(row["scene_image_path"])
        motion = np.load(resolve_project_path(row["motion_path"]))
        keypoints_norm = sanitize_array(motion["keypoints_2d_norm"].astype(np.float32))
        keypoints_px = sanitize_array(motion["keypoints_2d_px"].astype(np.float32))
        visibility = motion["visibility"].astype(np.float32)
        action_id = self.action_vocab[row["action_label"]]
        return {
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "split": row["split"],
            "action_label": row["action_label"],
            "action_id": torch.tensor(action_id, dtype=torch.long),
            "text_prompt": row["text_prompt"],
            "image": image,
            "motion": torch.from_numpy(keypoints_norm),
            "motion_flat": torch.from_numpy(keypoints_norm.reshape(keypoints_norm.shape[0], -1)),
            "keypoints_px": torch.from_numpy(keypoints_px),
            "visibility": torch.from_numpy(visibility),
            "image_width": torch.tensor(row["image_width"], dtype=torch.float32),
            "image_height": torch.tensor(row["image_height"], dtype=torch.float32),
            "scene_image_path": row["scene_image_path"],
            "motion_path": row["motion_path"],
        }

    def _load_image(self, path: str) -> torch.Tensor:
        image_path = resolve_project_path(path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)


def sanitize_array(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def collate_motion_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate mixed tensor/string sample dictionaries."""
    batch: dict[str, Any] = {}
    keys = samples[0].keys()
    for key in keys:
        values = [sample[key] for sample in samples]
        if torch.is_tensor(values[0]):
            batch[key] = torch.stack(values)
        else:
            batch[key] = values
    return batch

