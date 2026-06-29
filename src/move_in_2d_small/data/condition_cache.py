from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from move_in_2d_small.data import PennActionMotionDataset
from move_in_2d_small.paths import resolve_project_path


def cache_path_for_row(cache_root: str | Path, sample_id: str) -> Path:
    return resolve_project_path(cache_root) / f"{sample_id}.npz"


def load_condition_cache(path: str | Path) -> dict[str, torch.Tensor]:
    data = np.load(resolve_project_path(path))
    return {
        "text_embed": torch.from_numpy(data["text_embed"].astype(np.float32)),
        "scene_tokens": torch.from_numpy(data["scene_tokens"].astype(np.float32)),
    }


class CachedConditionMotionDataset(PennActionMotionDataset):
    """Penn Action dataset variant that also returns frozen CLIP/DINO features."""

    def __init__(self, *args: Any, condition_cache_root: str | Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.condition_cache_root = resolve_project_path(condition_cache_root)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        cache_path = cache_path_for_row(self.condition_cache_root, sample["sample_id"])
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing condition cache for {sample['sample_id']}: {cache_path}")
        cond = load_condition_cache(cache_path)
        sample["text_embed"] = cond["text_embed"]
        sample["scene_tokens"] = cond["scene_tokens"]
        sample["condition_cache_path"] = str(cache_path)
        return sample
