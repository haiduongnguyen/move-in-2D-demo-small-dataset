#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from move_in_2d_small.data import load_manifest
from move_in_2d_small.data.condition_cache import cache_path_for_row
from move_in_2d_small.paths import PROJECT_DATA, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute frozen CLIP text and DINOv2 scene embeddings.")
    parser.add_argument("--config", default="configs/penn_action_bbox_lama_full.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(PROJECT_DATA / "models" / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_DATA / "models" / "huggingface"))
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

    config = json.loads((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    rows = load_manifest(config["manifest_path"])
    if args.split:
        rows = [row for row in rows if row.get("split") == args.split]
    if args.max_samples:
        rows = rows[: args.max_samples]
    cache_root = resolve_project_path(config["condition_cache_root"])
    cache_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    clip_processor = CLIPProcessor.from_pretrained(config["clip_model"])
    clip_model = CLIPModel.from_pretrained(config["clip_model"]).to(device).eval()
    dino_processor = AutoImageProcessor.from_pretrained(config["dinov2_model"])
    dino_model = AutoModel.from_pretrained(config["dinov2_model"]).to(device).eval()

    written = 0
    skipped = 0
    with torch.inference_mode():
        for row in tqdm(rows, desc="condition-cache"):
            out_path = cache_path_for_row(cache_root, row["sample_id"])
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue
            text_embed = encode_text(clip_processor, clip_model, row["text_prompt"], device)
            scene_tokens = encode_scene(
                dino_processor,
                dino_model,
                resolve_project_path(row["scene_image_path"]),
                config["scene_tokens"],
                device,
            )
            np.savez_compressed(
                out_path,
                sample_id=row["sample_id"],
                text_embed=text_embed.astype(np.float16),
                scene_tokens=scene_tokens.astype(np.float16),
            )
            written += 1
    print(f"cache_root={cache_root}")
    print(f"written={written} skipped={skipped} total={len(rows)}")


def encode_text(processor, model, prompt: str, device: torch.device) -> np.ndarray:
    inputs = processor(text=[prompt], return_tensors="pt", padding=True, truncation=True).to(device)
    embed = model.get_text_features(**inputs)
    if not torch.is_tensor(embed):
        embed = embed.pooler_output
    embed = torch.nn.functional.normalize(embed, dim=-1)
    return embed[0].detach().cpu().numpy()


def encode_scene(processor, model, image_path: Path, scene_tokens: int, device: torch.device) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
    hidden = model(**inputs).last_hidden_state[0]
    patches = hidden[1:] if hidden.shape[0] > scene_tokens else hidden
    pooled = pool_tokens(patches, scene_tokens)
    pooled = torch.nn.functional.normalize(pooled, dim=-1)
    return pooled.detach().cpu().numpy()


def pool_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if tokens.shape[0] == target_tokens:
        return tokens
    chunks = torch.chunk(tokens, target_tokens, dim=0)
    pooled = [chunk.mean(dim=0) for chunk in chunks]
    if len(pooled) < target_tokens:
        pooled.extend([pooled[-1]] * (target_tokens - len(pooled)))
    return torch.stack(pooled[:target_tokens], dim=0)


if __name__ == "__main__":
    main()
