# DATASET_PLAN: Penn Action -> Mini Move-in-2D

This document is the implementation guide for step 1 of the mini Move-in-2D project.

## Goal

Convert full Penn Action into a small research dataset with:

```text
scene image + text prompt + 2D human motion
```

The pipeline follows the spirit of Move-in-2D: process all videos/sequences first, compute quality signals, then filter into a clean train/validation/test dataset.

## Storage

All heavy artifacts stay inside the project-level data folder:

```text
project_data/
  datasets/penn_action/raw/
  datasets/penn_action/processed/
  datasets/penn_action/mini_move_in_2d/
  models/
  logs/
```

The current Penn Action archive path is:

```text
project_data/datasets/penn_action/raw/Penn_Action.tar.gz
```

Expected archive size:

```text
3235203923 bytes
```

## Environment

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Set cache paths for auxiliary models:

```bash
source scripts/setup_project_data_env.sh
```

This sets:

```text
TORCH_HOME=project_data/models/torch
HF_HOME=project_data/models/huggingface
XDG_CACHE_HOME=project_data/models/cache
```

## Commands

Check or resume download:

```bash
python3 scripts/penn_action/prepare_dataset.py download
tail -f project_data/logs/download/download.log
```

Verify archive:

```bash
python3 scripts/penn_action/prepare_dataset.py verify-archive --require-complete
```

Extract after download completes:

```bash
python3 scripts/penn_action/prepare_dataset.py extract
```

Run the full dataset pipeline:

```bash
python3 scripts/penn_action/prepare_dataset.py all
```

Or run step by step:

```bash
python3 scripts/penn_action/prepare_dataset.py build-raw-manifest
python3 scripts/penn_action/prepare_dataset.py process
python3 scripts/penn_action/prepare_dataset.py filter
python3 scripts/penn_action/prepare_dataset.py render-previews
python3 scripts/penn_action/prepare_dataset.py report
python3 scripts/penn_action/prepare_dataset.py smoke
```

## Outputs

Raw index:

```text
project_data/datasets/penn_action/processed/index/manifest_raw.jsonl
```

Full processed index:

```text
project_data/datasets/penn_action/processed/index/manifest_processed.jsonl
```

Filtered mini dataset:

```text
project_data/datasets/penn_action/mini_move_in_2d/manifest_filtered.jsonl
project_data/datasets/penn_action/mini_move_in_2d/splits.json
project_data/datasets/penn_action/mini_move_in_2d/dataset_report.md
```

Per-sample motion files:

```text
project_data/datasets/penn_action/processed/motions/*.npz
```

Per-sample scene images:

```text
project_data/datasets/penn_action/processed/scenes/*.jpg
```

Preview images:

```text
project_data/datasets/penn_action/processed/previews/*.jpg
```

## Dataset Schema

Each filtered manifest row includes:

```text
sample_id
video_id
action_label
text_prompt
scene_image_path
motion_path
num_frames
image_width
image_height
bbox_stats
motion_stats
quality_flags
split
```

Each `.npz` motion file includes:

```text
keypoints_2d_norm  # [T, J, 2]
keypoints_2d_px    # [T, J, 2]
visibility         # [T, J]
bbox_xyxy          # [T, 4]
root_xy            # [T, 2]
action_id
```

## V1 Defaults

- Motion representation: Penn Action 2D keypoints.
- Sequence length: 64 frames.
- Scene image: original middle frame of the sequence window.
- Text prompt: deterministic template from action label.
- Split policy: by `video_id`, not by frame.
- Inpainting/person removal: deferred to V1.1.

