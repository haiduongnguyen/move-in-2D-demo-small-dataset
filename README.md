# Mini Move-in-2D

Small-scale reproduction scaffold for **Move-in-2D: 2D-Conditioned Human Motion Generation**.

Current focus: build a Penn Action based dataset with:

```text
scene image + text prompt + 2D human motion
```

## Project Data

All heavy data, model caches, logs, and generated artifacts live under:

```text
project_data/
```

This folder is ignored by git.

## Penn Action Download

The Penn Action archive is downloaded to:

```text
project_data/datasets/penn_action/raw/Penn_Action.tar.gz
```

Check progress:

```bash
ps -p $(cat project_data/logs/download/download.pid) -o pid,ppid,etime,cmd
ls -lh project_data/datasets/penn_action/raw/Penn_Action.tar.gz
tail -f project_data/logs/download/download.log
```

Expected final size:

```text
3235203923 bytes
```

## Setup

Install dependencies:

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
```

Set cache paths for future auxiliary models:

```bash
source scripts/setup_project_data_env.sh
```

## Dataset Pipeline

After the archive finishes downloading:

```bash
python3 scripts/penn_action/prepare_dataset.py verify-archive --require-complete
python3 scripts/penn_action/prepare_dataset.py extract
python3 scripts/penn_action/prepare_dataset.py all
```

See [DATASET_PLAN.md](DATASET_PLAN.md) for the detailed dataset plan.

## Paper-like Branch

The paper-like preprocessing branch is documented in
[PAPER_LIKE_DATA_PIPELINE.md](PAPER_LIKE_DATA_PIPELINE.md).

Check available tools and branch status:

```bash
python3 scripts/penn_action/paper_like_pipeline.py tool-check
python3 scripts/penn_action/paper_like_pipeline.py status
```

Run frame quality checks:

```bash
python3 scripts/penn_action/paper_like_pipeline.py frame-quality --max-videos 10
```

Debug inpainting runs are saved as timestamped experiment folders so previous
outputs are preserved:

```bash
python3 scripts/penn_action/paper_like_pipeline.py gt-mask-inpaint-baseline --max-samples 5 --run-name bbox_cv2_telea_test_001
```

## Dataset Loader And Baseline Training

The current trainable mini task is:

```text
bbox+LaMa background image + text/action prompt -> 64-frame 2D keypoint motion
```

The default local config points to the full bbox+LaMa dataset:

```text
configs/penn_action_bbox_lama_full.json
```

Smoke-test the PyTorch dataloader:

```bash
python3 scripts/train/smoke_dataset.py --split train --batch-size 4
```

Render a few skeleton previews from dataset rows:

```bash
python3 scripts/train/render_dataset_samples.py --split train --count 3 --write-video
```

Train the first tiny text/action-only baseline:

```bash
python3 scripts/train/train_text_baseline.py --epochs 3 --batch-size 64 --device cpu
```

Precompute frozen CLIP/DINOv2 condition embeddings:

```bash
python3 scripts/train/precompute_condition_embeddings.py --max-samples 8 --device cpu
python3 scripts/train/precompute_condition_embeddings.py --device cpu
```

Train the paper-like mini diffusion transformer:

```bash
python3 scripts/train/train_mini_diffusion.py --epochs 20 --batch-size 32 --device cpu
```

Run inference from a checkpoint:

```bash
python3 scripts/train/infer_mini_diffusion.py --checkpoint project_data/training_runs/<run_name>/best.pt --split val --count 3 --device cpu
```

Training/debug outputs are written under:

```text
project_data/debug_runs/
project_data/training_runs/
project_data/condition_cache/
```
