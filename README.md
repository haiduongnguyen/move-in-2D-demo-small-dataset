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
