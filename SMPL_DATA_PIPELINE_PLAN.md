# SMPL_DATA_PIPELINE_PLAN: Penn Action -> Paper-like SMPL Motion

This document is the implementation guide for the SMPL branch of the mini Move-in-2D project.

## Goal

Build a second motion dataset branch that is closer to the Move-in-2D paper:

```text
video frames + background image + text prompt -> SMPL motion sequence
```

The current working branch uses Penn Action 2D keypoints:

```text
project_data/datasets/penn_action/processed/motions/*.npz
```

The SMPL branch must not replace or delete that branch. It will run in parallel so we can compare:

```text
2D keypoint branch vs SMPL pseudo-motion branch
```

## Why SMPL

The current 2D motion representation is simple:

```text
[64, 13, 2]
```

It is easy to train and visualize, but it has weak body constraints. A model can predict scattered joints, broken limbs, or unstable body shapes.

The paper uses 4D-Humans to extract pseudo ground-truth motion in SMPL format. SMPL gives a stronger human body prior:

```text
body pose
global orientation
body shape
translation / camera
optional 3D joints and projected 2D joints
```

This should make the generated motion more human-like than raw 2D points, at the cost of a heavier preprocessing pipeline.

## Important Distinction

SMPL should be extracted from video, not from a single background image.

Single image SMPL:

```text
one frame -> one body pose
```

Video SMPL:

```text
video frames -> per-frame SMPL -> temporal motion sequence
```

For this project we need video SMPL because the training target is motion over 64 frames.

## Storage

All heavy artifacts stay inside `project_data/`.

```text
project_data/
  datasets/penn_action/
    smpl_processed/
      raw_outputs/
      motions/
      index/
      previews/
      comparison_with_2d/
    paper_like_smpl_mini_move_in_2d/
      <run_name>/
        manifest_smpl.jsonl
        splits.json
        dataset_report.md
        scenes/
        motions/
        previews/
  models/
    4d_humans/
    smpl/
  logs/
    smpl_extract/
```

The existing branches remain unchanged:

```text
project_data/datasets/penn_action/mini_move_in_2d/
project_data/datasets/penn_action/paper_like_mini_move_in_2d/
project_data/training_runs/mini_diffusion_full_20epoch_001/
```

## Inputs

Primary inputs:

```text
project_data/datasets/penn_action/raw/Penn_Action/frames/<video_id>/*.jpg
project_data/datasets/penn_action/paper_like_mini_move_in_2d/bbox_lama_dataset_full_001/manifest_filtered.jsonl
```

The existing paper-like manifest provides:

```text
sample_id
video_id
text_prompt
scene_image_path
split
```

The SMPL branch will reuse the same background images and text prompts, but replace the 2D keypoint target with SMPL motion.

## Tool Choice

Target tool:

```text
4D-Humans / HMR2 style SMPL estimator
```

Expected role:

```text
raw video frames -> per-frame SMPL parameters + optional joints/projections
```

Before full implementation, we need a small setup check because 4D-Humans is heavier and more version-sensitive than the current 2D pipeline.

Potential issues:

- Python and dependency version mismatch.
- CPU-only inference may be slow.
- CUDA may not work on the current GTX 980 Ti with the installed PyTorch build.
- Some SMPL model files may require manual download or license-gated access.

## Pipeline Overview

```text
1. Setup SMPL extraction environment
2. Run SMPL extraction on 3-5 test videos
3. Convert raw SMPL outputs into normalized project .npz files
4. Build 64-frame SMPL windows aligned with existing samples
5. Validate visually and numerically
6. Compare SMPL projected joints with Penn Action 2D keypoints
7. Build full SMPL dataset manifest
8. Train a new SMPL-aware model branch later
```

## Step 1: Setup SMPL Extraction Environment

### Purpose

Install or vendor the SMPL extraction tool without affecting the current 2D pipeline.

### Input

```text
project root
project_data/models/
```

### Output

```text
project_data/models/4d_humans/
project_data/models/smpl/
project_data/logs/smpl_extract/setup.log
```

### Logic

- Check current Python, PyTorch, CUDA, and device support.
- Decide whether to run 4D-Humans in the current environment or a separate environment.
- Put downloaded checkpoints and caches under `project_data/models/`.
- Add a small setup note documenting exact commands and limitations.

### Acceptance

- A minimal import or demo command runs.
- Model files are not stored in git.
- Existing 2D scripts still run.

## Step 2: Test Extraction On 3-5 Videos

### Purpose

Confirm that SMPL extraction works before running the full dataset.

### Input

```text
raw Penn Action frames for selected video_ids
```

### Output

```text
project_data/datasets/penn_action/smpl_processed/raw_outputs/test_<run_name>/<video_id>/
project_data/datasets/penn_action/smpl_processed/previews/test_<run_name>/
```

### Logic

- Select 3-5 videos from different actions.
- Run SMPL estimator frame by frame or video by video.
- Save raw model output exactly as produced by the tool.
- Do not overwrite old test runs; each test gets a unique run folder.

### Acceptance

- Raw SMPL outputs exist for selected videos.
- Preview render shows a plausible person pose for at least some frames.
- Failures are logged instead of crashing the whole run.

## Step 3: Convert Raw SMPL Output

### Purpose

Normalize tool-specific outputs into one project schema.

### Input

```text
smpl_processed/raw_outputs/<run_name>/<video_id>/
```

### Output

```text
smpl_processed/motions/<video_id>_smpl.npz
smpl_processed/index/manifest_smpl_raw.jsonl
```

### Proposed `.npz` Schema

```text
global_orient      # [T, 3]
body_pose          # [T, 23, 3] or [T, 69]
betas              # [10] or [T, 10]
transl             # [T, 3], if available
camera             # [T, C], if available
joints_3d          # [T, J, 3], if available
joints_2d          # [T, J, 2], if available/projected
joint_confidence   # [T, J], if available
bbox_xyxy          # [T, 4]
valid              # [T]
frame_indices      # [T]
```

### Logic

- Read raw SMPL estimator outputs.
- Convert tensors/lists to numpy arrays.
- Keep all per-frame validity/confidence information.
- Preserve frame indices so windows can be aligned later.
- Store enough camera/projection data to visualize the result.

### Acceptance

- Every `.npz` has consistent keys.
- Arrays have finite values where `valid=True`.
- Missing optional fields are documented in the manifest.

## Step 4: Build 64-frame SMPL Windows

### Purpose

Create training samples aligned with the current dataset window format.

### Input

```text
paper_like_mini_move_in_2d/bbox_lama_dataset_full_001/manifest_filtered.jsonl
smpl_processed/motions/<video_id>_smpl.npz
```

### Output

```text
paper_like_smpl_mini_move_in_2d/<run_name>/manifest_smpl.jsonl
paper_like_smpl_mini_move_in_2d/<run_name>/motions/<sample_id>_smpl.npz
paper_like_smpl_mini_move_in_2d/<run_name>/splits.json
```

### Logic

- Reuse existing `sample_id`, `video_id`, `scene_image_path`, `text_prompt`, and `split`.
- Slice the matching 64-frame SMPL interval for each sample.
- Reject samples where too many frames have invalid SMPL.
- Keep rejection reasons for reporting.

### Acceptance

- Dataset has train/val/test splits by `video_id`.
- No split leaks.
- Every manifest row points to an existing scene image and SMPL motion file.

## Step 5: Visual Validation

### Purpose

Make the SMPL output directly reviewable.

### Input

```text
paper_like_smpl_mini_move_in_2d/<run_name>/manifest_smpl.jsonl
```

### Output

```text
paper_like_smpl_mini_move_in_2d/<run_name>/previews/
```

Preview types:

```text
projected joints overlay on original frame
projected joints overlay on background image
side-by-side GT 2D vs SMPL projected joints
short mp4/contact sheet for several samples
```

### Logic

- Render SMPL projected joints if available.
- If mesh rendering is available, also render mesh preview.
- Compare against Penn Action 2D keypoints to detect gross alignment errors.
- Save previews, never only terminal logs.

### Acceptance

- Random sample previews are human-checkable.
- Bad cases have visible artifacts and logged reasons.
- The user can inspect the folder without rerunning code.

## Step 6: Comparison With 2D Branch

### Purpose

Measure whether the SMPL extraction is reliable enough to train on.

### Input

```text
processed/motions/*.npz
paper_like_smpl_mini_move_in_2d/<run_name>/motions/*_smpl.npz
```

### Output

```text
smpl_processed/comparison_with_2d/<run_name>/
```

Report metrics:

```text
2D reprojection error if projected joints are available
missing / invalid SMPL frame ratio
temporal pose jump score
bbox consistency
per-action success rate
example good/medium/bad previews
```

### Acceptance

- We know which actions/videos SMPL handles well.
- We have a concrete quality threshold before full training.
- The report can be used later for the project write-up.

## Step 7: Full Dataset Run

### Purpose

Run SMPL extraction across all accepted Penn Action samples/videos after the test run passes.

### Input

```text
all videos referenced by bbox_lama_dataset_full_001/manifest_filtered.jsonl
```

### Output

```text
paper_like_smpl_mini_move_in_2d/<full_run_name>/
```

### Logic

- Process video-level outputs first, then build sample-level windows.
- Resume from existing successful video outputs.
- Log failures per video.
- Save intermediate raw outputs and converted outputs.

### Acceptance

- Full SMPL manifest exists.
- Dataset report summarizes success/failure counts.
- Existing 2D dataset and training outputs remain untouched.

## Step 8: Training Branch After Dataset Is Ready

Training should be a separate phase after SMPL dataset QA.

Potential model target:

```text
CLIP text token + DINO scene tokens + noisy SMPL motion tokens
-> diffusion transformer
-> predicted SMPL noise
```

The first SMPL training target can be simplified:

```text
global_orient + body_pose + transl
```

Then optionally add:

```text
betas
camera
joint reprojection auxiliary loss
```

This should be implemented as a new model/training branch, not by overwriting the current 2D diffusion model.

## Commands To Add Later

Proposed scripts:

```text
scripts/smpl/setup_4d_humans.py
scripts/smpl/extract_smpl_test.py
scripts/smpl/convert_4dhumans_output.py
scripts/smpl/build_smpl_dataset.py
scripts/smpl/render_smpl_previews.py
scripts/smpl/compare_smpl_to_2d.py
scripts/smpl/run_smpl_pipeline.py
```

Proposed smoke commands:

```bash
python3 scripts/smpl/run_smpl_pipeline.py test --num-videos 5 --run-name smpl_test_001
python3 scripts/smpl/render_smpl_previews.py --run-name smpl_test_001
python3 scripts/smpl/compare_smpl_to_2d.py --run-name smpl_test_001
```

Full command after the test passes:

```bash
python3 scripts/smpl/run_smpl_pipeline.py full --run-name smpl_full_001
```

## Dataset Manifest Schema

Each final SMPL manifest row should include:

```text
sample_id
video_id
action_label
text_prompt
scene_image_path
smpl_motion_path
num_frames
image_width
image_height
smpl_stats
quality_flags
split
source_2d_motion_path
```

`source_2d_motion_path` is included only for comparison and debugging. The SMPL branch should train from `smpl_motion_path`.

## Dataset Report

The SMPL dataset report should include:

```text
number of videos processed
number of videos failed
number of final samples
split counts
per-action counts
invalid frame ratio distribution
pose jump distribution
2D vs SMPL projection comparison
good/medium/bad qualitative examples
known limitations
```

## Acceptance Criteria

- The current 2D dataset, code, checkpoints, and inference results are preserved.
- SMPL extraction has a test run on 3-5 videos with saved outputs.
- Converted SMPL `.npz` files use a stable schema.
- Preview images/videos allow direct human review.
- A comparison report against Penn Action 2D keypoints exists.
- Full dataset run is attempted only after the test run looks acceptable.

## Open Questions Before Coding

- Can 4D-Humans run reliably in the current environment, or do we need a separate environment?
- Are required SMPL model files already available, or must they be downloaded manually?
- Should the first SMPL training target include translation/camera, or only pose?
- Should we train on SMPL axis-angle directly, or convert to a more stable representation such as 6D rotations?

