# Paper-like Mini Dataset: bbox + LaMa

## Summary

- samples: 50
- videos: 39
- failures: 0
- scene method: padded Penn Action bbox mask + LaMa
- bbox mask pad: 32px
- motion source: existing Penn Action 2D keypoint motion files
- text source: existing action-label prompt templates

## Splits

- test: 7
- train: 28
- val: 15

## Actions

- baseball_pitch: 4
- baseball_swing: 4
- bench_press: 4
- bowl: 4
- clean_and_jerk: 4
- golf_swing: 3
- jump_rope: 3
- jumping_jacks: 3
- pullup: 3
- pushup: 3
- situp: 3
- squat: 3
- strum_guitar: 3
- tennis_forehand: 3
- tennis_serve: 3

## Files

- manifest: `manifest_filtered.jsonl`
- splits: `splits.json`
- scenes: `scenes/`
- bbox masks: `bbox_masks/`
- previews: `previews/`
- failures: `failures.jsonl`

## Notes

- This is the selected background-removal branch for dataset v1.
- It intentionally keeps GT/Penn Action motion for the first trainable dataset, while replacing original scene frames with human-removed LaMa backgrounds.
- Preview images are three panels: original frame, bbox mask, LaMa background.
