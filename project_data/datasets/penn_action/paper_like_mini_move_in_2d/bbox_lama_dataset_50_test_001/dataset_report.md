# Paper-like Mini Dataset: bbox + LaMa

## Summary

- samples: 50
- videos: 25
- failures: 0
- scene method: padded Penn Action bbox mask + LaMa
- bbox mask pad: 32px
- motion source: existing Penn Action 2D keypoint motion files
- text source: existing action-label prompt templates

## Splits

- train: 41
- val: 9

## Actions

- baseball_pitch: 50

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
