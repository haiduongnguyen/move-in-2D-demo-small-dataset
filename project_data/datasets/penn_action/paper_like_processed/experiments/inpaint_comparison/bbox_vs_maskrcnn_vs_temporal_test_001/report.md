# Inpaint Comparison: BBox vs Mask R-CNN vs Temporal Fill

- Samples: 5
- BBox and Mask R-CNN rows use OpenCV TELEA only.
- Temporal fill uses nearby video frames for clean pixels, then OpenCV TELEA only for remaining holes.

## Files

- 0001_000: `project_data/datasets/penn_action/paper_like_processed/experiments/inpaint_comparison/bbox_vs_maskrcnn_vs_temporal_test_001/contact_sheets/0001_000_comparison.jpg`; temporal_coverage=0.3509; neighbors=12
- 0001_001: `project_data/datasets/penn_action/paper_like_processed/experiments/inpaint_comparison/bbox_vs_maskrcnn_vs_temporal_test_001/contact_sheets/0001_001_comparison.jpg`; temporal_coverage=0.4563; neighbors=12
- 0001_002: `project_data/datasets/penn_action/paper_like_processed/experiments/inpaint_comparison/bbox_vs_maskrcnn_vs_temporal_test_001/contact_sheets/0001_002_comparison.jpg`; temporal_coverage=0.9256; neighbors=12
- 0001_003: `project_data/datasets/penn_action/paper_like_processed/experiments/inpaint_comparison/bbox_vs_maskrcnn_vs_temporal_test_001/contact_sheets/0001_003_comparison.jpg`; temporal_coverage=0.6835; neighbors=12
- 0002_000: `project_data/datasets/penn_action/paper_like_processed/experiments/inpaint_comparison/bbox_vs_maskrcnn_vs_temporal_test_001/contact_sheets/0002_000_comparison.jpg`; temporal_coverage=1.0000; neighbors=12

## Initial Assessment

- Temporal fill can improve static-camera cases because it reuses real pixels from neighboring frames.
- It can fail or create ghosting when the background/camera changes or when neighboring frames still cover the same region.
- For Penn Action baseball samples, this is worth keeping as a baseline before moving to LaMa.
