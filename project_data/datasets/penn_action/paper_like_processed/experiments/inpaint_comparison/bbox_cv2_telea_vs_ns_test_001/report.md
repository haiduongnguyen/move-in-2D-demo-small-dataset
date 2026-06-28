# Inpaint Comparison: OpenCV TELEA vs NS

- Mask source: GT bbox rectangle baseline
- Samples: 5
- Important limitation: this comparison isolates OpenCV inpaint method only; mask quality is still the dominant failure mode.

## Files

- 0001_000: `project_data/datasets/penn_action/paper_like_processed/experiments/inpaint_comparison/bbox_cv2_telea_vs_ns_test_001/contact_sheets/0001_000_telea_vs_ns.jpg`
- 0001_001: `project_data/datasets/penn_action/paper_like_processed/experiments/inpaint_comparison/bbox_cv2_telea_vs_ns_test_001/contact_sheets/0001_001_telea_vs_ns.jpg`
- 0001_002: `project_data/datasets/penn_action/paper_like_processed/experiments/inpaint_comparison/bbox_cv2_telea_vs_ns_test_001/contact_sheets/0001_002_telea_vs_ns.jpg`
- 0001_003: `project_data/datasets/penn_action/paper_like_processed/experiments/inpaint_comparison/bbox_cv2_telea_vs_ns_test_001/contact_sheets/0001_003_telea_vs_ns.jpg`
- 0002_000: `project_data/datasets/penn_action/paper_like_processed/experiments/inpaint_comparison/bbox_cv2_telea_vs_ns_test_001/contact_sheets/0002_000_telea_vs_ns.jpg`

## Initial Assessment

- Both methods struggle because the mask is a large rectangle, not a person silhouette.
- TELEA tends to smear nearby texture into the removed region.
- NS is not consistently better on large human-sized holes; it can preserve edges slightly differently but still leaves visible artifacts.
- Next real improvement should be person segmentation mask first, then LaMa/object-removal quality inpainting if OpenCV remains poor.
