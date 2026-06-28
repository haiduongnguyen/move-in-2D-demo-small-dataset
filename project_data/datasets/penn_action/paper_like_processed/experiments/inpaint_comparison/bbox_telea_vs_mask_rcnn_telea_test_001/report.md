# Inpaint Comparison: BBox Mask vs Mask R-CNN Mask

- Samples: 5
- Inpaint method: OpenCV TELEA for both
- BBox baseline uses GT bbox rectangle.
- Mask R-CNN run uses OpenCV DNN Mask R-CNN person mask, with GT bbox only for instance selection.

## Files

- 0001_000: `project_data/datasets/penn_action/paper_like_processed/experiments/inpaint_comparison/bbox_telea_vs_mask_rcnn_telea_test_001/contact_sheets/0001_000_bbox_vs_mask_rcnn.jpg`; score=0.9962706565856934; IoU_to_GT_bbox=0.5821172885511262
- 0001_001: `project_data/datasets/penn_action/paper_like_processed/experiments/inpaint_comparison/bbox_telea_vs_mask_rcnn_telea_test_001/contact_sheets/0001_001_bbox_vs_mask_rcnn.jpg`; score=0.9968562126159668; IoU_to_GT_bbox=0.5714611270411746
- 0001_002: `project_data/datasets/penn_action/paper_like_processed/experiments/inpaint_comparison/bbox_telea_vs_mask_rcnn_telea_test_001/contact_sheets/0001_002_bbox_vs_mask_rcnn.jpg`; score=0.9995927214622498; IoU_to_GT_bbox=0.6220511790741714
- 0001_003: `project_data/datasets/penn_action/paper_like_processed/experiments/inpaint_comparison/bbox_telea_vs_mask_rcnn_telea_test_001/contact_sheets/0001_003_bbox_vs_mask_rcnn.jpg`; score=0.9874815940856934; IoU_to_GT_bbox=0.46337683490930515
- 0002_000: `project_data/datasets/penn_action/paper_like_processed/experiments/inpaint_comparison/bbox_telea_vs_mask_rcnn_telea_test_001/contact_sheets/0002_000_bbox_vs_mask_rcnn.jpg`; score=0.9992792010307312; IoU_to_GT_bbox=0.7236349189848067

## Initial Assessment

- Mask R-CNN gives a smaller, person-shaped mask, so it preserves more background than bbox masking.
- OpenCV TELEA still leaves visible artifacts where the person covers structured background.
- This is an incremental improvement over bbox masking, but a stronger inpainter such as LaMa is still the likely next useful upgrade.
