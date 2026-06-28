from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


PENN_ACTION_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (7, 10),
    (10, 11),
    (11, 12),
]


def draw_skeleton(
    image: np.ndarray,
    keypoints_px: np.ndarray,
    visibility: np.ndarray,
    line_color: tuple[int, int, int] = (0, 255, 80),
    point_color: tuple[int, int, int] = (30, 30, 255),
) -> np.ndarray:
    out = image.copy()
    for a, b in PENN_ACTION_EDGES:
        if _is_visible_pair(keypoints_px, visibility, a, b):
            pa = tuple(np.round(keypoints_px[a]).astype(int))
            pb = tuple(np.round(keypoints_px[b]).astype(int))
            cv2.line(out, pa, pb, line_color, 3, cv2.LINE_AA)
    for idx, point in enumerate(keypoints_px):
        if idx < len(visibility) and visibility[idx] and np.isfinite(point).all():
            cv2.circle(out, tuple(np.round(point).astype(int)), 4, point_color, -1, cv2.LINE_AA)
    return out


def render_motion_contact_sheet(
    background_bgr: np.ndarray,
    keypoints_px: np.ndarray,
    visibility: np.ndarray,
    output_path: str | Path,
    label: str,
    frames: int = 8,
) -> None:
    frame_ids = np.linspace(0, len(keypoints_px) - 1, frames).round().astype(int)
    panels = []
    for frame_id in frame_ids:
        panel = draw_skeleton(background_bgr, keypoints_px[frame_id], visibility[frame_id])
        _write_label(panel, f"{label} f={frame_id}")
        panels.append(panel)
    sheet = np.concatenate(panels, axis=1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def render_motion_video(
    background_bgr: np.ndarray,
    keypoints_px: np.ndarray,
    visibility: np.ndarray,
    output_path: str | Path,
    label: str,
    fps: int = 12,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = background_bgr.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame_id in range(len(keypoints_px)):
        frame = draw_skeleton(background_bgr, keypoints_px[frame_id], visibility[frame_id])
        _write_label(frame, f"{label} {frame_id + 1}/{len(keypoints_px)}")
        writer.write(frame)
    writer.release()


def _is_visible_pair(keypoints_px: np.ndarray, visibility: np.ndarray, a: int, b: int) -> bool:
    return (
        a < len(visibility)
        and b < len(visibility)
        and bool(visibility[a])
        and bool(visibility[b])
        and np.isfinite(keypoints_px[[a, b]]).all()
    )


def _write_label(image: np.ndarray, label: str) -> None:
    cv2.putText(image, label[:80], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

