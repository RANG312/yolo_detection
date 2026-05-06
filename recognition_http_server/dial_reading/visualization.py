from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Visualization helpers are deliberately side-effect free except for save_canvas.
def draw_box(image: np.ndarray, box_xyxy: np.ndarray, label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box_xyxy.astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_point(image: np.ndarray, point: np.ndarray, label: str, color: tuple[int, int, int]) -> None:
    x, y = point.astype(int)
    cv2.circle(image, (x, y), 5, color, -1)
    if label:
        cv2.putText(image, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def save_canvas(canvas: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
