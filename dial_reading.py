from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import Any
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


LEGACY_EXPECTED_CLASSES = {"start", "point", "end"}
METER_DATA_9K_EXPECTED_CLASSES = {"gauge", "center", "pointer_tip", "max_tick", "min_tick"}
METER_DATA_9K_GAUGE_RETRY_SIZE = 640
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict dial reading from YOLO detections and ROI geometry.")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/train/meter_data_9k_yolov8m_20260407_145946/weights/best.pt",
        help="Model path (.pt or .onnx).",
    )
    parser.add_argument("--image", type=Path, help="Input image path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", type=str, default="0", help="Inference device, e.g. 'cpu' or '0'.")
    parser.add_argument("--min-value", type=float, default=0.0, help="Minimum meter value.")
    parser.add_argument("--max-value", type=float, default=25, help="Maximum meter value.")
    parser.add_argument(
        "--test-loop",
        action="store_true",
        help="Run all images under --input-dir and save visualizations to a new folder under --batch-save-root.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/new_datas/meter_data_9k/images/val"),
        help="Directory containing batch test images.",
    )
    parser.add_argument(
        "--batch-save-root",
        type=Path,
        default=Path("results"),
        help="Root directory used to create a timestamped batch result folder.",
    )
    parser.add_argument("--debug-center", action="store_true", help="Draw and print center-estimation intermediate points.")
    parser.add_argument("--show", action="store_true", help="Display the result image.")
    parser.add_argument("--save", type=Path, default=f"results/dial_reading_{timestamp}.png", help="Optional output image path.")
    args = parser.parse_args()
    if not args.test_loop and args.image is None:
        parser.error("--image is required unless --test-loop is used.")
    return args


def clip_box(box_xyxy: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    h, w = shape[:2]
    x1, y1, x2, y2 = box_xyxy.astype(int)
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return np.array([x1, y1, x2, y2], dtype=np.int32)


def crop_roi(image: np.ndarray, box_xyxy: np.ndarray, pad_ratio: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    if pad_ratio > 0.0:
        x1, y1, x2, y2 = box_xyxy.astype(np.float64)
        w = x2 - x1
        h = y2 - y1
        box_xyxy = np.array([x1 - w * pad_ratio, y1 - h * pad_ratio, x2 + w * pad_ratio, y2 + h * pad_ratio], dtype=np.float64)
    box = clip_box(box_xyxy, image.shape)
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2].copy(), box


def preprocess_edges(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, 50, 150)


def detect_segments(roi: np.ndarray, min_line_ratio: float = 0.25) -> list[np.ndarray]:
    edges = preprocess_edges(roi)
    min_len = max(10, int(min(roi.shape[:2]) * min_line_ratio))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=min_len, maxLineGap=10)
    if lines is None:
        return []
    return [line[0].astype(np.float64) for line in lines]


def line_length(segment: np.ndarray) -> float:
    x1, y1, x2, y2 = segment
    return float(np.hypot(x2 - x1, y2 - y1))


def segment_midpoint(segment: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = segment
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)


def line_alignment_to_center(segment: np.ndarray, center_local: np.ndarray) -> float:
    p1 = segment[:2]
    p2 = segment[2:]
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return 0.0
    midpoint = segment_midpoint(segment)
    radial = midpoint - center_local
    radial_norm = np.linalg.norm(radial)
    if radial_norm < 1e-6:
        return 0.0
    return float(abs(np.dot(direction / length, radial / radial_norm)))


def segment_direction(segment: np.ndarray) -> np.ndarray:
    direction = segment[2:] - segment[:2]
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float64)
    direction = direction / length
    # Normalize sign so parallel segments compare consistently.
    if direction[0] < 0 or (abs(direction[0]) < 1e-6 and direction[1] < 0):
        direction = -direction
    return direction.astype(np.float64)


def segment_border_margin(segment: np.ndarray, roi_shape: tuple[int, int, int]) -> float:
    h, w = roi_shape[:2]
    points = np.array([[segment[0], segment[1]], [segment[2], segment[3]]], dtype=np.float64)
    margins = np.stack([points[:, 0], points[:, 1], w - 1 - points[:, 0], h - 1 - points[:, 1]], axis=1)
    return float(margins.min())


def local_to_global(point: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    return point + np.array([box_xyxy[0], box_xyxy[1]], dtype=np.float64)


def box_center(box_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy.astype(np.float64)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)


def select_tick_endpoint(segment: np.ndarray, center_local: np.ndarray) -> np.ndarray:
    p1 = segment[:2]
    p2 = segment[2:]
    return p1 if np.linalg.norm(p1 - center_local) > np.linalg.norm(p2 - center_local) else p2


def combine_tick_support_points(
    segments: list[np.ndarray], primary_segment: np.ndarray, center_local: np.ndarray, roi_shape: tuple[int, int, int]
) -> np.ndarray:
    support_points = [select_tick_endpoint(primary_segment, center_local)]
    primary_direction = segment_direction(primary_segment)
    primary_midpoint = segment_midpoint(primary_segment)
    primary_endpoint_dist = np.linalg.norm(support_points[0] - center_local)
    midpoint_tol = max(8.0, min(roi_shape[:2]) * 0.18)
    endpoint_tol = max(10.0, min(roi_shape[:2]) * 0.22)

    for segment in segments:
        if np.array_equal(segment, primary_segment):
            continue
        if abs(np.dot(segment_direction(segment), primary_direction)) < 0.97:
            continue
        midpoint = segment_midpoint(segment)
        if np.linalg.norm(midpoint - primary_midpoint) > midpoint_tol:
            continue

        support_point = select_tick_endpoint(segment, center_local)
        endpoint_dist = np.linalg.norm(support_point - center_local)
        if abs(endpoint_dist - primary_endpoint_dist) > endpoint_tol:
            continue
        support_points.append(support_point)

    return np.mean(np.array(support_points, dtype=np.float64), axis=0)


def refine_dark_point(gray: np.ndarray, point_local: np.ndarray, window_radius: int = 14) -> np.ndarray:
    h, w = gray.shape
    x = int(round(point_local[0]))
    y = int(round(point_local[1]))
    x1 = max(0, x - window_radius)
    y1 = max(0, y - window_radius)
    x2 = min(w, x + window_radius + 1)
    y2 = min(h, y + window_radius + 1)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return point_local

    threshold = min(float(patch.mean()) - float(patch.std()) * 0.4, float(np.percentile(patch, 30)))
    mask = (patch <= threshold).astype(np.uint8)
    if mask.sum() < 8:
        return point_local

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return point_local

    target = point_local - np.array([x1, y1], dtype=np.float64)
    candidates = []
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < 6:
            continue
        centroid = centroids[idx]
        dist = np.linalg.norm(centroid - target)
        candidates.append((dist, -area, centroid))

    if not candidates:
        return point_local

    best = min(candidates, key=lambda item: (item[0], item[1]))[2]
    return np.array([best[0] + x1, best[1] + y1], dtype=np.float64)


def detect_farthest_dark_point(gray: np.ndarray, center_local: np.ndarray) -> np.ndarray | None:
    threshold = min(float(gray.mean()) - float(gray.std()) * 0.2, float(np.percentile(gray, 35)))
    ys, xs = np.where(gray <= threshold)
    if len(xs) == 0:
        return None

    candidates = np.stack([xs, ys], axis=1).astype(np.float64)
    distances = np.linalg.norm(candidates - center_local, axis=1)
    return candidates[int(np.argmax(distances))]


def refine_center_point(gray: np.ndarray, point_local: np.ndarray) -> np.ndarray:
    return refine_dark_point(gray, point_local, window_radius=9)


def detect_hub_center_full_roi(roi: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    dog = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    dog = cv2.addWeighted(enhanced, 1.55, dog, -0.55, 0.0)
    preprocesses = [gray, enhanced, dog]
    h, w = gray.shape
    hough_setups = [
        {"param1": 70, "param2": 16},
        {"param1": 90, "param2": 20},
    ]
    min_radius = max(8, min(h, w) // 22)
    max_radius = max(26, min(h, w) // 7)
    candidates: list[np.ndarray] = []
    for preprocessed in preprocesses:
        for setup in hough_setups:
            circles = cv2.HoughCircles(
                preprocessed,
                cv2.HOUGH_GRADIENT,
                dp=1.0,
                minDist=max(12, min(h, w) // 10),
                minRadius=min_radius,
                maxRadius=max_radius,
                **setup,
            )
            if circles is None:
                continue
            for circle in circles[0].astype(np.float64):
                candidates.append(circle)

    if not candidates:
        return None

    deduped: list[np.ndarray] = []
    for circle in candidates:
        if any(np.linalg.norm(circle[:2] - existing[:2]) <= 6.0 and abs(circle[2] - existing[2]) <= 4.0 for existing in deduped):
            continue
        deduped.append(circle)

    best_score = None
    best_center = None
    edges = cv2.Canny(enhanced, 40, 120)
    yy, xx = np.indices(gray.shape)
    target_radius = float(np.clip(min(h, w) * 0.11, min_radius, max_radius))
    for x, y, radius in deduped:
        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        ring_mask = np.abs(dist - radius) <= 2.0
        inner_mask = dist <= radius * 0.55
        annulus_mask = (dist >= radius * 0.8) & (dist <= radius * 1.2)
        if not np.any(ring_mask) or not np.any(inner_mask) or not np.any(annulus_mask):
            continue

        ring_support = float(edges[ring_mask].mean())
        contrast = float(abs(float(enhanced[inner_mask].mean()) - float(enhanced[annulus_mask].mean())))
        center_bias = float(np.linalg.norm(np.array([x, y], dtype=np.float64) - np.array([w / 2.0, h / 2.0], dtype=np.float64)))
        radius_bias = abs(float(radius) - target_radius)
        score = ring_support * 1.2 + contrast * 0.8 - radius_bias * 1.4 - center_bias * 0.08
        if best_score is None or score > best_score:
            best_score = score
            best_center = np.array([x, y], dtype=np.float64)

    return best_center


def refine_radial_endpoint(
    gray: np.ndarray, point_local: np.ndarray, center_local: np.ndarray, window_radius: int = 10
) -> np.ndarray:
    h, w = gray.shape
    x = int(round(point_local[0]))
    y = int(round(point_local[1]))
    x1 = max(0, x - window_radius)
    y1 = max(0, y - window_radius)
    x2 = min(w, x + window_radius + 1)
    y2 = min(h, y + window_radius + 1)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return point_local

    threshold = min(float(patch.mean()) - float(patch.std()) * 0.3, float(np.percentile(patch, 35)))
    ys, xs = np.where(patch <= threshold)
    if len(xs) == 0:
        return point_local

    candidates = np.stack([xs + x1, ys + y1], axis=1).astype(np.float64)
    radial = point_local - center_local
    radial_norm = np.linalg.norm(radial)
    if radial_norm < 1e-6:
        return point_local
    direction = radial / radial_norm

    rel = candidates - center_local
    axial = rel @ direction
    projections = center_local + np.outer(axial, direction)
    perp = np.linalg.norm(candidates - projections, axis=1)

    valid = perp <= max(2.5, window_radius * 0.35)
    if not np.any(valid):
        return point_local

    valid_candidates = candidates[valid]
    valid_axial = axial[valid]
    max_axial = float(valid_axial.max())
    axial_band = max(1.5, window_radius * 0.18)
    tip_band = valid_axial >= (max_axial - axial_band)
    best_cluster = valid_candidates[tip_band]
    if len(best_cluster) == 0:
        best_cluster = valid_candidates[[int(np.argmax(valid_axial))]]
    return np.mean(best_cluster, axis=0).astype(np.float64)


def refine_hub_circle_center(gray: np.ndarray, point_local: np.ndarray, patch_radius: int = 22) -> np.ndarray:
    h, w = gray.shape
    x = int(round(point_local[0]))
    y = int(round(point_local[1]))
    x1 = max(0, x - patch_radius)
    y1 = max(0, y - patch_radius)
    x2 = min(w, x + patch_radius + 1)
    y2 = min(h, y + patch_radius + 1)
    patch = gray[y1:y2, x1:x2]
    if min(patch.shape[:2]) < 12:
        return point_local

    blurred = cv2.medianBlur(patch, 5)
    target = point_local - np.array([x1, y1], dtype=np.float64)
    candidates: list[tuple[float, np.ndarray, float]] = []

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(6, min(patch.shape[:2]) // 4),
        param1=80,
        param2=12,
        minRadius=max(4, min(patch.shape[:2]) // 10),
        maxRadius=max(12, min(patch.shape[:2]) // 3),
    )
    if circles is not None:
        for circle in circles[0].astype(np.float64):
            center = circle[:2]
            radius = float(circle[2])
            score = float(np.linalg.norm(center - target) + abs(radius - 8.0) * 0.8)
            candidates.append((score, center, radius))

    edges = cv2.Canny(blurred, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 30:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 1e-6:
            continue
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < 0.45:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < 4.0 or radius > 14.0:
            continue
        center = np.array([cx, cy], dtype=np.float64)
        score = float(np.linalg.norm(center - target) + abs(radius - 8.0) * 0.5 - circularity * 6.0)
        candidates.append((score, center, float(radius)))

    if not candidates:
        return point_local

    best_score, best_center, best_radius = min(candidates, key=lambda item: item[0])
    if np.linalg.norm(best_center - target) > max(6.0, patch_radius * 0.28):
        return point_local
    if not (4.0 <= best_radius <= 12.0):
        return point_local
    best_x = int(round(best_center[0]))
    best_y = int(round(best_center[1]))
    target_x = int(round(target[0]))
    target_y = int(round(target[1]))
    best_x = max(0, min(best_x, patch.shape[1] - 1))
    best_y = max(0, min(best_y, patch.shape[0] - 1))
    target_x = max(0, min(target_x, patch.shape[1] - 1))
    target_y = max(0, min(target_y, patch.shape[0] - 1))
    # Only accept hub refinement when the new center is at least as dark as the current
    # estimate. This prevents bright dial marks from pulling the center away.
    if float(patch[best_y, best_x]) > float(patch[target_y, target_x]) - 3.0:
        return point_local

    best = best_center
    return np.array([best[0] + x1, best[1] + y1], dtype=np.float64)


def detect_concentric_center_local(roi: np.ndarray, preferred_center: np.ndarray | None = None) -> np.ndarray | None:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(6, min(h, w) // 16),
        param1=100,
        param2=18,
        minRadius=max(10, min(h, w) // 9),
        maxRadius=max(18, int(min(h, w) * 0.42)),
    )
    if circles is None:
        return None

    circles = np.round(circles[0]).astype(np.float64)
    grouped: list[list[np.ndarray]] = []
    for circle in circles:
        placed = False
        for group in grouped:
            center_mean = np.mean([c[:2] for c in group], axis=0)
            if np.linalg.norm(circle[:2] - center_mean) <= 10:
                group.append(circle)
                placed = True
                break
        if not placed:
            grouped.append([circle])

    if not grouped:
        return None

    roi_center = np.array([w / 2.0, h / 2.0], dtype=np.float64)

    def group_score(group: list[np.ndarray]) -> tuple[float, float, float, float, float]:
        centers = np.array([c[:2] for c in group], dtype=np.float64)
        radii = np.array([c[2] for c in group], dtype=np.float64)
        center_mean = centers.mean(axis=0)
        center_spread = float(np.mean(np.linalg.norm(centers - center_mean, axis=1)))
        radius_mean = float(radii.mean())
        center_bias = float(np.linalg.norm(center_mean - roi_center))
        preferred_bias = (
            float(np.linalg.norm(center_mean - preferred_center)) if preferred_center is not None else float("inf")
        )
        # Prioritize true concentric support first: more circles and tighter shared center
        # are stronger evidence than mere proximity to ROI center.
        return (len(group), -center_spread, -preferred_bias, -center_bias, radius_mean)

    best_group = max(grouped, key=group_score)
    center_local = np.mean([c[:2] for c in best_group], axis=0)
    return refine_center_point(gray, center_local)


def detect_pointer_root_candidate(
    roi: np.ndarray, anchor_local: np.ndarray | None = None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    segments = detect_segments(roi, min_line_ratio=0.3)
    if not segments:
        return None, None

    h, w = roi.shape[:2]
    anchor = anchor_local if anchor_local is not None else np.array([w / 2.0, h / 2.0], dtype=np.float64)

    def segment_score(segment: np.ndarray) -> tuple[float, float, float, float]:
        midpoint = segment_midpoint(segment)
        anchor_dist = point_line_distance(anchor, segment)
        midpoint_dist = np.linalg.norm(midpoint - anchor)
        endpoint_dist = max(np.linalg.norm(segment[:2] - anchor), np.linalg.norm(segment[2:] - anchor))
        return (anchor_dist, midpoint_dist, -endpoint_dist, -line_length(segment))

    chosen = min(segments, key=segment_score)
    p1 = chosen[:2]
    p2 = chosen[2:]
    root_local = p1 if np.linalg.norm(p1 - anchor) <= np.linalg.norm(p2 - anchor) else p2
    tip_local = p2 if np.allclose(root_local, p1) else p1
    return root_local.astype(np.float64), tip_local.astype(np.float64)


def detect_center_from_roi(
    roi: np.ndarray, box_xyxy: np.ndarray, debug_info: dict[str, np.ndarray | str] | None = None
) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    roi_center = np.array([w / 2.0, h / 2.0], dtype=np.float64)
    point_hub_center = detect_hub_center_full_roi(roi)
    roi_hub_center = refine_hub_circle_center(gray, roi_center, patch_radius=max(28, int(min(h, w) * 0.22)))
    initial_root_anchor = point_hub_center if point_hub_center is not None else roi_hub_center
    initial_root_candidate, _ = detect_pointer_root_candidate(roi, initial_root_anchor)
    circle_center = detect_concentric_center_local(roi, initial_root_candidate)
    root_anchor = point_hub_center if point_hub_center is not None else (circle_center if circle_center is not None else roi_hub_center)
    root_candidate, _ = detect_pointer_root_candidate(roi, root_anchor)
    pre_hub_center = None
    center_source = "roi_center"
    center_consistency_tol = max(12.0, min(h, w) * 0.08)

    if point_hub_center is not None and circle_center is not None:
        if np.linalg.norm(point_hub_center - circle_center) > center_consistency_tol:
            point_hub_center = None
    elif point_hub_center is not None and np.linalg.norm(point_hub_center - roi_center) > max(18.0, min(h, w) * 0.18):
        point_hub_center = None

    if point_hub_center is not None:
        center_local = point_hub_center
        center_source = "point_hub_center"
    elif circle_center is not None:
        center_local = circle_center
        center_source = "circle_center"
        if root_candidate is not None and np.linalg.norm(circle_center - root_candidate) <= max(8.0, min(h, w) * 0.05):
            # Pointer-root evidence can only make a small local correction when it already
            # agrees with the center inferred from circular structure.
            center_local = refine_center_point(gray, circle_center * 0.9 + root_candidate * 0.1)
            center_source = "blend(circle,root)"
    elif np.linalg.norm(roi_hub_center - roi_center) >= 2.0:
        center_local = roi_hub_center
        center_source = "roi_hub_center"
        if root_candidate is not None and np.linalg.norm(roi_hub_center - root_candidate) <= max(8.0, min(h, w) * 0.05):
            center_local = refine_center_point(gray, roi_hub_center * 0.9 + root_candidate * 0.1)
            center_source = "blend(roi_hub,root)"
    else:
        center_local = refine_center_point(gray, roi_center)
        center_source = "roi_center"

    pre_hub_center = center_local.copy()
    if center_source != "point_hub_center":
        center_local = refine_hub_circle_center(gray, center_local)
    center_global = local_to_global(center_local, box_xyxy)

    if debug_info is not None:
        debug_info["roi_center"] = local_to_global(roi_center, box_xyxy)
        if point_hub_center is not None:
            debug_info["point_hub_center"] = local_to_global(point_hub_center, box_xyxy)
        debug_info["roi_hub_center"] = local_to_global(roi_hub_center, box_xyxy)
        if circle_center is not None:
            debug_info["circle_center"] = local_to_global(circle_center, box_xyxy)
        if root_candidate is not None:
            debug_info["root_candidate"] = local_to_global(root_candidate, box_xyxy)
        if pre_hub_center is not None:
            debug_info["pre_hub_center"] = local_to_global(pre_hub_center, box_xyxy)
        debug_info["final_center"] = center_global
        debug_info["center_source"] = center_source

    return center_global


def point_line_distance(point: np.ndarray, segment: np.ndarray) -> float:
    p = point.astype(np.float64)
    a = segment[:2].astype(np.float64)
    b = segment[2:].astype(np.float64)
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-6:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    projection = a + t * ab
    return float(np.linalg.norm(p - projection))


def detect_tick_point(roi: np.ndarray, box_xyxy: np.ndarray, center_global: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    segments = detect_segments(roi, min_line_ratio=0.10)
    if segments:
        center_local = center_global - np.array([box_xyxy[0], box_xyxy[1]], dtype=np.float64)
        candidates: list[tuple[np.ndarray, float, float, float, float]] = []
        for segment in segments:
            length = line_length(segment)
            if length < max(8.0, min(roi.shape[:2]) * 0.10):
                continue
            border_margin = segment_border_margin(segment, roi.shape)
            if border_margin < 2.0:
                continue
            alignment = line_alignment_to_center(segment, center_local)
            midpoint = segment_midpoint(segment)
            radial_dist = np.linalg.norm(midpoint - center_local)
            endpoint_dist = max(np.linalg.norm(segment[:2] - center_local), np.linalg.norm(segment[2:] - center_local))
            candidates.append((segment, radial_dist, alignment, length, endpoint_dist))

        if candidates:
            def tick_score(item: tuple[np.ndarray, float, float, float, float]) -> tuple[float, float, float, float]:
                _, radial_dist, alignment, length, endpoint_dist = item
                return (-endpoint_dist, -length, -alignment, -radial_dist)

            segment = min(candidates, key=tick_score)[0]
        else:
            segment = max(segments, key=line_length)

        tick_local = combine_tick_support_points(segments, segment, center_local, roi.shape)
        tick_local = refine_radial_endpoint(gray, tick_local, center_local)
        return local_to_global(tick_local, box_xyxy)
    h, w = roi.shape[:2]
    return local_to_global(np.array([w / 2.0, h / 2.0], dtype=np.float64), box_xyxy)


def detect_pointer_tip(roi: np.ndarray, box_xyxy: np.ndarray, center_global: np.ndarray) -> np.ndarray:
    center_local = center_global - np.array([box_xyxy[0], box_xyxy[1]], dtype=np.float64)
    root_candidate, tip_candidate = detect_pointer_root_candidate(roi, center_local)
    if tip_candidate is not None:
        tip_local = tip_candidate if np.linalg.norm(tip_candidate - center_local) > np.linalg.norm(root_candidate - center_local) else root_candidate
        return local_to_global(tip_local, box_xyxy)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    segments = detect_segments(roi, min_line_ratio=0.3)
    candidates: list[tuple[np.ndarray, float, float, float]] = []
    for segment in segments:
        length = line_length(segment)
        if length < max(12.0, min(roi.shape[:2]) * 0.22):
            continue
        center_dist = point_line_distance(center_local, segment)
        endpoint_dist = max(np.linalg.norm(segment[:2] - center_local), np.linalg.norm(segment[2:] - center_local))
        midpoint_dist = np.linalg.norm(segment_midpoint(segment) - center_local)
        candidates.append((segment, center_dist, endpoint_dist, length + midpoint_dist * 0.15))

    if candidates:
        chosen = min(candidates, key=lambda item: (item[1], -item[2], -item[3]))[0]
        tip_local = select_tick_endpoint(chosen, center_local)
        tip_local = refine_radial_endpoint(gray, tip_local, center_local)
        return local_to_global(tip_local, box_xyxy)

    fallback = detect_farthest_dark_point(gray, center_local)
    if fallback is not None:
        fallback = refine_radial_endpoint(gray, fallback, center_local)
        return local_to_global(fallback, box_xyxy)

    h, w = roi.shape[:2]
    roi_center = np.array([w / 2.0, h / 2.0], dtype=np.float64)
    tip_local = roi_center if np.linalg.norm(roi_center - center_local) > 1e-6 else center_local.copy()
    return local_to_global(tip_local, box_xyxy)


def angle_from_center(center: np.ndarray, point: np.ndarray) -> float:
    vector = point - center
    return float(np.arctan2(-vector[1], vector[0]) % (2.0 * np.pi))


def ccw_distance(start: float, end: float) -> float:
    return float((end - start) % (2.0 * np.pi))


def select_arc_and_ratio(start_angle: float, end_angle: float, pointer_angle: float) -> tuple[float, float, str]:
    ccw_total = ccw_distance(start_angle, end_angle)
    ccw_pointer = ccw_distance(start_angle, pointer_angle)
    if ccw_pointer <= ccw_total:
        ratio = 0.0 if ccw_total < 1e-6 else ccw_pointer / ccw_total
        return ratio, ccw_total, "ccw"

    cw_total = ccw_distance(end_angle, start_angle)
    cw_pointer = ccw_distance(pointer_angle, start_angle)
    if cw_pointer <= cw_total:
        ratio = 0.0 if cw_total < 1e-6 else cw_pointer / cw_total
        return ratio, cw_total, "cw"

    if ccw_total >= cw_total:
        ratio = np.clip(0.0 if ccw_total < 1e-6 else ccw_pointer / ccw_total, 0.0, 1.0)
        return float(ratio), ccw_total, "ccw_clamped"

    ratio = np.clip(0.0 if cw_total < 1e-6 else ccw_distance(pointer_angle, start_angle) / cw_total, 0.0, 1.0)
    return float(ratio), cw_total, "cw_clamped"


def collect_best_detections(result) -> dict[str, dict[str, np.ndarray | float]]:
    detections: dict[str, dict[str, np.ndarray | float]] = {}
    names = result.names
    for box in result.boxes.cpu().numpy().data:
        xyxy = box[:4]
        conf = float(box[4])
        cls_name = names[int(box[5])]
        previous = detections.get(cls_name)
        if previous is None or conf > previous["conf"]:
            detections[cls_name] = {"box": xyxy, "conf": conf}
    return detections


def collect_all_detections(result) -> dict[str, list[dict[str, np.ndarray | float]]]:
    detections: dict[str, list[dict[str, np.ndarray | float]]] = {}
    names = result.names
    for box in result.boxes.cpu().numpy().data:
        xyxy = box[:4]
        conf = float(box[4])
        cls_name = names[int(box[5])]
        detections.setdefault(cls_name, []).append({"box": xyxy, "conf": conf})
    for values in detections.values():
        values.sort(key=lambda item: float(item["conf"]), reverse=True)
    return detections


def point_in_box(point: np.ndarray, box_xyxy: np.ndarray, margin: float = 0.0) -> bool:
    x1, y1, x2, y2 = box_xyxy.astype(np.float64)
    return bool((x1 - margin) <= point[0] <= (x2 + margin) and (y1 - margin) <= point[1] <= (y2 + margin))


def sort_boxes_reading_order(boxes: list[np.ndarray]) -> list[np.ndarray]:
    return sorted(boxes, key=lambda box: (float(box[1]), float(box[0])))


def resize_gauge_crop_for_retry(image: np.ndarray, gauge_box: np.ndarray) -> np.ndarray:
    gauge_roi, _ = crop_roi(image, gauge_box)
    return cv2.resize(
        gauge_roi,
        (METER_DATA_9K_GAUGE_RETRY_SIZE, METER_DATA_9K_GAUGE_RETRY_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )


def ensure_retry_gauge_detection(
    detections: dict[str, list[dict[str, np.ndarray | float]]],
    image_shape: tuple[int, int, int],
) -> dict[str, list[dict[str, np.ndarray | float]]]:
    if detections.get("gauge"):
        return detections

    point_classes = METER_DATA_9K_EXPECTED_CLASSES - {"gauge"}
    if not any(detections.get(name) for name in point_classes):
        return detections

    height, width = image_shape[:2]
    detections = {name: values.copy() for name, values in detections.items()}
    detections["gauge"] = [
        {
            "box": np.array([0.0, 0.0, float(width - 1), float(height - 1)], dtype=np.float64),
            "conf": 1.0,
        }
    ]
    return detections


def assign_meter_data_9k_instances(
    detections: dict[str, list[dict[str, np.ndarray | float]]],
) -> list[dict[str, dict[str, np.ndarray | float]]]:
    gauge_detections = detections.get("gauge", [])
    if not gauge_detections:
        raise ValueError("Missing required detections: ['gauge']")

    point_classes = ("center", "min_tick", "max_tick", "pointer_tip")
    missing_classes = [name for name in point_classes if not detections.get(name)]
    if missing_classes:
        raise ValueError(f"Missing required detections: {missing_classes}")

    sorted_gauges = sort_boxes_reading_order([item["box"] for item in gauge_detections])
    gauge_lookup = {
        tuple(np.asarray(item["box"], dtype=np.float64).tolist()): {"box": item["box"], "conf": item["conf"]}
        for item in gauge_detections
    }
    instances: list[dict[str, dict[str, np.ndarray | float]]] = []
    used_detection_ids: dict[str, set[int]] = {name: set() for name in point_classes}

    for recognize_image_index, gauge_box in enumerate(sorted_gauges, start=1):
        gauge_key = tuple(np.asarray(gauge_box, dtype=np.float64).tolist())
        gauge_detection = dict(gauge_lookup[gauge_key])
        gauge_detection["recognize_image_index"] = recognize_image_index
        gauge_center = box_center(gauge_box)
        gauge_size = max(float(gauge_box[2] - gauge_box[0]), float(gauge_box[3] - gauge_box[1]))
        assign_margin = max(6.0, gauge_size * 0.08)
        instance: dict[str, dict[str, np.ndarray | float]] = {"gauge": gauge_detection}

        for class_name in point_classes:
            candidates: list[tuple[tuple[float, float, float], int, dict[str, np.ndarray | float]]] = []
            for item in detections[class_name]:
                item_id = id(item)
                if item_id in used_detection_ids[class_name]:
                    continue
                item_box = item["box"]
                item_center = box_center(item_box)
                inside = point_in_box(item_center, gauge_box, margin=assign_margin)
                center_distance = float(np.linalg.norm(item_center - gauge_center))
                score = (0.0 if inside else 1.0, center_distance, -float(item["conf"]))
                candidates.append((score, item_id, item))

            if not candidates:
                continue

            best_score, best_id, best_item = min(candidates, key=lambda candidate: candidate[0])
            if best_score[0] > 0.0:
                continue
            selected_item = dict(best_item)
            selected_item["recognize_image_index"] = recognize_image_index
            instance[class_name] = selected_item
            used_detection_ids[class_name].add(best_id)

        instances.append(instance)

    return instances


def compute_reading_from_detection_instance(
    image: np.ndarray,
    instance: dict[str, dict[str, np.ndarray | float]],
    annotation_mode: str,
    debug_center: bool,
) -> dict[str, Any]:
    required_classes = METER_DATA_9K_EXPECTED_CLASSES if annotation_mode == "meter_data_9k" else LEGACY_EXPECTED_CLASSES
    missing = sorted(required_classes - set(instance))
    if missing:
        raise ValueError(f"Missing required detections for one meter: {missing}")

    center_debug: dict[str, np.ndarray | str] = {}
    gauge_box = None

    if annotation_mode == "meter_data_9k":
        gauge_box = instance["gauge"]["box"]
        center_box = instance["center"]["box"]
        min_tick_box = instance["min_tick"]["box"]
        max_tick_box = instance["max_tick"]["box"]
        pointer_tip_box = instance["pointer_tip"]["box"]

        center = box_center(center_box)
        start_tick = box_center(min_tick_box)
        end_tick = box_center(max_tick_box)
        pointer_tip = box_center(pointer_tip_box)

        if debug_center:
            center_debug["bbox_center"] = center.copy()
            center_debug["final_center"] = center.copy()
            center_debug["center_source"] = "bbox_center"
    else:
        start_roi, start_box = crop_roi(image, instance["start"]["box"])
        end_roi, end_box = crop_roi(image, instance["end"]["box"])
        point_roi, point_box = crop_roi(image, instance["point"]["box"])
        center = detect_center_from_roi(point_roi, point_box, center_debug if debug_center else None)
        start_tick = detect_tick_point(start_roi, start_box, center)
        end_tick = detect_tick_point(end_roi, end_box, center)
        pointer_tip = detect_pointer_tip(point_roi, point_box, center)
        center_box = None
        min_tick_box = None
        max_tick_box = None
        pointer_tip_box = None

    start_angle = angle_from_center(center, start_tick)
    end_angle = angle_from_center(center, end_tick)
    pointer_angle = angle_from_center(center, pointer_tip)
    ratio, arc_span, arc_mode = select_arc_and_ratio(start_angle, end_angle, pointer_angle)

    return {
        "gauge_box": gauge_box,
        "center_box": center_box,
        "min_tick_box": min_tick_box,
        "max_tick_box": max_tick_box,
        "pointer_tip_box": pointer_tip_box,
        "center": center,
        "start_tick": start_tick,
        "end_tick": end_tick,
        "pointer_tip": pointer_tip,
        "ratio": ratio,
        "arc_span": arc_span,
        "arc_mode": arc_mode,
        "center_debug": center_debug,
    }


def draw_box(image: np.ndarray, box_xyxy: np.ndarray, label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box_xyxy.astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_point(image: np.ndarray, point: np.ndarray, label: str, color: tuple[int, int, int]) -> None:
    x, y = point.astype(int)
    cv2.circle(image, (x, y), 5, color, -1)
    if label:
        cv2.putText(image, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def infer_annotation_mode(model: YOLO) -> str:
    model_names = set(model.names.values()) if isinstance(model.names, dict) else set(model.names)
    if METER_DATA_9K_EXPECTED_CLASSES.issubset(model_names):
        return "meter_data_9k"
    if LEGACY_EXPECTED_CLASSES.issubset(model_names):
        return "legacy"
    raise ValueError(
        "Model classes must include either "
        f"{sorted(METER_DATA_9K_EXPECTED_CLASSES)} or {sorted(LEGACY_EXPECTED_CLASSES)}, "
        f"got {sorted(model_names)}"
    )


def load_model(model_path: str) -> tuple[YOLO, str]:
    model = YOLO(model_path)
    annotation_mode = infer_annotation_mode(model)
    return model, annotation_mode


def predict_image_instances(
    image_path: Path, model: YOLO, args: argparse.Namespace, annotation_mode: str
) -> tuple[np.ndarray, list[dict[str, Any]], float, float]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    inference_start = time.perf_counter()
    results = model.predict(source=image, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
    inference_time = time.perf_counter() - inference_start
    detections = collect_all_detections(results[0])

    if annotation_mode == "meter_data_9k":
        gauge_detections = detections.get("gauge", [])
        if not gauge_detections:
            raise ValueError("Missing required detections: ['gauge']")

        sorted_gauge_boxes = sort_boxes_reading_order([item["box"] for item in gauge_detections])
        gauge_box = sorted_gauge_boxes[0]
        gauge_detection = next(
            item
            for item in gauge_detections
            if tuple(np.asarray(item["box"], dtype=np.float64).tolist())
            == tuple(np.asarray(gauge_box, dtype=np.float64).tolist())
        )
        retry_image = resize_gauge_crop_for_retry(image, gauge_box)

        retry_start = time.perf_counter()
        retry_results = model.predict(
            source=retry_image,
            imgsz=METER_DATA_9K_GAUGE_RETRY_SIZE,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        inference_time += time.perf_counter() - retry_start
        retry_detections = collect_all_detections(retry_results[0])
        retry_detections = ensure_retry_gauge_detection(retry_detections, retry_image.shape)
        try:
            detection_instances = assign_meter_data_9k_instances(retry_detections)
            image = retry_image
        except Exception as exc:
            detection_instances = [
                {
                    "gauge": {
                        "box": gauge_box,
                        "conf": gauge_detection["conf"],
                        "recognize_image_index": 1,
                    },
                    "_error": {"message": str(exc)},
                }
            ]
    else:
        best_detections = collect_best_detections(results[0])
        required_classes = LEGACY_EXPECTED_CLASSES
        missing = sorted(required_classes - set(best_detections))
        if missing:
            raise ValueError(f"Missing required detections: {missing}")
        detection_instances = [best_detections]

    compute_start = time.perf_counter()
    prediction_instances: list[dict[str, Any]] = []
    for index, detection_instance in enumerate(detection_instances, start=1):
        try:
            if "_error" in detection_instance:
                raise ValueError(str(detection_instance["_error"]["message"]))
            geometry = compute_reading_from_detection_instance(image, detection_instance, annotation_mode, args.debug_center)
            reading = args.min_value + geometry["ratio"] * (args.max_value - args.min_value)
            geometry["reading"] = reading
            geometry["recognize_image_index"] = int(detection_instance.get("gauge", {}).get("recognize_image_index", index))
            geometry["error"] = None
            prediction_instances.append(geometry)
        except Exception as exc:
            prediction_instances.append(
                {
                    "recognize_image_index": int(detection_instance.get("gauge", {}).get("recognize_image_index", index)),
                    "error": str(exc),
                    "gauge_box": detection_instance.get("gauge", {}).get("box"),
                    "center_debug": {},
                }
            )
    compute_time = time.perf_counter() - compute_start

    canvas = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for instance in prediction_instances:
        recognize_image_index = instance["recognize_image_index"]
        if instance.get("error") is not None:
            if isinstance(instance.get("gauge_box"), np.ndarray):
                draw_box(canvas, instance["gauge_box"], f"gauge_{recognize_image_index}_error", (0, 165, 255))
            continue
        if annotation_mode == "meter_data_9k":
            draw_box(canvas, instance["gauge_box"], f"gauge_{recognize_image_index}", (0, 255, 255))
            draw_box(canvas, instance["min_tick_box"], f"min_tick_{recognize_image_index}", (0, 0, 255))
            draw_box(canvas, instance["max_tick_box"], f"max_tick_{recognize_image_index}", (255, 215, 0))
            draw_box(canvas, instance["center_box"], f"center_{recognize_image_index}", (255, 0, 255))
            draw_box(canvas, instance["pointer_tip_box"], f"pointer_tip_{recognize_image_index}", (0, 255, 0))
        else:
            draw_box(canvas, detection_instances[0]["start"]["box"], "start", (0, 0, 255))
            draw_box(canvas, detection_instances[0]["end"]["box"], "end", (255, 215, 0))
            draw_box(canvas, detection_instances[0]["point"]["box"], "point", (0, 255, 0))

        draw_point(canvas, instance["center"], "", (255, 0, 255))
        draw_point(canvas, instance["start_tick"], "", (0, 0, 255))
        draw_point(canvas, instance["end_tick"], "", (255, 215, 0))
        draw_point(canvas, instance["pointer_tip"], "", (0, 255, 0))
        cv2.line(canvas, tuple(instance["center"].astype(int)), tuple(instance["start_tick"].astype(int)), (0, 0, 255), 2)
        cv2.line(canvas, tuple(instance["center"].astype(int)), tuple(instance["end_tick"].astype(int)), (255, 215, 0), 2)
        cv2.line(
            canvas,
            tuple(instance["center"].astype(int)),
            tuple(instance["pointer_tip"].astype(int)),
            (0, 255, 0),
            2,
        )

    overlay = canvas.copy()
    panel_height = 18 + 30 * max(1, len(prediction_instances))
    cv2.rectangle(overlay, (12, 12), (360, 12 + panel_height), (245, 245, 245), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0.0, canvas)
    for row_index, instance in enumerate(prediction_instances, start=1):
        y = 20 + row_index * 24
        if instance.get("error") is not None:
            text = f"#{instance['recognize_image_index']}: error"
        else:
            text = f"#{instance['recognize_image_index']}: {instance['reading']:.3f} ({instance['arc_mode']})"
        cv2.putText(canvas, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 30, 30), 2, cv2.LINE_AA)

        if args.debug_center and instance.get("error") is None:
            debug_points = [
                ("bbox_center", (255, 0, 255)),
                ("roi_center", (160, 160, 160)),
                ("point_hub_center", (255, 255, 255)),
                ("roi_hub_center", (0, 128, 255)),
                ("circle_center", (0, 255, 255)),
                ("root_candidate", (255, 128, 0)),
                ("pre_hub_center", (255, 105, 180)),
            ]
            for key, color in debug_points:
                point = instance["center_debug"].get(key)
                if isinstance(point, np.ndarray):
                    draw_point(canvas, point, f"{key}_{instance['recognize_image_index']}", color)
            if isinstance(instance["center_debug"].get("pre_hub_center"), np.ndarray):
                cv2.line(
                    canvas,
                    tuple(instance["center_debug"]["pre_hub_center"].astype(int)),
                    tuple(instance["center"].astype(int)),
                    (255, 105, 180),
                    1,
                )

    return canvas, prediction_instances, inference_time, compute_time


def predict_single_image(
    image_path: Path, model: YOLO, args: argparse.Namespace, annotation_mode: str
) -> tuple[np.ndarray, float, str, float, float, dict[str, np.ndarray | str]]:
    canvas, prediction_instances, inference_time, compute_time = predict_image_instances(
        image_path, model, args, annotation_mode
    )
    successful_instances = [instance for instance in prediction_instances if instance.get("error") is None]
    if not successful_instances:
        raise ValueError("No meter instances were recognized.")

    first_instance = successful_instances[0]
    return (
        canvas,
        float(first_instance["reading"]),
        str(first_instance["arc_mode"]),
        inference_time,
        compute_time,
        first_instance["center_debug"],
    )


def save_canvas(canvas: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def run_test_loop(args: argparse.Namespace, model: YOLO) -> None:
    image_paths = sorted(
        [path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.input_dir}")

    batch_dir = args.batch_save_root / f"dial_reading_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    summary_path = batch_dir / "summary.csv"

    rows: list[dict[str, str]] = []
    failures = 0
    for index, image_path in enumerate(image_paths, start=1):
        try:
            canvas, reading, arc_mode, inference_time, compute_time, _ = predict_single_image(
                image_path, model, args, args.annotation_mode
            )
            save_path = batch_dir / image_path.name
            save_canvas(canvas, save_path)
            rows.append(
                {
                    "image": image_path.name,
                    "reading": f"{reading:.6f}",
                    "arc_mode": arc_mode,
                    "inference_ms": f"{inference_time * 1000:.3f}",
                    "compute_ms": f"{compute_time * 1000:.3f}",
                    "status": "ok",
                    "error": "",
                }
            )
            print(f"[{index}/{len(image_paths)}] ok {image_path.name} -> {save_path.name} reading={reading:.3f}")
        except Exception as exc:
            failures += 1
            rows.append(
                {
                    "image": image_path.name,
                    "reading": "",
                    "arc_mode": "",
                    "inference_ms": "",
                    "compute_ms": "",
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(image_paths)}] error {image_path.name}: {exc}")

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "reading", "arc_mode", "inference_ms", "compute_ms", "status", "error"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed images: {len(image_paths)}")
    print(f"Failures: {failures}")
    print(f"Batch results saved to: {batch_dir}")
    print(f"Summary saved to: {summary_path}")


def main() -> None:
    args = parse_args()
    model_load_start = time.perf_counter()
    model, args.annotation_mode = load_model(args.model)
    model_load_time = time.perf_counter() - model_load_start
    print(f"Model load time: {model_load_time * 1000:.1f} ms")
    print(f"Annotation mode: {args.annotation_mode}")

    if args.test_loop:
        run_test_loop(args, model)
        return

    canvas, reading, arc_mode, inference_time, compute_time, center_debug = predict_single_image(
        args.image, model, args, args.annotation_mode
    )
    print(f"Model inference time: {inference_time * 1000:.1f} ms")
    print(f"Reading compute time: {compute_time * 1000:.1f} ms")
    print(f"Reading range: {args.min_value} -> {args.max_value}")
    print(f"Predicted reading: {reading:.3f}")
    print(f"Arc mode: {arc_mode}")
    if args.debug_center:
        print(f"Center source: {center_debug.get('center_source')}")
        for key in (
            "bbox_center",
            "roi_center",
            "point_hub_center",
            "roi_hub_center",
            "circle_center",
            "root_candidate",
            "pre_hub_center",
            "final_center",
        ):
            point = center_debug.get(key)
            if isinstance(point, np.ndarray):
                print(f"{key}: ({point[0]:.1f}, {point[1]:.1f})")

    if args.save is not None:
        save_canvas(canvas, args.save)
        print(f"Saved visualization to: {args.save}")

    if args.show:
        plt.imshow(canvas)
        plt.title("Dial Reading")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
