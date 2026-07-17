from __future__ import annotations

import cv2
import numpy as np

from recognition_http_server.dial_reading.constants import (
    METER_DATA_9K_EXPECTED_CLASSES,
    METER_DATA_9K_GAUGE_RETRY_SIZE,
)
from recognition_http_server.dial_reading.geometry import box_center, crop_roi


# 检测结果整理模块：把 YOLO 原始框转换成“每块表一个实例”的结构。
# 几何模块只关心实例里的 gauge/center/tick/tip，不直接依赖 YOLO Results 对象。
def collect_best_detections(result) -> dict[str, dict[str, np.ndarray | float]]:
    """旧版三类标注只需要每类置信度最高的一个框。."""
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
    """9k 标注可能有多块表，需要保留每个类别的全部候选框。."""
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
    """按从上到下、从左到右的阅读顺序给多块表编号。."""
    return sorted(boxes, key=lambda box: (float(box[1]), float(box[0])))


def resize_gauge_crop_for_retry(image: np.ndarray, gauge_box: np.ndarray) -> np.ndarray:
    """裁剪第一阶段 gauge，并缩放到 9k 模型训练时使用的正方形输入尺寸。."""
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
    """兜底补全裁剪图上的 gauge。.

    第二阶段输入已经是单块表盘 ROI，如果模型只检测到点位而漏掉 gauge，可以把 整张裁剪图视为 gauge，避免因为一个冗余框缺失而丢弃有效点位。
    """
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
    """将 9k 多类别检测框分配到对应表盘实例。.

    每个点位类别最多分给一块表；优先选择中心落在 gauge 内、距离 gauge 中心近、 置信度高的候选框。这样能减少相邻表盘点位互相串用。
    """
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
            # score 越小越优：先保证点位在表盘范围内，再看空间距离和置信度。
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
