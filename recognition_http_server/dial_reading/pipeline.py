from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from recognition_http_server.dial_reading.constants import LEGACY_EXPECTED_CLASSES, METER_DATA_9K_EXPECTED_CLASSES, METER_DATA_9K_GAUGE_RETRY_SIZE
from recognition_http_server.dial_reading.detections import (
    assign_meter_data_9k_instances,
    collect_all_detections,
    collect_best_detections,
    ensure_retry_gauge_detection,
    resize_gauge_crop_for_retry,
    sort_boxes_reading_order,
)
from recognition_http_server.dial_reading.geometry import compute_reading_from_detection_instance
from recognition_http_server.dial_reading.visualization import draw_box, draw_point

# 表计推理主流程：负责模型加载、YOLO 推理、检测结果转实例、几何读数和可视化。
# HTTP 服务和命令行入口都复用这里，避免两条链路出现行为差异。
def infer_annotation_mode(model: YOLO) -> str:
    """根据模型类别名判断标注协议，避免调用方手工传错后处理模式。"""
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
    """加载 YOLO 模型，并返回该模型对应的表计标注协议。"""
    model = YOLO(model_path)
    annotation_mode = infer_annotation_mode(model)
    return model, annotation_mode


def predict_image_instances(
    image_path: Path, model: YOLO, args: argparse.Namespace, annotation_mode: str
) -> tuple[np.ndarray, list[dict[str, Any]], float, float]:
    """
    对单张图执行表计识别，返回可视化图、逐表实例、推理耗时和几何耗时。

    `meter_data_9k` 模型先在原图中找 gauge，再把首个 gauge 裁剪缩放到训练尺寸
    进行第二次推理。这样能让中心点、刻度和指针 tip 在表盘 ROI 内获得更稳定的
    相对尺度。
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    inference_start = time.perf_counter()
    results = model.predict(source=image, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
    inference_time = time.perf_counter() - inference_start
    detections = collect_all_detections(results[0])

    if annotation_mode == "meter_data_9k":
        # 第一阶段只信任 gauge 定位；细粒度点位由裁剪后的第二阶段重新检测。
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
        # 有些模型在裁剪图中只输出点位不输出 gauge；此时裁剪图全幅就是表盘。
        retry_detections = ensure_retry_gauge_detection(retry_detections, retry_image.shape)
        try:
            detection_instances = assign_meter_data_9k_instances(retry_detections)
            image = retry_image
        except Exception as exc:
            # 保留原始 gauge 框作为错误实例，后续可视化仍能标出失败表盘。
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
            # 单个表盘失败不影响同图其他表盘，错误会被下沉到对应实例结果中。
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
    # 左上角结果面板用于人工抽检；服务返回结构化 JSON，不依赖这里的文字。
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
    """兼容旧脚本：只返回第一块识别成功表盘的读数。"""
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
