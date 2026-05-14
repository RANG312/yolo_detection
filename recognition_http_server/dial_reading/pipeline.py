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
)
from recognition_http_server.dial_reading.geometry import compute_reading_from_detection_instance
from recognition_http_server.ptz_alignment import maybe_align_gauge, maybe_zoom_gauge
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


def _save_ptz_capture(args: argparse.Namespace, image: np.ndarray, label: str, index: int) -> None:
    capture_dir = getattr(args, "ptz_capture_dir", None)
    if capture_dir is None:
        return
    capture_stem = str(getattr(args, "ptz_capture_stem", "meter"))
    capture_path = Path(capture_dir) / f"{capture_stem}_{label}_{index}.jpg"
    capture_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(capture_path), image)


def _save_aligned_capture(args: argparse.Namespace, image: np.ndarray) -> None:
    aligned_path = getattr(args, "ptz_aligned_image_path", None)
    if aligned_path is None:
        return
    aligned_path = Path(aligned_path)
    aligned_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(aligned_path), image)


def _refresh_ptz_fov(args: argparse.Namespace, config):  # noqa: ANN001, ANN202
    provider = getattr(args, "ptz_fov_provider", None)
    if provider is None or config is None or not getattr(config, "enabled", False):
        return config
    resolved = provider(config)
    if isinstance(resolved, tuple):
        refreshed_config, refreshed_controller = resolved
        args.ptz_controller = refreshed_controller
        return refreshed_config
    return resolved


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
        ptz_alignment_config = getattr(args, "ptz_alignment_config", None)
        ptz_controller = getattr(args, "ptz_controller", None)
        ptz_logger = getattr(args, "ptz_logger", None)
        ptz_enabled = bool(getattr(ptz_alignment_config, "enabled", False)) if ptz_alignment_config is not None else False
        max_alignment_passes = 1
        if ptz_alignment_config is not None:
            max_alignment_passes = max(1, int(getattr(ptz_alignment_config, "max_passes", 1)))

        for alignment_pass in range(1, max_alignment_passes + 1):
            gauge_detections = detections.get("gauge", [])
            if not gauge_detections:
                raise ValueError("Missing required detections: ['gauge']")

            gauge_detection = max(
                gauge_detections,
                key=lambda item: float(item["conf"]),
            )
            gauge_box = np.asarray(gauge_detection["box"], dtype=np.float64)
            if ptz_alignment_config is None:
                break
            ptz_alignment_config = _refresh_ptz_fov(args, ptz_alignment_config)
            ptz_controller = getattr(args, "ptz_controller", ptz_controller)
            try:
                ptz_request = maybe_align_gauge(
                    ptz_controller,
                    ptz_alignment_config,
                    image.shape,
                    gauge_box,
                )
            except Exception as exc:  # noqa: BLE001
                if ptz_logger is not None:
                    ptz_logger.exception("meter ptz alignment skipped after sdk error: %s", exc)
                break
            if ptz_request is not None and ptz_logger is not None:
                ptz_logger.info(
                    (
                        "meter ptz alignment checked: pass=%s should_align=%s dx_px=%.3f dy_px=%.3f "
                        "pan_delta=%.3f tilt_delta=%.3f hfov=%.3f vfov=%.3f gauge_box=%s"
                    ),
                    alignment_pass,
                    ptz_request.should_align,
                    ptz_request.dx_px,
                    ptz_request.dy_px,
                    ptz_request.pan_delta_deg,
                    ptz_request.tilt_delta_deg,
                    ptz_alignment_config.horizontal_fov_deg,
                    ptz_alignment_config.vertical_fov_deg,
                    ptz_request.gauge_box,
                )
            if ptz_request is None or not ptz_request.should_align:
                break
            if ptz_controller is None or not hasattr(ptz_controller, "capture_image"):
                break
            image = ptz_controller.capture_image()
            _save_ptz_capture(args, image, "align", alignment_pass)
            inference_start = time.perf_counter()
            results = model.predict(source=image, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
            inference_time += time.perf_counter() - inference_start
            detections = collect_all_detections(results[0])
            gauge_detections = detections.get("gauge", [])
            if not gauge_detections:
                raise ValueError("Missing required detections: ['gauge']")
            gauge_detection = max(
                gauge_detections,
                key=lambda item: float(item["conf"]),
            )
            gauge_box = np.asarray(gauge_detection["box"], dtype=np.float64)
            if alignment_pass >= max_alignment_passes:
                break

        if ptz_alignment_config is not None:
            max_zoom_passes = max(0, int(getattr(ptz_alignment_config, "zoom_max_passes", 0)))
            for zoom_pass in range(1, max_zoom_passes + 1):
                try:
                    ptz_zoom_request = maybe_zoom_gauge(
                        ptz_controller,
                        ptz_alignment_config,
                        image.shape,
                        gauge_box,
                    )
                except Exception as exc:  # noqa: BLE001
                    if ptz_logger is not None:
                        ptz_logger.exception("meter ptz zoom skipped after sdk error: %s", exc)
                    break
                if ptz_zoom_request is not None and ptz_logger is not None:
                    ptz_logger.info(
                        (
                            "meter ptz zoom checked: pass=%s should_zoom=%s direction=%s "
                            "height_ratio=%.3f target=%.3f tolerance=%.3f gauge_box=%s"
                        ),
                        zoom_pass,
                        ptz_zoom_request.should_zoom,
                        ptz_zoom_request.zoom_direction,
                        ptz_zoom_request.current_height_ratio,
                        ptz_zoom_request.target_height_ratio,
                        ptz_zoom_request.ratio_tolerance,
                        ptz_zoom_request.gauge_box,
                    )
                if ptz_zoom_request is None or not ptz_zoom_request.should_zoom:
                    break
                if ptz_controller is None or not hasattr(ptz_controller, "capture_image"):
                    break
                image = ptz_controller.capture_image()
                _save_ptz_capture(args, image, "zoom", zoom_pass)
                inference_start = time.perf_counter()
                results = model.predict(source=image, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
                inference_time += time.perf_counter() - inference_start
                detections = collect_all_detections(results[0])
                gauge_detections = detections.get("gauge", [])
                if not gauge_detections:
                    raise ValueError("Missing required detections: ['gauge']")
                gauge_detection = max(
                    gauge_detections,
                    key=lambda item: float(item["conf"]),
                )
                gauge_box = np.asarray(gauge_detection["box"], dtype=np.float64)

        if ptz_enabled:
            _save_aligned_capture(args, image)

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
