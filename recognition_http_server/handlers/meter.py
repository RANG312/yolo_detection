from __future__ import annotations

from pathlib import Path

import cv2

from dial_reading import predict_image_instances, save_canvas
from recognition_http_server.helpers import build_error_data_entry, build_meter_predict_args, build_success_data_entry


def run_pointer_meter_recognition(service, local_image_path: Path, data_types: list[dict[str, str]], visualize_path: Path, debug_center: bool, scale: float) -> list[dict[str, str]]:
    """
    执行指针表计读数，并返回应用量程后的最终业务值。

    底层表盘模型输出的仍是归一化读数，这里会在返回前叠加解析后的满量
    程配置，得到最终读数。
    """
    predict_args = build_meter_predict_args(service.config, visualize_path, debug_center=debug_center)
    service.logger.info("meter recognition running: image=%s output=%s", local_image_path, visualize_path)
    with service._predict_lock:
        canvas, prediction_instances, _, _ = predict_image_instances(
            local_image_path, service.meter_model, predict_args, service.config.annotation_mode
        )
    recognize_items: list[dict[str, str]] = []
    has_success = False
    for meter_instance in prediction_instances:
        recognize_image_index = str(meter_instance["recognize_image_index"])
        if meter_instance.get("error") is None:
            has_success = True
            normalized_reading = float(meter_instance["reading"])
            final_reading = normalized_reading * scale
            arc_mode = str(meter_instance["arc_mode"])
            for data_type in data_types:
                desc = (
                    f"识别成功，表计#{recognize_image_index}归一化读数={normalized_reading:.6f}，"
                    f"量程={scale:.6f}，最终读数={final_reading:.6f}，arc_mode={arc_mode}"
                )
                recognize_item = build_success_data_entry(data_type, final_reading, desc)
                recognize_item["recognize_image_index"] = recognize_image_index
                recognize_items.append(recognize_item)
        else:
            error_text = str(meter_instance["error"])
            for data_type in data_types:
                recognize_item = build_error_data_entry(data_type, error_text)
                recognize_item["recognize_image_index"] = recognize_image_index
                recognize_items.append(recognize_item)

    if has_success:
        save_canvas(canvas, visualize_path)

    service.logger.info(
        "meter recognition finished: image=%s meters=%s output=%s",
        local_image_path,
        len(recognize_items),
        visualize_path,
    )
    return recognize_items


def run_digital_meter_recognition(service, local_image_path: Path, data_types: list[dict[str, str]], visualize_path: Path, extra_info: dict[str, str]) -> list[dict[str, str]]:
    """
    数码表链路的占位实现。

    后续会在这里接入 ROI 检测、ROI 预处理、OCR 推理以及数值清洗和校
    验逻辑。
    """
    del extra_info
    image = cv2.imread(str(local_image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {local_image_path}")
    save_canvas(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), visualize_path)
    return [
        {
            **build_error_data_entry(data_type, "数码表OCR链路尚未接入，当前仅完成server骨架整理"),
            "recognize_image_index": "0",
        }
        for data_type in data_types
    ]
