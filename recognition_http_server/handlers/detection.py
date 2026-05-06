from __future__ import annotations

from pathlib import Path

import cv2

from recognition_http_server.dial_reading import save_canvas


def run_detection_recognition(service, local_image_path: Path, model, data_types: list[dict[str, str]], visualize_path: Path, task_desc: str) -> list[dict[str, str]]:
    """
    执行通用目标检测链路，供纯检测型任务复用。

    该路径适用于火源、安全帽以及后续不需要几何推理或 OCR 后处理的
    检测任务。
    """
    image = cv2.imread(str(local_image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {local_image_path}")

    service.logger.info("detection recognition running: task=%s image=%s output=%s", task_desc, local_image_path, visualize_path)
    with service._predict_lock:
        results = model.predict(
            source=image,
            imgsz=service.config.imgsz,
            conf=service.config.conf,
            device=service.config.device,
            verbose=False,
        )
    result = results[0]

    recognize_items: list[dict[str, str]] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        service.logger.warning("detection recognition found no targets: task=%s image=%s", task_desc, local_image_path)
        raise ValueError(f"{task_desc}未检测到目标")

    plotted = result.plot()
    save_canvas(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB), visualize_path)

    names = result.names if isinstance(result.names, dict) else {i: name for i, name in enumerate(result.names)}
    for detect_index, box in enumerate(boxes, start=1):
        cls_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
        label = str(names.get(cls_id, cls_id))
        for data_type in data_types:
            recognize_items.append(
                {
                    "recognize_image_index": str(detect_index),
                    "recognize_type": data_type["recognize_type"],
                    "recognize_subtype": data_type["recognize_subtype"],
                    "recognize_value": label,
                    "confidence": str(int(round(confidence * 100))),
                    "recognize_desc": f"识别成功，{task_desc}检测到{label}，置信度={confidence:.4f}",
                }
            )

    service.logger.info(
        "detection recognition finished: task=%s image=%s detections=%s output=%s",
        task_desc,
        local_image_path,
        len(boxes),
        visualize_path,
    )
    return recognize_items
