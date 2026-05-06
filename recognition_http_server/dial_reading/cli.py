from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from recognition_http_server.dial_reading.pipeline import load_model, predict_single_image
from recognition_http_server.dial_reading.visualization import save_canvas

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# CLI code stays separate from the HTTP service import path.
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
