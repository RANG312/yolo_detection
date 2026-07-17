from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEST = "runs/train/dial_thermometer_20260401_133934/weights/best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained YOLO .pt model to ONNX.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_BEST, help="Path to best.pt.")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    parser.add_argument("--device", type=str, default="cpu", help="Export device, usually 'cpu'.")
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic input shapes.")
    parser.add_argument("--simplify", action="store_true", help="Simplify the ONNX graph after export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        device=args.device,
    )
    print(f"Exported ONNX model: {exported_path}")


if __name__ == "__main__":
    main()
