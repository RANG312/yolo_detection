#!/usr/bin/env python3
"""Capture timed snapshots from a gimbal video stream and record RGB statistics."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Union
from urllib.parse import urlsplit, urlunsplit

ManifestRow = dict[str, Union[int, float, str]]
ROI = tuple[int, int, int, int]


def parse_source(value: str) -> int | str:
    """Return an OpenCV device index for integer sources, otherwise the original source string."""
    return int(value) if value.isdigit() else value


def mask_source(value: int | str) -> str:
    """Return a display-safe source string with URL passwords hidden."""
    if isinstance(value, int):
        return str(value)
    parts = urlsplit(value)
    if not parts.password:
        return value
    hostname = parts.hostname or ""
    username = parts.username or ""
    port = f":{parts.port}" if parts.port is not None else ""
    netloc = f"{username}:***@{hostname}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def parse_roi(value: str | None) -> ROI | None:
    """Parse an ROI string in x,y,w,h format."""
    if not value:
        return None
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must use x,y,w,h format")
    try:
        x, y, width, height = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--roi values must be integers") from exc
    if width <= 0 or height <= 0 or x < 0 or y < 0:
        raise argparse.ArgumentTypeError("--roi requires non-negative x/y and positive width/height")
    return x, y, width, height


def _image_size(image) -> tuple[int, int]:
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    return len(image[0]), len(image)


def _iter_bgr_pixels(image, roi: ROI | None = None) -> Iterable[tuple[int, int, int]]:
    image_width, image_height = _image_size(image)
    if roi is None:
        x0, y0, width, height = 0, 0, image_width, image_height
    else:
        x0, y0, width, height = roi
        if x0 + width > image_width or y0 + height > image_height:
            raise ValueError(
                f"ROI x={x0} y={y0} width={width} height={height} exceeds image size {image_width}x{image_height}"
            )

    for y in range(y0, y0 + height):
        for x in range(x0, x0 + width):
            b, g, r = image[y][x]
            yield int(b), int(g), int(r)


def compute_frame_statistics(image, roi: ROI | None = None) -> dict[str, float]:
    """Compute RGB mean/std and simple magenta-bias metrics for one BGR image."""
    count = 0
    sums = {"r": 0.0, "g": 0.0, "b": 0.0}
    sums2 = {"r": 0.0, "g": 0.0, "b": 0.0}
    magenta_count = 0
    green_deficit_sum = 0.0
    chroma_sum = 0.0

    for b, g, r in _iter_bgr_pixels(image, roi):
        count += 1
        sums["r"] += r
        sums["g"] += g
        sums["b"] += b
        sums2["r"] += r * r
        sums2["g"] += g * g
        sums2["b"] += b * b
        green_deficit = ((r + b) / 2.0) - g
        if green_deficit > 0:
            green_deficit_sum += green_deficit
        if green_deficit > 30.0 and r > 80 and b > 80:
            magenta_count += 1
        chroma_sum += max(r, g, b) - min(r, g, b)

    if count == 0:
        raise ValueError("Cannot compute statistics for an empty image or ROI.")

    means = {channel: sums[channel] / count for channel in ("r", "g", "b")}
    stds = {
        channel: math.sqrt(max(0.0, sums2[channel] / count - means[channel] * means[channel]))
        for channel in ("r", "g", "b")
    }
    rgb_total = means["r"] + means["g"] + means["b"]

    return {
        "mean_r": means["r"],
        "mean_g": means["g"],
        "mean_b": means["b"],
        "std_r": stds["r"],
        "std_g": stds["g"],
        "std_b": stds["b"],
        "r_minus_g": means["r"] - means["g"],
        "b_minus_g": means["b"] - means["g"],
        "g_ratio": means["g"] / rgb_total if rgb_total else 0.0,
        "green_deficit": green_deficit_sum / count,
        "magenta_pct": magenta_count * 100.0 / count,
        "chroma": chroma_sum / count,
    }


def create_run_dir(base_dir: Path, now: datetime | None = None) -> Path:
    """Create and return a timestamped capture output directory."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def write_manifest_row(writer: csv.DictWriter, row: ManifestRow) -> None:
    """Write one manifest row after normalizing float precision."""
    normalized: dict[str, int | str] = {}
    for key, value in row.items():
        normalized[key] = f"{value:.6f}" if isinstance(value, float) else value
    writer.writerow(normalized)


def capture_frames(args: argparse.Namespace) -> Path:
    """Capture frames according to parsed CLI arguments."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for capture. Install opencv-python or use the Jetson runtime env."
        ) from exc

    source = parse_source(args.source)
    safe_source = mask_source(source)
    output_dir = create_run_dir(args.output_dir)
    manifest_path = output_dir / "manifest.csv"
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open capture source: {safe_source}")

    fieldnames = [
        "index",
        "captured_at",
        "elapsed_seconds",
        "image_path",
        "width",
        "height",
        "roi",
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
        "r_minus_g",
        "b_minus_g",
        "g_ratio",
        "green_deficit",
        "magenta_pct",
        "chroma",
    ]
    print(f"capture_source={safe_source}")
    print(f"output_dir={output_dir}")
    print(f"manifest={manifest_path}")

    try:
        with manifest_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            failures = 0
            captured = 0
            started_at = time.monotonic()
            next_capture_at = started_at + max(0.0, args.warmup)

            while args.count <= 0 or captured < args.count:
                now = time.monotonic()
                if now < next_capture_at:
                    time.sleep(min(0.05, next_capture_at - now))
                    continue

                ok, frame = capture.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= args.max_read_failures:
                        raise RuntimeError(f"Failed to read {failures} consecutive frames from {safe_source}")
                    time.sleep(0.1)
                    continue
                failures = 0

                stats = compute_frame_statistics(frame, args.roi)
                image_height, image_width = int(frame.shape[0]), int(frame.shape[1])
                timestamp = datetime.now()
                image_name = f"frame_{captured:06d}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                image_path = output_dir / image_name
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
                if not cv2.imwrite(str(image_path), frame, encode_params):
                    raise RuntimeError(f"Failed to write image: {image_path}")

                row: ManifestRow = {
                    "index": captured,
                    "captured_at": timestamp.isoformat(timespec="milliseconds"),
                    "elapsed_seconds": time.monotonic() - started_at,
                    "image_path": str(image_path),
                    "width": image_width,
                    "height": image_height,
                    "roi": "" if args.roi is None else ",".join(str(value) for value in args.roi),
                    **stats,
                }
                write_manifest_row(writer, row)
                file.flush()
                print(
                    f"captured index={captured} path={image_path} "
                    f"rgb=({stats['mean_r']:.1f},{stats['mean_g']:.1f},{stats['mean_b']:.1f}) "
                    f"green_deficit={stats['green_deficit']:.2f} magenta_pct={stats['magenta_pct']:.2f}"
                )
                captured += 1
                next_capture_at += max(0.0, args.interval)
    except KeyboardInterrupt:
        print("capture interrupted by user")
    finally:
        capture.release()

    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="RTSP/HTTP/file video source, or numeric device index such as 0."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gimbal_capture"),
        help="Base directory where a timestamped capture folder will be created.",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between saved frames.")
    parser.add_argument("--count", type=int, default=60, help="Number of frames to save. Use 0 for unlimited capture.")
    parser.add_argument(
        "--warmup", type=float, default=0.0, help="Seconds to wait after opening the source before saving."
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, choices=range(1, 101), metavar="[1-100]")
    parser.add_argument("--roi", type=parse_roi, default=None, help="Optional statistics ROI in x,y,w,h format.")
    parser.add_argument(
        "--max-read-failures",
        type=int,
        default=30,
        help="Abort after this many consecutive frame read failures.",
    )
    args = parser.parse_args(argv)
    if args.interval < 0:
        parser.error("--interval must be non-negative")
    if args.count < 0:
        parser.error("--count must be non-negative")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.max_read_failures <= 0:
        parser.error("--max-read-failures must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        capture_frames(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
