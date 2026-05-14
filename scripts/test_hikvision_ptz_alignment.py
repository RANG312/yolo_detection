#!/usr/bin/env python3
"""Run a Hikvision PTZ alignment validation against a live meter scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from recognition_http_server.ptz_alignment import auto_tune_ptz_settings


class PrintLogger:
    def info(self, message: str, *args) -> None:  # noqa: ANN002
        print(message % args if args else message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Hikvision device IP or host.")
    parser.add_argument("--username", default="admin", help="Hikvision login username.")
    parser.add_argument("--password", required=True, help="Hikvision login password.")
    parser.add_argument("--channel", type=int, default=1, help="Hikvision channel number used by SDK and snapshot URL.")
    parser.add_argument("--model", default="runs/weights/1_dial_reading/best.pt", help="Meter YOLO weight path.")
    parser.add_argument("--device", default="cpu", help="Ultralytics inference device, e.g. cpu or 0.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO first-stage inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence threshold.")
    # FOV maps pixel offset to approximate PTZ angle: delta = offset_px / image_size * fov_deg.
    parser.add_argument("--horizontal-fov-deg", type=float, default=2.90, help="Fallback horizontal optical field of view.")
    parser.add_argument("--vertical-fov-deg", type=float, default=1.63, help="Fallback vertical optical field of view.")
    parser.set_defaults(auto_fov=True, auto_tune=True)
    fov_group = parser.add_mutually_exclusive_group()
    fov_group.add_argument("--auto-fov", dest="auto_fov", action="store_true", help="Read current FOV from HCNetSDK GIS info.")
    fov_group.add_argument("--no-auto-fov", dest="auto_fov", action="store_false", help="Use manual FOV arguments.")
    tune_group = parser.add_mutually_exclusive_group()
    tune_group.add_argument("--auto-tune", dest="auto_tune", action="store_true", help="Tune PTZ thresholds and nudge settings from FOV.")
    tune_group.add_argument("--no-auto-tune", dest="auto_tune", action="store_false", help="Use explicit PTZ tuning arguments.")
    parser.add_argument(
        "--threshold-deg",
        type=float,
        default=None,
        help="Skip PTZ correction when both pan and tilt deltas are below this angle.",
    )
    parser.add_argument(
        "--max-delta-deg",
        type=float,
        default=None,
        help="Clamp each correction pass to this maximum absolute pan/tilt angle.",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=3,
        help="Maximum number of detect-move-capture correction passes before final ROI reading.",
    )
    # Nudge parameters translate requested angle to continuous PTZControl duration.
    parser.add_argument("--nudge-speed", type=int, default=1, help="Hikvision PTZ speed level passed to SDK, usually 1-7.")
    parser.add_argument(
        "--nudge-degrees-per-second",
        type=float,
        default=2.5,
        help="Estimated movement speed used to convert requested degrees to nudge duration.",
    )
    parser.add_argument(
        "--nudge-max-seconds",
        type=float,
        default=1.0,
        help="Maximum duration of a single PTZ start/stop nudge command.",
    )
    parser.add_argument(
        "--nudge-max-steps",
        type=int,
        default=None,
        help="Maximum number of nudge commands used for one pan or tilt delta.",
    )
    parser.add_argument(
        "--tilt-nudge-scale",
        type=float,
        default=None,
        help="Multiplier applied to tilt nudge durations; useful when vertical movement is less sensitive.",
    )
    parser.add_argument(
        "--zoom-target-height-ratio",
        type=float,
        default=0.8,
        help="Target gauge ROI height as a fraction of image height after pan/tilt alignment.",
    )
    parser.add_argument(
        "--zoom-ratio-tolerance",
        type=float,
        default=0.05,
        help="Do not adjust zoom when gauge ROI height is within this ratio of the target.",
    )
    parser.add_argument(
        "--zoom-max-passes",
        type=int,
        default=3,
        help="Maximum detect-zoom-capture passes after pan/tilt alignment.",
    )
    parser.add_argument("--zoom-nudge-speed", type=int, default=1, help="Hikvision zoom speed level passed to SDK.")
    parser.add_argument(
        "--zoom-nudge-seconds",
        type=float,
        default=0.6,
        help="Duration of each zoom-in or zoom-out command.",
    )
    parser.add_argument(
        "--zoom-nudge-steps",
        type=int,
        default=1,
        help="Number of zoom nudge commands per zoom pass.",
    )
    parser.add_argument(
        "--zoom-focus-timeout",
        type=float,
        default=2.0,
        help="Sleep after optical zoom so autofocus can settle before the next snapshot.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.5,
        help="Sleep after each PTZ correction so the next snapshot sees a stable image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hik_ptz_alignment"),
        help="Directory for before/result/after images.",
    )
    return parser.parse_args()


def describe_offset(image, model, predict_args, alignment_config: PTZAlignmentConfig, label: str) -> None:  # noqa: ANN001
    from recognition_http_server.dial_reading.detections import collect_all_detections
    from recognition_http_server.ptz_alignment import compute_alignment_request

    results = model.predict(
        source=image,
        imgsz=predict_args.imgsz,
        conf=predict_args.conf,
        device=predict_args.device,
        verbose=False,
    )
    detections = collect_all_detections(results[0])
    gauges = detections.get("gauge", [])
    print(f"{label}: shape={image.shape} gauges={len(gauges)}")
    if not gauges:
        return
    gauge = max(gauges, key=lambda item: float(item["conf"]))
    request = compute_alignment_request(
        image.shape,
        gauge["box"],
        alignment_config.horizontal_fov_deg,
        alignment_config.vertical_fov_deg,
    )
    print(
        (
            f"{label}: selected_conf={float(gauge['conf']):.4f} "
            f"dx={request.dx_px:.1f}px dy={request.dy_px:.1f}px "
            f"pan_delta={request.pan_delta_deg:.3f} tilt_delta={request.tilt_delta_deg:.3f} "
            f"box={tuple(round(float(value), 1) for value in request.gauge_box)}"
        )
    )


def main() -> int:
    import cv2

    from recognition_http_server.dial_reading import load_model, predict_image_instances
    from recognition_http_server.hikvision_ptz import HikvisionPTZConfig, HikvisionPTZController
    from recognition_http_server.ptz_alignment import PTZAlignmentConfig

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger = PrintLogger()
    base_config = HikvisionPTZConfig(
        host=args.host,
        username=args.username,
        password=args.password,
        channel=args.channel,
        settle_seconds=args.settle_seconds,
        nudge_speed=args.nudge_speed,
    )
    fov_source = "manual"
    horizontal_fov_deg = args.horizontal_fov_deg
    vertical_fov_deg = args.vertical_fov_deg
    if args.auto_fov:
        try:
            fov = HikvisionPTZController(base_config, logger=logger).read_field_of_view()
        except Exception as exc:  # noqa: BLE001
            print(f"auto_fov_failed: {exc}; using fallback hfov={horizontal_fov_deg:.3f} vfov={vertical_fov_deg:.3f}")
        else:
            horizontal_fov_deg = fov.horizontal_deg
            vertical_fov_deg = fov.vertical_deg
            fov_source = "sdk"
            print(
                f"auto_fov_ok: hfov={horizontal_fov_deg:.3f} vfov={vertical_fov_deg:.3f} "
                f"hfov_range={fov.min_horizontal_deg:.3f}-{fov.max_horizontal_deg:.3f} "
                f"vfov_range={fov.min_vertical_deg:.3f}-{fov.max_vertical_deg:.3f} zoom={fov.zoom:.1f}"
            )

    tuned = auto_tune_ptz_settings(horizontal_fov_deg, vertical_fov_deg)
    threshold_deg = args.threshold_deg if args.threshold_deg is not None else (tuned.threshold_deg if args.auto_tune else 0.1)
    max_delta_deg = args.max_delta_deg if args.max_delta_deg is not None else (tuned.max_delta_deg if args.auto_tune else 10.0)
    nudge_degrees_per_second = (
        args.nudge_degrees_per_second
        if args.nudge_degrees_per_second is not None
        else (tuned.nudge_degrees_per_second if args.auto_tune else 8.0)
    )
    nudge_max_steps = args.nudge_max_steps if args.nudge_max_steps is not None else (tuned.nudge_max_steps if args.auto_tune else 2)
    tilt_nudge_scale = args.tilt_nudge_scale if args.tilt_nudge_scale is not None else (tuned.tilt_nudge_scale if args.auto_tune else 1.0)
    print(
        f"ptz_config: fov_source={fov_source} hfov={horizontal_fov_deg:.3f} vfov={vertical_fov_deg:.3f} "
        f"threshold={threshold_deg:.3f} max_delta={max_delta_deg:.3f} "
        f"nudge_speed={args.nudge_speed} nudge_dps={nudge_degrees_per_second:.3f} "
        f"nudge_max_seconds={args.nudge_max_seconds:.3f} nudge_max_steps={nudge_max_steps} "
        f"tilt_nudge_scale={tilt_nudge_scale:.3f} "
        f"zoom_target_h={args.zoom_target_height_ratio:.3f} zoom_tol={args.zoom_ratio_tolerance:.3f} "
        f"zoom_passes={args.zoom_max_passes} zoom_nudge={args.zoom_nudge_seconds:.3f}s*{args.zoom_nudge_steps} "
        f"zoom_focus_timeout={args.zoom_focus_timeout:.3f}"
    )
    model, annotation_mode = load_model(args.model)
    alignment_config = PTZAlignmentConfig(
        enabled=True,
        horizontal_fov_deg=horizontal_fov_deg,
        vertical_fov_deg=vertical_fov_deg,
        threshold_deg=threshold_deg,
        max_delta_deg=max_delta_deg,
        max_passes=args.max_passes,
        zoom_enabled=args.zoom_max_passes > 0,
        zoom_target_height_ratio=args.zoom_target_height_ratio,
        zoom_ratio_tolerance=args.zoom_ratio_tolerance,
        zoom_max_passes=args.zoom_max_passes,
    )
    controller = HikvisionPTZController(
        HikvisionPTZConfig(
            host=args.host,
            username=args.username,
            password=args.password,
            channel=args.channel,
            settle_seconds=args.settle_seconds,
            nudge_speed=args.nudge_speed,
            nudge_degrees_per_second=nudge_degrees_per_second,
            nudge_max_seconds=args.nudge_max_seconds,
            nudge_max_steps=nudge_max_steps,
            tilt_nudge_scale=tilt_nudge_scale,
            zoom_nudge_speed=args.zoom_nudge_speed,
            zoom_nudge_seconds=args.zoom_nudge_seconds,
            zoom_nudge_steps=args.zoom_nudge_steps,
            zoom_focus_timeout=args.zoom_focus_timeout,
        ),
        logger=logger,
    )
    predict_args = SimpleNamespace(
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        min_value=0.0,
        max_value=1.0,
        debug_center=False,
        ptz_alignment_config=alignment_config,
        ptz_controller=controller,
        ptz_logger=logger,
    )

    before_image = controller.capture_image()
    before_path = args.output_dir / "before.jpg"
    cv2.imwrite(str(before_path), before_image)
    print(f"before_snapshot={before_path}")
    describe_offset(before_image, model, predict_args, alignment_config, "before")

    canvas, instances, inference_time, compute_time = predict_image_instances(
        before_path,
        model,
        predict_args,
        annotation_mode,
    )
    result_path = args.output_dir / "result.jpg"
    cv2.imwrite(str(result_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"result_snapshot={result_path}")
    print(f"annotation_mode={annotation_mode} inference_time={inference_time:.3f}s compute_time={compute_time:.3f}s")
    print(
        "instances="
        + str(
            [
                (item.get("recognize_image_index"), item.get("error"), item.get("reading"))
                for item in instances
            ]
        )
    )

    after_image = controller.capture_image()
    after_path = args.output_dir / "after.jpg"
    cv2.imwrite(str(after_path), after_image)
    print(f"after_snapshot={after_path}")
    describe_offset(after_image, model, predict_args, alignment_config, "after")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
