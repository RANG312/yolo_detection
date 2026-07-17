from __future__ import annotations

from pathlib import Path

import cv2
import requests

from recognition_http_server.helpers import is_http_url, make_filename_from_url

HTTP_IMAGE_SCALE = 1.5  # 图片放大倍率


def _scaled_image_path(task_input_dir: Path, index: int, source_path: Path) -> Path:
    suffix = source_path.suffix or ".jpg"
    return task_input_dir / f"image_{index}_{source_path.stem}_scaled_{HTTP_IMAGE_SCALE}{suffix}"


def _resize_image(logger, source_path: Path, target_path: Path, req_id: str, index: int) -> Path:
    image = cv2.imread(str(source_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image {source_path}")

    height, width = image.shape[:2]
    resized = cv2.resize(image, None, fx=HTTP_IMAGE_SCALE, fy=HTTP_IMAGE_SCALE, interpolation=cv2.INTER_LINEAR)
    if not cv2.imwrite(str(target_path), resized):
        raise RuntimeError(f"Cannot write resized image: {target_path}")
    resized_height, resized_width = resized.shape[:2]
    logger.info(
        "Image resized for http inference: req_id=%s index=%s source=%s target=%s scale=%s size=%sx%s resized_size=%sx%s",
        req_id,
        index,
        source_path,
        target_path,
        HTTP_IMAGE_SCALE,
        width,
        height,
        resized_height,
        resized_width,
    )
    return target_path


def prepare_image(logger, input_root: Path, request_timeout: int, req_id: str, index: int, image_path: str) -> Path:
    """将输入图片统一准备为本地文件路径。.

    同时兼容远程 HTTP/HTTPS 图片和已有本地文件，保证下游识别链路始终 只处理本地路径。
    """
    task_input_dir = input_root / req_id
    task_input_dir.mkdir(parents=True, exist_ok=True)

    if is_http_url(image_path):
        filename = make_filename_from_url(image_path, f"image_{index}.jpg")
        target_path = task_input_dir / filename
        logger.info("downloading image: req_id=%s index=%s url=%s target=%s", req_id, index, image_path, target_path)
        response = requests.get(image_path, timeout=request_timeout)
        response.raise_for_status()
        target_path.write_bytes(response.content)
        logger.info(
            "image downloaded: req_id=%s index=%s target=%s bytes=%s", req_id, index, target_path, len(response.content)
        )
        resized_path = _scaled_image_path(task_input_dir, index, target_path)
        return _resize_image(logger, target_path, resized_path, req_id, index)

    local_path = Path(image_path).expanduser()
    if not local_path.is_absolute():
        local_path = Path.cwd() / local_path
    if not local_path.exists():
        raise FileNotFoundError(f"Image does not exist: {local_path}")
    logger.info("using local image: req_id=%s index=%s path=%s", req_id, index, local_path)
    resized_path = _scaled_image_path(task_input_dir, index, local_path)
    return _resize_image(logger, local_path, resized_path, req_id, index)
