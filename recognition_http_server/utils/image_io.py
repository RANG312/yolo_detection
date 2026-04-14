from __future__ import annotations

from pathlib import Path

import requests

from recognition_http_server.helpers import is_http_url, make_filename_from_url


def prepare_image(logger, input_root: Path, request_timeout: int, req_id: str, index: int, image_path: str) -> Path:
    """
    将输入图片统一准备为本地文件路径。

    同时兼容远程 HTTP/HTTPS 图片和已有本地文件，保证下游识别链路始终
    只处理本地路径。
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
        logger.info("image downloaded: req_id=%s index=%s target=%s bytes=%s", req_id, index, target_path, len(response.content))
        return target_path

    local_path = Path(image_path).expanduser()
    if not local_path.is_absolute():
        local_path = Path.cwd() / local_path
    if not local_path.exists():
        raise FileNotFoundError(f"Image does not exist: {local_path}")
    logger.info("using local image: req_id=%s index=%s path=%s", req_id, index, local_path)
    return local_path
