from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from log_manager import GlobalLogManager
from recognition_http_server.helpers import make_error_payload, now_text


class RecognitionHandler(BaseHTTPRequestHandler):
    """识别服务的 HTTP 协议层，负责健康检查、任务提交和任务查询。"""

    server: "RecognitionAPIServer"

    @property
    def logger(self) -> logging.Logger:
        return GlobalLogManager.get_logger("recognition.http")

    def do_GET(self) -> None:  # noqa: N802
        """处理健康检查和任务状态查询接口。"""
        if self.path == "/health":
            self.logger.debug("health check requested: client=%s", self.client_address[0])
            self._send_json(HTTPStatus.OK, {"code": 0, "resp_msg": "ok", "data": {"status": "healthy", "time": now_text()}})
            return

        if self.path.startswith("/api/v1/recognition/tasks/"):
            req_id = self.path.rsplit("/", 1)[-1]
            self.logger.info("task query requested: req_id=%s client=%s", req_id, self.client_address[0])
            task = self.server.service.get_task(req_id)
            if task is None:
                self._send_json(HTTPStatus.NOT_FOUND, make_error_payload(req_id, "Task not found."))
                return
            payload = {
                "req_id": task.req_id,
                "code": 0,
                "resp_msg": "success",
                "data": {
                    "task_status": task.status,
                    "created_at": task.created_at,
                    "updated_at": task.updated_at,
                    "callback_payload": task.callback_payload,
                    "callback_status_code": task.callback_status_code,
                    "callback_error": task.callback_error,
                    "error": task.error,
                },
            }
            self._send_json(HTTPStatus.OK, payload)
            return

        self._send_json(HTTPStatus.NOT_FOUND, make_error_payload(None, "Path not found."))

    def do_POST(self) -> None:  # noqa: N802
        """处理异步任务提交，并立即返回 accepted 响应。"""
        if self.path != "/api/v1/recognition/tasks":
            self._send_json(HTTPStatus.NOT_FOUND, make_error_payload(None, "Path not found."))
            return

        payload: dict[str, Any] | None = None
        try:
            payload = self._read_json_body()
            callback_host = self._get_callback_host()
            self.logger.info(
                "task submit requested: client=%s callback_host=%s req_id=%s",
                self.client_address[0],
                callback_host,
                payload.get("req_id"),
            )
            task = self.server.service.create_task(payload, callback_host)
        except ValueError as exc:
            req_id = None
            if isinstance(exc.args[0], str) and "req_id" in str(exc):
                req_id = str(payload.get("req_id")) if isinstance(payload, dict) else None
            self.logger.warning("bad request: path=%s req_id=%s error=%s", self.path, req_id, exc)
            self._send_json(HTTPStatus.BAD_REQUEST, make_error_payload(req_id, str(exc)))
            return
        except Exception as exc:
            req_id = payload.get("req_id") if isinstance(payload, dict) else None
            self.logger.exception("request handling failed: path=%s req_id=%s", self.path, req_id)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, make_error_payload(req_id, str(exc)))
            return

        response = {
            "req_id": task.req_id,
            "code": 0,
            "resp_msg": "Task accepted successfully.",
            "data": {"task_status": task.status},
        }
        self.logger.info("task accepted: req_id=%s", task.req_id)
        self._send_json(HTTPStatus.OK, response)

    def _get_callback_host(self) -> str:
        """从代理头或客户端连接信息中推断回调主机地址。"""
        forwarded_for = self.headers.get("X-Forwarded-For", "").strip()
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        real_ip = self.headers.get("X-Real-IP", "").strip()
        if real_ip:
            return real_ip
        return self.client_address[0]

    def log_message(self, format: str, *args: Any) -> None:
        """将默认 HTTP 访问日志重定向到项目日志系统。"""
        message = format % args
        self.logger.info("access: client=%s message=%s", self.address_string(), message)

    def _read_json_body(self) -> dict[str, Any]:
        """读取并校验当前 HTTP 请求中的 JSON 请求体。"""
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("Request body is empty.")

        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            raise ValueError("Content-Type must be application/json.")

        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Request body is not valid JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError("Request body JSON must be an object.")
        return payload

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        """按指定状态码发送 UTF-8 编码的 JSON 响应。"""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class RecognitionAPIServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], request_handler_class: type[RecognitionHandler], service) -> None:
        """创建 HTTP 服务实例，并注入识别业务服务对象。"""
        super().__init__(server_address, request_handler_class)
        self.service = service
