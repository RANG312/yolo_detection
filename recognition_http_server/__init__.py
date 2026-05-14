from recognition_http_server.config import parse_args
from recognition_http_server.constants import (
    DEFAULT_CALLBACK_PATH,
    DEFAULT_CALLBACK_PORT,
    DEFAULT_CONF,
    DEFAULT_DATA_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_FIRE_EXTINGUISHER_MODEL_PATH,
    DEFAULT_FIRE_MODEL_PATH,
    DEFAULT_FIRE_PROTECTION_FACILITIES_MODEL_PATH,
    DEFAULT_HOST,
    DEFAULT_IMGSZ,
    DEFAULT_LOG_DIR_NAME,
    DEFAULT_MAX_VALUE,
    DEFAULT_METER_MODEL_PATH,
    DEFAULT_MIN_VALUE,
    DEFAULT_PERSON_AND_CARS_MODEL_PATH,
    DEFAULT_PERSON_FALL_DOWN_MODEL_PATH,
    DEFAULT_PORT,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RESULT_ROOT,
    DEFAULT_SAFEHAT_MODEL_PATH,
    OCR_SUBTYPES,
    RECOGNIZE_TYPE_ALIASES,
    RECOGNIZE_TYPE_FIRE,
    RECOGNIZE_TYPE_FIRE_EXTINGUISHER,
    RECOGNIZE_TYPE_FIRE_PROTECTION_FACILITIES,
    RECOGNIZE_TYPE_METER,
    RECOGNIZE_TYPE_PERSON_AND_CARS,
    RECOGNIZE_TYPE_PERSON_FALL_DOWN,
    RECOGNIZE_TYPE_SAFEHAT,
    TASK_KIND_FIRE,
    TASK_KIND_FIRE_EXTINGUISHER,
    TASK_KIND_FIRE_PROTECTION_FACILITIES,
    TASK_KIND_METER,
    TASK_KIND_PERSON_AND_CARS,
    TASK_KIND_PERSON_FALL_DOWN,
    TASK_KIND_SAFEHAT,
)
from recognition_http_server.helpers import (
    align_data_types_to_images,
    build_callback_url,
    build_error_data_entry,
    build_meter_predict_args,
    build_success_data_entry,
    is_http_url,
    make_error_payload,
    make_filename_from_url,
    normalize_data_types,
    now_text,
    parse_extra_info,
    resolve_meter_subtype,
    resolve_task_kind,
    split_image_paths,
)
from recognition_http_server.schemas import MeterSubtypeResolution, TaskHandler, TaskState


def __getattr__(name: str):  # noqa: ANN202
    if name == "main":
        from recognition_http_server.app import main

        return main
    if name == "RecognitionAPIServer":
        from recognition_http_server.http_api import RecognitionAPIServer

        return RecognitionAPIServer
    if name == "RecognitionHandler":
        from recognition_http_server.http_api import RecognitionHandler

        return RecognitionHandler
    if name == "RecognitionService":
        from recognition_http_server.service import RecognitionService

        return RecognitionService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DEFAULT_CALLBACK_PATH",
    "DEFAULT_CALLBACK_PORT",
    "DEFAULT_CONF",
    "DEFAULT_DATA_TYPE",
    "DEFAULT_DEVICE",
    "DEFAULT_FIRE_EXTINGUISHER_MODEL_PATH",
    "DEFAULT_FIRE_MODEL_PATH",
    "DEFAULT_FIRE_PROTECTION_FACILITIES_MODEL_PATH",
    "DEFAULT_HOST",
    "DEFAULT_IMGSZ",
    "DEFAULT_LOG_DIR_NAME",
    "DEFAULT_MAX_VALUE",
    "DEFAULT_METER_MODEL_PATH",
    "DEFAULT_MIN_VALUE",
    "DEFAULT_PERSON_AND_CARS_MODEL_PATH",
    "DEFAULT_PERSON_FALL_DOWN_MODEL_PATH",
    "DEFAULT_PORT",
    "DEFAULT_REQUEST_TIMEOUT",
    "DEFAULT_RESULT_ROOT",
    "DEFAULT_SAFEHAT_MODEL_PATH",
    "OCR_SUBTYPES",
    "RECOGNIZE_TYPE_ALIASES",
    "RECOGNIZE_TYPE_FIRE",
    "RECOGNIZE_TYPE_FIRE_EXTINGUISHER",
    "RECOGNIZE_TYPE_FIRE_PROTECTION_FACILITIES",
    "RECOGNIZE_TYPE_METER",
    "RECOGNIZE_TYPE_PERSON_AND_CARS",
    "RECOGNIZE_TYPE_PERSON_FALL_DOWN",
    "RECOGNIZE_TYPE_SAFEHAT",
    "TASK_KIND_FIRE",
    "TASK_KIND_FIRE_EXTINGUISHER",
    "TASK_KIND_FIRE_PROTECTION_FACILITIES",
    "TASK_KIND_METER",
    "TASK_KIND_PERSON_AND_CARS",
    "TASK_KIND_PERSON_FALL_DOWN",
    "TASK_KIND_SAFEHAT",
    "MeterSubtypeResolution",
    "RecognitionAPIServer",
    "RecognitionHandler",
    "RecognitionService",
    "TaskHandler",
    "TaskState",
    "align_data_types_to_images",
    "build_callback_url",
    "build_error_data_entry",
    "build_meter_predict_args",
    "build_success_data_entry",
    "is_http_url",
    "main",
    "make_error_payload",
    "make_filename_from_url",
    "normalize_data_types",
    "now_text",
    "parse_extra_info",
    "parse_args",
    "resolve_meter_subtype",
    "resolve_task_kind",
    "split_image_paths",
]
