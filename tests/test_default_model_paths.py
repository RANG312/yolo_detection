from __future__ import annotations

import importlib.util
from pathlib import Path


def load_constants_module():
    constants_path = Path(__file__).resolve().parents[1] / "recognition_http_server" / "constants.py"
    spec = importlib.util.spec_from_file_location("recognition_http_server_constants", constants_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_task_default_model_paths_use_runs_weights() -> None:
    constants = load_constants_module()

    assert constants.DEFAULT_METER_MODEL_PATH == "runs/weights/1_dial_reading/best.pt"
    assert constants.DEFAULT_FIRE_MODEL_PATH == "runs/weights/6_fire_and_smoke/best_fire.pt"
    assert constants.DEFAULT_SAFEHAT_MODEL_PATH == "runs/weights/7_safe_hat/best_person.pt"
    assert constants.DEFAULT_FIRE_PROTECTION_FACILITIES_MODEL_PATH == (
        "runs/weights/8_fire_protection_facilities/fire-fighting-facilitie_best.pt"
    )
    assert constants.DEFAULT_PERSON_FALL_DOWN_MODEL_PATH == "runs/weights/9_person_fall_down/fall_best.pt"
    assert constants.DEFAULT_FIRE_EXTINGUISHER_MODEL_PATH == ("runs/weights/10_fire_extinguisher/extinguisher_best.pt")
    assert constants.DEFAULT_PERSON_AND_CARS_MODEL_PATH == "runs/weights/11_person_and_cars/car_best.pt"


def test_new_detection_task_type_aliases_match_weight_folder_numbers() -> None:
    constants = load_constants_module()

    assert constants.RECOGNIZE_TYPE_ALIASES["8"] == constants.TASK_KIND_FIRE_PROTECTION_FACILITIES
    assert constants.RECOGNIZE_TYPE_ALIASES["9"] == constants.TASK_KIND_PERSON_FALL_DOWN
    assert constants.RECOGNIZE_TYPE_ALIASES["10"] == constants.TASK_KIND_FIRE_EXTINGUISHER
    assert constants.RECOGNIZE_TYPE_ALIASES["11"] == constants.TASK_KIND_PERSON_AND_CARS
