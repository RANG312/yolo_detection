from __future__ import annotations

import scripts.test_hikvision_ptz_alignment as alignment_script


def test_auto_tuning_uses_long_focus_nudge_defaults_for_narrow_fov() -> None:
    settings = alignment_script.auto_tune_ptz_settings(horizontal_fov_deg=2.9, vertical_fov_deg=1.63)

    assert settings.threshold_deg == 0.02
    assert settings.max_delta_deg == 1.45
    assert settings.nudge_degrees_per_second == 0.5
    assert settings.nudge_max_steps == 4
    assert settings.tilt_nudge_scale == 2.0


def test_auto_tuning_uses_faster_defaults_for_wide_fov() -> None:
    settings = alignment_script.auto_tune_ptz_settings(horizontal_fov_deg=54.93, vertical_fov_deg=32.6)

    assert settings.threshold_deg == 0.1
    assert settings.max_delta_deg == 5.0
    assert settings.nudge_degrees_per_second == 4.0
    assert settings.nudge_max_steps == 2
    assert settings.tilt_nudge_scale == 1.0
