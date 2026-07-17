from __future__ import annotations

LEGACY_EXPECTED_CLASSES = {"start", "point", "end"}
METER_DATA_9K_EXPECTED_CLASSES = {"gauge", "center", "pointer_tip", "max_tick", "min_tick"}

# The retry pass crops one detected gauge and normalizes it to the model training size.
METER_DATA_9K_GAUGE_RETRY_SIZE = 640
