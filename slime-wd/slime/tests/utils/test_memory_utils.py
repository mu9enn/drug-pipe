from __future__ import annotations

from unittest.mock import patch

from slime.utils.memory_utils import available_memory


def test_available_memory_survives_transient_procfs_failure():
    gib = 1024**3
    with (
        patch("slime.utils.memory_utils.torch.cuda.current_device", return_value=3),
        patch("slime.utils.memory_utils.torch.cuda.mem_get_info", return_value=(6 * gib, 8 * gib)),
        patch("slime.utils.memory_utils.torch.cuda.memory_allocated", return_value=1 * gib),
        patch("slime.utils.memory_utils.torch.cuda.memory_reserved", return_value=2 * gib),
        patch(
            "slime.utils.memory_utils.psutil.virtual_memory",
            side_effect=OSError(107, "Transport endpoint is not connected"),
        ),
    ):
        result = available_memory()

    assert result == {
        "gpu": "3",
        "total_GB": 8.0,
        "free_GB": 6.0,
        "used_GB": 2.0,
        "allocated_GB": 1.0,
        "reserved_GB": 2.0,
    }
