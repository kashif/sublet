import pytest
import torch

MiB = 1 << 20
BLOCK = 64 * MiB


def pytest_collection_modifyitems(config, items):
    # Under a non-native allocator backend memlend refuses to run: only the test checking that refusal applies.
    if torch.cuda.get_allocator_backend() != "native":
        for item in items:
            if "backend" not in item.name:
                item.add_marker(pytest.mark.skip(reason="memlend needs the native allocator backend"))
