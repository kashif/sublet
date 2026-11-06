"""Lend a GPU memory region to arbitrary PyTorch code and learn exactly which blocks it used.

```python
region = LendableRegion(50 << 30)            # region.tensor: the owner's memory, fixed address
with region.lend() as loan:                  # CUDA allocations of every thread and stream may land in the region
    train_step()
loan.touched_blocks                          # every other block is bit-identical, no copy made
```
"""

from .region import PAGE, LeakError, LendableRegion, Loan

__all__ = ["PAGE", "LeakError", "LendableRegion", "Loan"]
__version__ = "0.1.0"
