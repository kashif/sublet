"""Lend a GPU memory region to arbitrary PyTorch code and learn exactly which blocks it used."""

from .region import PAGE, LeakError, LendableRegion, Loan

__all__ = ["PAGE", "LeakError", "LendableRegion", "Loan"]
__version__ = "0.1.0"
