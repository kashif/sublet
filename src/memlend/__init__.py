"""Lend the memory of a GPU tensor to any PyTorch code, and get back the parts you keep."""

from .lend import LeakError, Loan, empty, lend

__all__ = ["LeakError", "Loan", "empty", "lend"]
__version__ = "0.1.0"
