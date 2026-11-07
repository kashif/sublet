"""Builds sublet's C allocator (src/sublet/_allocator.c) as a plain shared library loaded with ctypes.

It only needs a C compiler: the CUDA driver is opened at runtime, so no CUDA toolkit or headers are required.
"""

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildAllocator(build_ext):
    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            raise RuntimeError(
                "sublet could not compile its C allocator (src/sublet/_allocator.c). It needs a C compiler such as "
                "gcc or clang on PATH (e.g. `apt install gcc`); no CUDA toolkit is required. Original error: " + str(e)
            ) from e


setup(
    ext_modules=[Extension("sublet._allocator", ["src/sublet/_allocator.c"], libraries=["dl", "pthread"])],
    cmdclass={"build_ext": BuildAllocator},
)
