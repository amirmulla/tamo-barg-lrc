"""Build script for tamo-barg-lrc with optional Cython GF(2^m) accelerator."""

from __future__ import annotations

import sys
import warnings

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Build Cython extensions when possible, falling back to pure NumPy on failure."""

    def run(self) -> None:
        try:
            super().run()
        except Exception as exc:
            warnings.warn(
                f"Optional Cython extension build skipped ({exc}); "
                "falling back to pure-NumPy implementation.",
                stacklevel=2,
            )

    def build_extension(self, ext: Extension) -> None:
        try:
            super().build_extension(ext)
        except Exception as exc:
            warnings.warn(
                f"Optional Cython extension '{ext.name}' failed to build ({exc}); "
                "falling back to pure-NumPy implementation.",
                stacklevel=2,
            )


def _get_extensions() -> list[Extension]:
    try:
        import numpy as np
        from Cython.Build import cythonize
    except ImportError:
        return []

    extra_compile_args = ["/O2"] if sys.platform == "win32" else ["-O3", "-std=c99"]

    ext = Extension(
        name="tamo_barg_lrc._ff._ff_cy",
        sources=["src/tamo_barg_lrc/_ff/_ff_cy.pyx"],
        include_dirs=[np.get_include(), "src/tamo_barg_lrc/_ff"],
        depends=["src/tamo_barg_lrc/_ff/_ff_core.h"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=extra_compile_args,
    )
    return cythonize(
        [ext],
        compiler_directives={"language_level": "3"},
        build_dir="build/cython",
    )


setup(
    ext_modules=_get_extensions(),
    cmdclass={"build_ext": OptionalBuildExt},
)
