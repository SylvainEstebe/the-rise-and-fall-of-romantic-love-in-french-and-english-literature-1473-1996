"""Compatibility with the compiler used by the pinned PyTensor release."""

import sys
from functools import wraps


def configure():
    if sys.platform != "darwin":
        return
    from pytensor.link.c.cmodule import GCC_compiler

    original = GCC_compiler.compile_args
    if getattr(original, "_study_compat", False):
        return

    @wraps(original)
    def compile_args(*args, **kwargs):
        # PyTensor #2268: Apple clang treats -ld64 as a nonexistent library.
        # This reproduces the compiler fix in the original study environment.
        return [flag for flag in original(*args, **kwargs) if flag != "-ld64"]

    compile_args._study_compat = True
    GCC_compiler.compile_args = staticmethod(compile_args)


configure()
