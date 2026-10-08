"""Print the real call depth at the moment a violation is handled.

The point is to calibrate the ``skip`` argument of ``_callsite`` against the
actual frame stack, rather than trusting a number carried over from an earlier
version of the wrapper.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import exlex.residency as residency  # noqa: E402


class T:
    is_cuda = True

    def item(self):
        return 1.0


def main() -> None:
    module = types.ModuleType("m")
    module.Tensor = T
    guard = residency.ResidencyGuard(torch_module=module)
    guard.install()

    original_callsite = residency._callsite

    def dump(skip=0):
        frame = sys._getframe(1)
        index = 0
        while frame is not None:
            name = frame.f_code.co_filename.rsplit("/", 1)[-1]
            print(f"  [{index}] {name}:{frame.f_lineno} in {frame.f_code.co_name}()")
            frame = frame.f_back
            index += 1
        return original_callsite(skip)

    residency._callsite = dump
    try:
        with guard.scope():
            T().item()
    except Exception as exc:
        print(f"  -> {type(exc).__name__}: {exc}")
    finally:
        residency._callsite = original_callsite
        guard.uninstall()


if __name__ == "__main__":
    main()
