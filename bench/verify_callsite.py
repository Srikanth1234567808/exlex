"""Verify the reported callsite is the user's code, not exlex internals.

A violation that points at residency.py instead of the offending line is
useless for the person who has to fix it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exlex.errors import ResidencyViolation  # noqa: E402
from exlex.residency import ResidencyGuard  # noqa: E402

MARKER = "CALLSITE_SENTINEL_LINE"


class T:
    is_cuda = True

    def item(self):
        return 1.0  # MARKER_SENTINEL


def main() -> int:
    import types

    module = types.ModuleType("m")
    module.Tensor = T
    guard = ResidencyGuard(torch_module=module)
    guard.install()
    reported = None
    try:
        with guard.scope():
            try:
                T().item()  # MARKER
            except ResidencyViolation as exc:
                reported = exc.where
    finally:
        guard.uninstall()

    # The MARKER line is this one.
    with open(__file__, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    expected = next(
        i + 1 for i, line in enumerate(lines) if "MARKER" in line and "T().item()" in line
    )

    text = lines[reported - 1].strip() if reported else "<none>"
    print(f"reported lineno : {reported}")
    print(f"that line is    : {text!r}")
    print(f"expected        : {expected}")
    ok = text.startswith("T().item()")
    print("RESULT:", "ok, points at the caller" if ok else "WRONG, points elsewhere")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
