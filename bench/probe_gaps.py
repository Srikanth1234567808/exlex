"""Probe for enforcement gaps: find leak paths the guard does not close.

Each check installs a real guard over a fake torch and asks whether a known
leak vector is actually refused. Anything reported as OPEN is a real hole, not
a hypothesis.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exlex.residency import ResidencyGuard  # noqa: E402


class T:
    """Stand-in tensor. Named methods stand in for the instrumented ones;
    ``__getattr__`` supplies everything else so uninstrumented names resolve to
    a callable, which is how the un-covered vectors get exercised.
    """
    is_cuda = True

    def cpu(self):
        return "<cpu ran>"

    def numpy(self):
        return "<numpy ran>"

    def item(self):
        return "<item ran>"

    def tolist(self):
        return "<tolist ran>"

    def tobytes(self):
        return b"x"

    def to(self, *a, **k):
        return "<to ran>"

    def pin_memory(self):
        return "<pin_memory ran>"

    def share_memory_(self):
        return "<share_memory_ ran>"

    def __repr__(self):
        return "<T>"

    # Real class attributes, so the guard's install-time scan can see them.
    def __dlpack__(self, *a, **k):
        return "<dlpack buffer>"

    def __dlpack_device__(self):
        return (1, 0)

    def untyped_storage(self):
        return "<storage>"

    def storage(self):
        return "<storage>"

    def __getattr__(self, name):
        return lambda *a, **k: f"<{name} ran>"


def check(label: str, closed: bool, detail: str = "") -> bool:
    status = "CLOSED" if closed else "OPEN"
    print(f"  [{status:6}] {label}" + (f"  ({detail})" if detail else ""))
    return closed


def try_call(guard, name: str, *args, **kwargs) -> bool:
    """Return True if the call was refused."""
    t = T()
    try:
        getattr(t, name)(*args, **kwargs)
        return False
    except Exception:
        return True


def main() -> None:
    module = types.ModuleType("m")
    module.Tensor = T
    written = []
    module.save = lambda obj, *a, **k: written.append(obj) or "saved"
    module.savez = lambda *a, **k: written.append("savez") or "saved"

    guard = ResidencyGuard(torch_module=module)
    guard.install()
    results = []
    try:
        with guard.scope():
            # Vectors the guard's TRANSFER_METHODS claims to cover.
            results.append(check("Tensor.cpu()", try_call(guard, "cpu")))
            results.append(check("Tensor.numpy()", try_call(guard, "numpy")))
            results.append(check("Tensor.item()", try_call(guard, "item")))
            results.append(check("Tensor.tolist()", try_call(guard, "tolist")))
            results.append(check("Tensor.tobytes()", try_call(guard, "tobytes")))
            results.append(check("Tensor.to('cpu')", try_call(guard, "to", "cpu")))
            results.append(check("Tensor.pin_memory()", try_call(guard, "pin_memory")))
            results.append(check("Tensor.share_memory_()", try_call(guard, "share_memory_")))

            # Vectors that plausibly leak but are NOT in TRANSFER_METHODS.
            results.append(
                check(
                    "Tensor.__dlpack__()  [zero-copy export to host]",
                    try_call(guard, "__dlpack__"),
                    "gives a host buffer sharing VRAM",
                )
            )
            results.append(
                check(
                    "Tensor.cpu().data_ptr()  [pointer disclosure]",
                    try_call(guard, "data_ptr"),
                    "leaks a device address",
                )
            )
            results.append(
                check(
                    "Tensor.untyped_storage()  [storage handle]",
                    try_call(guard, "untyped_storage"),
                    "bypasses the method list entirely",
                )
            )
            results.append(
                check(
                    "repr(tensor)  [value disclosure]",
                    try_call(guard, "__repr__"),
                )
            )

        # Module-level leaks. torch.save is instrumented, so test it live.
        saved = False
        try:
            with guard.scope():
                module.save(T())
            saved = True
        except Exception:
            saved = False
        results.append(
            check("torch.save(t)  [plaintext to disk]", not saved, "module function")
        )

        print()
        print("  Leaks the guard cannot see from Python, documented not fixed:")
        print("    [OPEN  ] torch.load(map_location='cpu')  lands weights in host RAM")
        print("    [OPEN  ] pickling for DataLoader workers  crosses to another process")
        print("    [OPEN  ] .cuda() -> DLPack export to a C consumer")
        print("    [OPEN  ] tensor.data_ptr()  device address disclosure")
    finally:
        guard.uninstall()

    open_count = results.count(False)
    print()
    print(f"  {len(results) - open_count}/{len(results)} tensor-level vectors closed")
    if open_count:
        print("  gaps found above must be fixed or documented")


if __name__ == "__main__":
    main()
