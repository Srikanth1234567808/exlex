"""Measure what exlex actually costs, before deciding anything about language.

Run: .venv/bin/python bench/guard_overhead.py
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exlex.residency import ResidencyGuard  # noqa: E402

ITERATIONS = 200_000


class BenchTensor:
    """Minimal tensor stand-in; the guard only touches is_cuda and the method."""

    def __init__(self, is_cuda: bool = True) -> None:
        self.is_cuda = is_cuda
        self.counter = 0

    def to(self, device: str) -> "BenchTensor":
        self.counter += 1
        return self

    def item(self) -> float:
        self.counter += 1
        return 1.0

    def cpu(self) -> "BenchTensor":
        self.counter += 1
        return self

    def __repr__(self) -> str:
        return "x"


def bench(label: str, fn, iterations: int = ITERATIONS, repeat: int = 5) -> float:
    """Return the best per-call microseconds, which is the least noisy stat."""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn(iterations)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed / iterations * 1e6)
    print(f"  {label:<44} {best:7.3f} us/call")
    return best


def main() -> None:
    module = types.ModuleType("bench_torch")
    module.Tensor = BenchTensor

    tensor = BenchTensor()

    print("Baseline: cost of the tensor method itself, no guard")
    bench("t.to('cuda')", lambda n: [tensor.to("cuda") for _ in range(n)])
    bench("t.item()", lambda n: [tensor.item() for _ in range(n)])

    guard = ResidencyGuard(torch_module=module)
    guard.install()
    try:
        print("\nGuarded, outside any secret region (region check only)")
        bench("t.to('cuda')", lambda n: [tensor.to("cuda") for _ in range(n)])
        bench("t.item()", lambda n: [tensor.item() for _ in range(n)])

        print("\nGuarded, inside a secret region (full decision path)")
        with guard.scope():
            bench("t.to('cuda')  [allowed]", lambda n: [tensor.to("cuda") for _ in range(n)])
            # item() is correctly *blocked* here, so measure a permitted case
            # instead: an allow() scope, which is the realistic hot path for
            # code that legitimately needs a host-side scalar.
            with guard.allow("benchmarked scalar read"):
                bench("t.item()  [allowed]", lambda n: [tensor.item() for _ in range(n)])

        print("\nRefused path (what a leak attempt costs)")
        blocked = 0
        start = time.perf_counter()
        for _ in range(50_000):
            try:
                with guard.scope():
                    tensor.item()
            except Exception:
                blocked += 1
        per_call = (time.perf_counter() - start) / 50_000 * 1e6
        print(f"  refused transfer {per_call:7.3f} us/call ({blocked} raised)")

        print("\nBaseline for a real GPU op, for scale")
        print("  a single 7B-param fwd/bwd on an H100 is on the order of 10-100 ms")
        print("  at 0.05 us/call, a million guarded .to() calls cost ~50 ms total")
    finally:
        guard.uninstall()


if __name__ == "__main__":
    main()
