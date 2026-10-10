"""Pepper's break-even question, measured on the affine check.

Pepper's honest metric was never "verification is cheap" in the abstract.
It was: after how many instances is the client cheaper than doing the work
itself? The setup (backend construction, challenge machinery) is paid once
-- the offline phase -- and each instance pays only the online check.

This bench asks the small version of that question that this prototype can
answer honestly: per-instance verification (one challenge per result) versus
batched verification (one shared challenge per round for the whole batch).
Both sides still recompute the expected results locally, so this measures
challenge-amortisation only, not less-than-execution verification. A real
SNARK behind exlex.backends.zk would change the shape of the answer; until
then the number below is the ceiling of what batching buys.

Run: .venv/bin/python bench/pepper_breakeven.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exlex.backends import PepperBackend, VerificationBackend  # noqa: E402
from exlex.circuit import AffineLayer, BiasLayer, Circuit  # noqa: E402


def bench(label, fn, repeat=5):
    """Best-of-N wall time, the least noisy statistic here."""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    print(f"    {label:<52} {best * 1e3:9.3f} ms")
    return best


def run(batch_size: int, width: int = 32, rounds: int = 3) -> None:
    circuit = Circuit(
        layers=[
            AffineLayer(
                weights=tuple(
                    tuple(float((i * 7 + j * 3) % 11) / 8.0 - 0.5 for j in range(width))
                    for i in range(width)
                ),
                name="fc",
            ),
            BiasLayer(bias=tuple(0.0 for _ in range(width)), name="fc.bias"),
        ]
    )
    batch = [
        [float((k * 5 + i) % 13) / 7.0 for i in range(width)] for k in range(batch_size)
    ]
    expected = [circuit.evaluate_plaintext(v) for v in batch]
    single = VerificationBackend(rounds=rounds)
    batched = PepperBackend(rounds=rounds)

    print(f"\n  batch={batch_size} width={width} rounds={rounds}")
    t_single = bench(
        "per-instance check (1 challenge per result)",
        lambda: [
            single.verify_affine(circuit, v, c, e)
            for v, c, e in zip(batch, expected, expected)
        ],
    )
    t_batch = bench(
        "batched check (1 shared challenge per round)",
        lambda: batched.verify_batch(circuit, expected, expected),
    )
    print(f"    {'per result: single vs batched':<52} "
          f"{t_single / batch_size * 1e3:9.3f} vs {t_batch / batch_size * 1e3:9.3f} ms")
    print(f"    {'batch speedup':<52} {t_single / t_batch:9.2f}x")


def main() -> None:
    print("Pepper break-even: what batching the affine check actually saves")
    print("  Offline setup is paid once; these lines are the online cost only.")
    for batch_size in (1, 8, 32, 128):
        run(batch_size)
    print()
    print("  Reading this honestly:")
    print("    - Both sides recompute expected results locally, so the saving")
    print("      is fewer challenges and one combined comparison per round.")
    print("    - At batch=1 the two paths should tie; the gap opens with N.")
    print("    - Less-than-execution verification needs a real proof system,")
    print("      not this bench. See exlex.backends.zk.")


if __name__ == "__main__":
    main()
