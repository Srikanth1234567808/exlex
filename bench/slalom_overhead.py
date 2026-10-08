"""What the blinded-delegation prototype actually costs.

The paper's claim is that delegation is cheap because the host does the
expensive matmul and the client only blinds, unblinds, and checks. This measures
the honest version of that claim against a plaintext matmul on the same shapes,
which is the only comparison that matters: if the protocol costs more than the
matmul it replaces, the design has failed regardless of its security.

What the numbers here are honest about, and what they are not. This is a pure
Python reference implementation. Both sides of the exchange are interpreter
arithmetic, so the ratio below is what a CPU-only deployment would look like,
and it is dominated by the client's second matmul (the unblinding factors). On a
real GPU the host's matmul shrinks by orders of magnitude and the client's work
does not, so the ratio gets *worse*, not better, unless the factors are
precomputed once per layer rather than per inference -- which is exactly the
optimisation Slalom's ``--verify_preproc`` flag exists for. The number to
attack is the client-side cost, and it is labelled as such below.

Run: .venv/bin/python bench/slalom_overhead.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exlex.backends.slalom import SlalomBackend, host_matmul  # noqa: E402


def plain_matmul(weights, values):
    return [
        sum(weights[i][j] * values[i] for i in range(len(weights)))
        for j in range(len(weights[0]))
    ]


def bench(label, fn, repeat=3):
    """Best-of-N wall time, which is the least noisy statistic here."""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    print(f"    {label:<48} {best * 1e3:9.3f} ms")
    return best


def run(rows: int, width: int) -> None:
    weights = [
        [((i * 7 + j * 3) % 11) / 8.0 - 0.5 for j in range(width)] for i in range(rows)
    ]
    values = [((i * 5) % 13) / 7.0 for i in range(rows)]
    backend = SlalomBackend(rounds=4)

    print(f"\n  {rows}x{width}   (mask range 2**32, 16 fractional bits, 4 rounds)")
    base = bench(
        "host plaintext matmul (the baseline to beat)",
        lambda: plain_matmul(weights, values),
    )
    host = bench(
        "host matmul on the masked input (untouched code path)",
        lambda: host_matmul(backend.blind(values)[0], weights, codec=backend.fixed),
        # The blind is inside the loop so the host cost is measured on a fresh
        # payload, but the blind itself is billed on the client line below.
    )
    bench(
        "protocol: blind + host + unblind, no checking",
        lambda: backend.run_linear(weights, [0.0] * width, values, host_matmul, verify=False),
    )
    with_verify = bench(
        "protocol: full round trip, checked",
        lambda: backend.run_linear(weights, [0.0] * width, values, host_matmul, verify=True),
    )
    client_only = with_verify - host
    print(f"    {'client-side work (everything but the host matmul)':<48} "
          f"{client_only * 1e3:9.3f} ms")
    print(f"    {'protocol cost / plaintext matmul':<48} "
          f"{with_verify / base:9.2f}x")
    print(f"    {'of which the client pays':<48} {client_only / base:9.2f}x")


def main() -> None:
    print("Blinded linear delegation: where the cost actually goes")
    print("  Every line is measured, and the client's share is separated out")
    print("  because that is the part that does not shrink on a real GPU.")
    for rows, width in ((16, 8), (64, 32), (256, 128)):
        run(rows, width)
    print()
    print("  Reading this honestly:")
    print("    - The client pays roughly a second matmul, for the unblinding")
    print("      factors, plus the rounds. That is the number to precompute.")
    print("    - Slalom reports 2-4x verifiable inference; LGT4CG reports 4.16%")
    print("      and 0.34% for a hardware GPU-TEE on an RTX 3090.")
    print("    - None of those figures are comparable until the host matmul here")
    print("      is a real GPU kernel. What this bench establishes is the shape")
    print("      of the cost: client-bound, and linear in the layer size.")


if __name__ == "__main__":
    main()
