"""Tour of the Slalom-derived prototype. No GPU, no torch, no network.

    python examples/slalom_prototype.py

The demonstration is of one thing: a host that receives the payload cannot read
it, and a host that lies about the answer is caught. Both halves are shown, and
the second half is shown by actually attacking the host.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from exlex.backends.slalom import SlalomBackend, host_matmul  # noqa: E402

WEIGHTS = [[0.5, -1.25], [2.0, 0.0], [-0.75, 0.125]]
VALUES = [1.5, -2.0, 0.25]
BIAS = [0.1, -0.2]


def expected():
    return [
        sum(WEIGHTS[i][j] * VALUES[i] for i in range(3)) + BIAS[j] for j in range(2)
    ]


def main() -> None:
    backend = SlalomBackend(rounds=4)

    print("=" * 74)
    print("1. An honest host: the client's answer is exact")
    print("=" * 74)
    out = backend.run_linear(WEIGHTS, BIAS, VALUES, host_matmul)
    print(f"  plaintext reference : {expected()}")
    print(f"  delegated result    : {[round(v, 6) for v in out['result']]}")
    print(f"  verified            : {out['verified']}  ({out['rounds']} rounds)")

    print()
    print("=" * 74)
    print("2. What the host actually received")
    print("=" * 74)
    fixed = backend.fixed
    for sent, value in zip(out["masked_input"], VALUES):
        print(f"  input: value {value:>6}  sent {sent:>22}  (encoded {fixed.encode_scaled(value):>12})")
    print("  the encoded value is present only inside a 32-bit pad, so the host")
    print("  sees W @ r and learns nothing about the input it came from.")

    print()
    print("=" * 74)
    print("3. Attack: the host returns a wrong answer")
    print("=" * 74)
    out2 = backend.run_linear(
        WEIGHTS, [0.0, 0.0], VALUES, lambda m, w: [0, 0]
    )
    print(f"  host returned all zeros -> verified: {out2['verified']}")

    def off_by_one(masked, weights_seen):
        return [v + 1 for v in host_matmul(masked, weights_seen, codec=backend.fixed)]

    out3 = backend.run_linear(WEIGHTS, [0.0, 0.0], VALUES, off_by_one)
    print(f"  host off by one unit   -> verified: {out3['verified']}")

    def lazy(masked, weights_seen):
        width = len(weights_seen[0])
        return [sum(masked) % backend.fixed.mask_range] * width

    out4 = backend.run_linear(WEIGHTS, [0.0, 0.0], VALUES, lazy)
    print(f"  host skipped the matmul-> verified: {out4['verified']}")

    print()
    print("=" * 74)
    print("4. What the claim is, and what it is not")
    print("=" * 74)
    claim = backend.establish()
    print(f"  concern   : {claim.concern.value}")
    print(f"  status    : {claim.status.value}")
    print(f"  mechanism : {claim.mechanism}")
    print("  pass probability for a cheating host, stated:")
    print(f"    {backend.pass_probability():.3e}  (2**-128 at 4 rounds)")
    print("  residuals:")
    for r in claim.residual:
        print(f"    - {r}")


if __name__ == "__main__":
    main()
