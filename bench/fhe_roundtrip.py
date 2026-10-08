"""Exercises the real FHE path end to end, if TenSEAL is installed.

    python bench/fhe_roundtrip.py

Skips cleanly when the extra is absent, which is the point: the library is
usable without it, and the report says so rather than raising.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import exlex  # noqa: E402
from exlex.backends import FHEBackend  # noqa: E402

VALUES = [1.0, 2.0, 3.0]
WEIGHTS = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
BIAS = [0.5, 0.5]


def main() -> int:
    backend = FHEBackend()
    if backend.available() is not None:
        print("TenSEAL not usable; nothing to demonstrate.")
        print("Install with: pip install 'exlex[fhe]'")
        return 0

    claim = backend.establish()
    print(f"claim: {claim.status.value} via {claim.mechanism}")
    print(f"state: {backend.state()}")
    print()

    public = backend.public_context()
    print("The host receives only this context:")
    print(f"  is_public={public.is_public()}  has_secret_key={public.has_secret_key()}")
    print()

    def remote_evaluate(ciphertext, weights, bias):
        """Stands in for the host. No secret key, so it cannot read anything."""
        return ciphertext.mm(weights) + list(bias)

    result = backend.run_remote(remote_evaluate, VALUES, WEIGHTS, BIAS)

    expected = [
        sum(VALUES[i] * WEIGHTS[i][j] for i in range(len(VALUES))) + BIAS[j]
        for j in range(len(BIAS))
    ]
    print(f"inputs:  {VALUES}")
    print(f"weights: {WEIGHTS}")
    print(f"bias:    {BIAS}")
    print(f"result:    {result}")
    print(f"expected:  {expected}")
    print()
    print("The result is approximate. CKKS evaluates on real numbers to finite")
    print("precision, so it lands near the plaintext answer, not on it exactly.")
    print()

    report = exlex.protect(backends=[backend])
    print(report.explain())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
