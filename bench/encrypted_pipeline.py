"""End to end: compile a circuit, evaluate it on ciphertext, verify the result.

    python bench/encrypted_pipeline.py

The interesting part is the last section. A well-formed but wrong result is fed
back through the verifier, because "the host returned a number" and "the host
returned the right number" are different claims and only the second one matters.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import exlex  # noqa: E402
from exlex.backends import FHEBackend, VerificationBackend  # noqa: E402
from exlex.circuit import AffineLayer, BiasLayer, Circuit  # noqa: E402

INPUTS = [1.0, 2.0, 3.0]


def build_circuit() -> Circuit:
    """Two layers, 3 inputs to 2 outputs. Rows are inputs, columns are outputs."""
    return Circuit(
        layers=[
            AffineLayer(
                weights=((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
                name="fc1",
            ),
            BiasLayer(bias=(0.5, 0.5), name="fc1.bias"),
            AffineLayer(
                weights=((2.0, 0.0), (0.0, 2.0)),
                name="fc2",
            ),
            BiasLayer(bias=(1.0, 1.0), name="fc2.bias"),
        ]
    )


def main() -> int:
    circuit = build_circuit()
    print("=== what the host receives ===")
    print(circuit.describe())
    print()
    print("This is the leak: operation sequence and every width, in the clear.")
    print("Layer names and weights stay home.")
    print()

    expected = circuit.evaluate_plaintext(INPUTS)
    print(f"local reference: {expected}")
    print()

    backend = FHEBackend()
    if backend.available() is not None:
        print("TenSEAL not usable; showing the report only.")
        print(backend.available())
        print()
        print(exlex.protect().explain())
        return 0

    backend.establish()
    result = backend.run_circuit(circuit, INPUTS)
    print(f"encrypted result: {result}")
    print()

    verifier = VerificationBackend(rounds=5)
    honest = verifier.verify_affine(circuit, INPUTS, result, expected)
    print(f"honest result verifies: {honest}")
    assert honest, "an honest result must verify"

    # Now the interesting case: a host that returns well-formed nonsense.
    corrupted = list(result)
    corrupted[0] += 1.0
    caught = verifier.verify_affine(circuit, INPUTS, corrupted, expected)
    print(f"corrupted result accepted: {caught}")
    assert not caught, "a corrupted result must be rejected"
    print("  -> caught, as it should be")
    print()

    print("=== protection report ===")
    report = exlex.protect(backends=[backend, verifier])
    print(report.explain())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
