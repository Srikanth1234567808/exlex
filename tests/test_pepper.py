"""Pepper-style batched verification: one challenge for the whole batch.

The load-bearing property is the same as the single-instance check: a batch
with any corrupted instance must fail. Tests corrupt each position in turn,
because a batch check that only catches the first instance is not a check.
"""

from __future__ import annotations

import pytest

import exlex
from exlex.backends import PepperBackend
from exlex.circuit import AffineLayer, BiasLayer, Circuit, PolynomialLayer
from exlex.concerns import Concern, Status
from exlex.errors import UnsupportedComputation

WEIGHTS = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
BIAS = [0.5, 0.5]


def affine_circuit() -> Circuit:
    return Circuit(
        layers=[
            AffineLayer(weights=tuple(tuple(r) for r in WEIGHTS), name="fc"),
            BiasLayer(bias=tuple(BIAS), name="fc.bias"),
        ]
    )


def expected_for(values) -> list:
    return affine_circuit().evaluate_plaintext(values)


BATCH_INPUTS = [[1.0, 2.0, 3.0], [0.5, -1.0, 2.0], [-2.0, 0.0, 1.0]]
BATCH_EXPECTED = [expected_for(v) for v in BATCH_INPUTS]


def test_honest_batch_passes():
    backend = PepperBackend(rounds=5)
    assert backend.verify_batch(affine_circuit(), BATCH_EXPECTED, BATCH_EXPECTED)


def test_corruption_in_any_instance_is_caught():
    backend = PepperBackend(rounds=5)
    for index in range(len(BATCH_EXPECTED)):
        corrupted = [list(row) for row in BATCH_EXPECTED]
        corrupted[index][0] += 1.0
        assert not backend.verify_batch(affine_circuit(), corrupted, BATCH_EXPECTED), (
            f"corruption in instance {index} slipped through"
        )


def test_corruption_in_any_output_position_is_caught():
    backend = PepperBackend(rounds=5)
    for pos in range(len(BATCH_EXPECTED[0])):
        corrupted = [list(row) for row in BATCH_EXPECTED]
        corrupted[1][pos] += 0.5
        assert not backend.verify_batch(affine_circuit(), corrupted, BATCH_EXPECTED), (
            f"corruption at output {pos} slipped through"
        )


def test_batch_of_one_matches_single_instance():
    backend = PepperBackend(rounds=5)
    assert backend.verify_affine(
        affine_circuit(), BATCH_INPUTS[0], BATCH_EXPECTED[0], BATCH_EXPECTED[0]
    )
    assert not backend.verify_affine(
        affine_circuit(),
        BATCH_INPUTS[0],
        [BATCH_EXPECTED[0][0] + 1.0, BATCH_EXPECTED[0][1]],
        BATCH_EXPECTED[0],
    )


def test_length_mismatch_is_rejected():
    backend = PepperBackend()
    assert not backend.verify_batch(affine_circuit(), BATCH_EXPECTED, BATCH_EXPECTED[:2])
    assert not backend.verify_batch(affine_circuit(), [], [])
    assert not backend.verify_batch(affine_circuit(), [[1.0]], [[1.0, 2.0]])


def test_polynomial_circuit_is_refused():
    backend = PepperBackend()
    circuit = Circuit(
        layers=[
            AffineLayer(weights=((1.0,),), name="fc"),
            PolynomialLayer(coefficients=(0.0, 1.0), name="act", approximates="ReLU"),
        ]
    )
    with pytest.raises(UnsupportedComputation, match="not affine"):
        backend.verify_batch(circuit, [[1.0]], [[1.0]])


def test_rounds_must_be_positive():
    with pytest.raises(ValueError):
        PepperBackend(rounds=0)


def test_claim_is_partial_with_batching_residual():
    claim = PepperBackend(rounds=3).establish()
    assert claim.concern is Concern.CORRECTNESS
    assert claim.status is Status.PARTIAL
    joined = " ".join(claim.residual)
    assert "batch" in joined.lower()
    assert "recomputes" in joined.lower() or "recompute" in joined.lower()


def test_residual_bound_matches_single_instance_semantics():
    backend = PepperBackend(rounds=3)
    assert backend.residual_bound(1) == 1.0
    assert backend.residual_bound(0) == 1.0
    assert backend.residual_bound(64) < 1.0
    assert PepperBackend(rounds=10).residual_bound(64) < PepperBackend(rounds=1).residual_bound(64)


def test_available_and_protect_compose():
    assert PepperBackend().available() is None
    report = exlex.protect(backends=[PepperBackend()])
    assert report.status(Concern.CORRECTNESS) is Status.PARTIAL
