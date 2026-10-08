"""Result verification: does a wrong answer actually get caught?

The property under test is that a cheating host is detected. A test that only
checked a correct answer passes would be worthless, so the corruption cases are
the load-bearing ones here.
"""

from __future__ import annotations

import pytest

import exlex
from exlex.backends import VerificationBackend
from exlex.circuit import AffineLayer, BiasLayer, Circuit, PolynomialLayer
from exlex.concerns import Concern, Status
from exlex.errors import UnsupportedComputation

VALUES = [1.0, 2.0, 3.0]
# Rows are outputs (in, out): 3 inputs, 2 outputs.
WEIGHTS = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
BIAS = [0.5, 0.5]
EXPECTED = [4.5, 5.5]


def affine_circuit() -> Circuit:
    return Circuit(
        layers=[
            AffineLayer(weights=tuple(tuple(r) for r in WEIGHTS), name="fc"),
            BiasLayer(bias=tuple(BIAS), name="fc.bias"),
        ]
    )


def test_correct_answer_passes():
    backend = VerificationBackend()
    assert backend.verify_affine(affine_circuit(), VALUES, EXPECTED, EXPECTED)


def test_single_corrupted_output_is_caught():
    backend = VerificationBackend(rounds=5)
    corrupted = [EXPECTED[0] + 1.0, EXPECTED[1]]
    assert not backend.verify_affine(affine_circuit(), VALUES, corrupted, EXPECTED)


def test_every_single_position_corruption_is_caught():
    """A check that only catches the first output is not a check."""
    backend = VerificationBackend(rounds=5)
    for index in range(len(EXPECTED)):
        corrupted = list(EXPECTED)
        corrupted[index] += 0.5
        assert not backend.verify_affine(
            affine_circuit(), VALUES, corrupted, EXPECTED
        ), f"corruption at index {index} slipped through"


def test_swapped_outputs_are_caught():
    backend = VerificationBackend(rounds=5)
    swapped = [EXPECTED[1], EXPECTED[0]]
    assert not backend.verify_affine(affine_circuit(), VALUES, swapped, EXPECTED)


def test_wrong_length_is_rejected():
    backend = VerificationBackend()
    assert not backend.verify_affine(affine_circuit(), VALUES, [1.0], EXPECTED)


def test_challenge_is_random_not_deterministic():
    backend = VerificationBackend()
    draws = {tuple(backend.challenge(4)) for _ in range(20)}
    assert len(draws) > 1


def test_challenge_is_bounded():
    backend = VerificationBackend()
    for value in backend.challenge(64):
        assert -1.0 <= value <= 1.0


def test_more_rounds_shrink_the_residual_bound():
    few = VerificationBackend(rounds=1)
    many = VerificationBackend(rounds=10)
    assert many.residual_bound(64) < few.residual_bound(64)


def test_residual_bound_shrinks_with_wider_output():
    """A narrow output is harder to check, so it is the *worse* case.

    The bound is 1/d per round, so a 2-wide output is a far weaker guarantee
    than a 1024-wide one. Worth pinning because the intuitive direction is
    backwards.
    """
    backend = VerificationBackend(rounds=3)
    assert backend.residual_bound(2) < backend.residual_bound(1024)


def test_residual_bound_is_one_for_degenerate_widths():
    backend = VerificationBackend(rounds=3)
    assert backend.residual_bound(1) == 1.0
    assert backend.residual_bound(0) == 1.0


def test_rounds_must_be_positive():
    with pytest.raises(ValueError):
        VerificationBackend(rounds=0)


def test_polynomial_circuit_is_refused_not_guessed():
    backend = VerificationBackend()
    circuit = Circuit(
        layers=[
            AffineLayer(weights=((1.0,),), name="fc"),
            PolynomialLayer(coefficients=(0.0, 1.0), name="act", approximates="ReLU"),
        ]
    )
    with pytest.raises(UnsupportedComputation, match="not affine"):
        backend.verify_affine(circuit, [1.0], [1.0], [1.0])


def test_claim_is_partial_and_names_the_bound():
    backend = VerificationBackend(rounds=3)
    claim = backend.establish()
    assert claim.concern is Concern.CORRECTNESS
    assert claim.status is Status.PARTIAL
    assert any("probabilistic" in item for item in claim.residual)
    assert any("polynomial" in item for item in claim.residual)


def test_backend_is_always_available():
    assert VerificationBackend().available() is None


def test_verify_compares_against_a_known_expected():
    backend = VerificationBackend()
    assert backend.verify(VALUES, EXPECTED, [4.5, 5.5])
    assert not backend.verify(VALUES, EXPECTED, [4.5, 5.6])


def test_verify_rejects_length_mismatch():
    backend = VerificationBackend()
    assert not backend.verify(VALUES, EXPECTED, [1.0])


def test_partial_does_not_satisfy_is_protected():
    report = exlex.protect(backends=[VerificationBackend()])
    assert report.status(Concern.CORRECTNESS) is Status.PARTIAL
    assert report.is_protected(Concern.CORRECTNESS) is False
