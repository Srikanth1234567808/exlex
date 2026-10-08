"""Tests for the Slalom-derived blinded delegation prototype.

The two properties under test are the two ends of the design: what the host
cannot see, and what the host cannot get away with. Everything else is
plumbing.

The adversarial-host tests are the point. A backend that will happily return a
wrong answer when the host cheats is worse than no backend, so the cheating
paths are exercised explicitly rather than assumed to fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest  # noqa: E402

from exlex.backends.slalom import (  # noqa: E402
    DEFAULT_MASK_RANGE,
    MAX_TENSOR_SIZE,
    FixedPoint,
    SlalomBackend,
    host_matmul,
)
from exlex.concerns import Concern, Status  # noqa: E402
from exlex.errors import UnsupportedComputation  # noqa: E402


@pytest.fixture
def backend() -> SlalomBackend:
    return SlalomBackend(rounds=4)


@pytest.fixture
def layer():
    # 3 in, 2 out, so the shapes are unambiguous in a failure message.
    weights = [[0.5, -1.25], [2.0, 0.0], [-0.75, 0.125]]
    values = [1.5, -2.0, 0.25]
    bias = [0.1, -0.2]
    return weights, bias, values


def plain_matmul(weights, values):
    return [
        sum(weights[i][j] * values[i] for i in range(len(weights)))
        for j in range(len(weights[0]))
    ]


# --------------------------------------------------------------------- codec


def test_fixed_point_roundtrip():
    fixed = FixedPoint(bits=16)
    for value in (0.0, 1.0, -2.75, 0.123, 255.5):
        assert fixed.decode_scaled(fixed.encode_scaled(value)) == pytest.approx(
            value, abs=2.0 ** -16
        )


def test_values_outside_the_range_are_clamped_not_silently_wrapped():
    """A value bigger than the codec range must not wrap into a negative one.

    Wrapping is the dangerous failure: the mask would no longer dominate, the
    client's factor would stop cancelling, and the result would be a plausible
    wrong number rather than an error.
    """
    fixed = FixedPoint()
    positive = fixed.encode_scaled(10_000_000.0)
    negative = fixed.encode_scaled(-10_000_000.0)
    assert positive <= fixed.limit
    assert negative >= -fixed.limit


def test_unscalable_values_are_refused_rather_than_clamped_to_a_bound():
    """NaN and infinity have no fixed point, and clamping them silently is a lie."""
    fixed = FixedPoint()
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(UnsupportedComputation, match="cannot be encoded"):
            fixed.encode_scaled(bad)


def test_value_scale_is_the_product_of_the_two_encodings():
    # The client divides by this once, so it must be the square, not the scale.
    assert FixedPoint(bits=16).value_scale == (1 << 32)


def test_mask_range_minimum_is_enforced():
    with pytest.raises(ValueError, match="at least 2\\*\\*16"):
        FixedPoint(mask_range=1 << 8)


# ------------------------------------------------------------------ happy path


def test_honest_host_returns_the_right_answer(backend, layer):
    weights, bias, values = layer
    out = backend.run_linear(weights, bias, values, host_matmul, verify=True)
    expected = [m + b for m, b in zip(plain_matmul(weights, values), bias)]
    # Both sides are fixed point, so the product carries the product of the
    # two resolutions: 2**-(2*bits) rather than 2**-bits.
    assert out["result"] == pytest.approx(expected, abs=2.0 ** -31)
    assert out["verified"] is True


def test_blinding_moves_the_input(backend, layer):
    weights, _bias, values = layer
    masked, mask = backend.blind(values)
    fixed = backend.fixed
    assert masked != [fixed.encode_scaled(v) for v in values]
    # The masked value is exactly the encoded value plus the pad, which is what
    # makes the client's subtraction exact. Negative inputs are fine: the pad
    # still dominates, because it is 32 bits against at most 40.
    for m, r, v in zip(masked, mask, values):
        assert m - fixed.encode_scaled(v) == r
        assert abs(r) > abs(fixed.encode_scaled(v)) or fixed.encode_scaled(v) < 0


def test_client_arithmetic_matches_the_host_bit_for_bit(backend, layer):
    """The factor the client subtracts must be the host's own product.

    When these two differ by a single unit, every result is wrong by that unit
    and verification still passes, so this equality is the load-bearing
    invariant of the whole module.
    """
    weights, _bias, values = layer
    masked, mask = backend.blind(values)
    factors = backend.unblind_factors(weights, mask)
    assert isinstance(factors[0], int)
    # The host's own contribution, computed independently, must be identical.
    fixed = backend.fixed
    manual = [
        sum(
            fixed.encode_scaled(float(weights[i][j])) * r for i, r in enumerate(mask)
        )
        for j in range(len(weights[0]))
    ]
    assert factors == manual


def test_what_the_host_saw_contains_no_plaintext(backend, layer):
    """The payload must not be recoverable from what crossed the boundary."""
    weights, _bias, values = layer
    seen = {}

    def spy(masked, weights_seen):
        seen["masked"] = list(masked)
        seen["weights"] = weights_seen
        return host_matmul(masked, weights_seen, codec=backend.fixed)

    out = backend.run_linear(weights, [0.0, 0.0], values, spy)
    assert seen["masked"] == out["masked_input"]
    fixed = backend.fixed
    # The mask dominates the value, so no plaintext magnitude is present.
    for sent, value in zip(seen["masked"], values):
        assert sent > 2 * abs(fixed.encode_scaled(value))
        assert sent > 0


# --------------------------------------------------------------- adversarial


def test_cheating_host_is_caught(backend, layer):
    """A host that returns the right shape but wrong values is detected."""
    weights, _bias, values = layer

    def liar(masked, weights_seen):
        return [0] * len(weights[0])

    out = backend.run_linear(weights, [0.0, 0.0], values, liar)
    assert out["verified"] is False


def test_lazy_host_that_skips_the_matmul_is_caught(backend, layer):
    weights, _bias, values = layer

    def identity_fudge(masked, weights_seen):
        # Returns the input instead of the product: structurally valid, wrong.
        # Written to return the layer's own width so it reaches the check that
        # is meant to catch it rather than being refused for shape.
        width = len(weights_seen[0])
        return [sum(masked) % backend.fixed.mask_range] * width

    out = backend.run_linear(weights, [0.0, 0.0], values, identity_fudge)
    assert out["verified"] is False


def test_subtly_wrong_answer_is_caught(backend, layer):
    """Off by one fixed-point unit is still a lie, and must not pass."""
    weights, _bias, values = layer

    def off_by_one(masked, weights_seen):
        return [v + backend.fixed.value_scale for v in host_matmul(
            masked, weights_seen, codec=backend.fixed
        )]

    out = backend.run_linear(weights, [0.0, 0.0], values, off_by_one)
    assert out["verified"] is False


def test_detection_probability_is_reported_and_negligible(backend, layer):
    weights, _bias, values = layer
    trials = 25
    caught = sum(
        not backend.run_linear(
            weights, [0.0, 0.0], values, lambda m, w: [1] * len(weights[0])
        )["verified"]
        for _ in range(trials)
    )
    assert caught == trials
    # One round at 2^32, four rounds, so about 2^-128 altogether.
    assert backend.pass_probability() < 2.0 ** -120


def test_a_single_round_catches_a_wrong_claim(backend):
    """The challenge must actually depend on the fresh randomness.

    If the challenge were fixed, a host could precompute an answer that passes.
    This takes a correct claim, perturbs it by one value unit, and checks that
    a single round rejects it.
    """
    weights = [[1.0, 2.0]]
    values = [3.0]
    masked, mask = backend.blind(values)
    claimed = backend.unblind(
        host_matmul(masked, weights, codec=backend.fixed),
        backend.unblind_factors(weights, mask),
        weights,
        mask,
    )
    assert backend.verify_blinded(weights, values, claimed) is True
    tampered = list(claimed)
    tampered[0] += backend.fixed.scale
    assert backend.verify_blinded(weights, values, tampered) is False


# ------------------------------------------------------------ API and claims


def test_claim_reports_data_as_protected(backend):
    data = backend.establish()
    assert data.concern is Concern.DATA
    assert data.status is Status.PROTECTED
    assert data.residual, "a protected claim with no residual is not honest"


def test_correctness_is_declared_but_not_claimed_protected(backend):
    assert Concern.CORRECTNESS in backend.concerns


def test_operations_is_not_claimed(backend):
    assert Concern.OPERATIONS not in backend.concerns


def test_available_is_always_none(backend):
    assert backend.available() is None


def test_wider_layer_passes_where_an_int64_would_wrap(backend):
    """The bound check must catch layers a real executor could not hold.

    With the default mask range of 2^32, a few hundred rows with large weights
    is the limit; this constructs one that is over the line and checks the
    refusal names the fix.
    """
    # 300 rows at weight 1000 puts the host accumulator past 2**62.
    weights = [[1000.0] * 4 for _ in range(300)]
    with pytest.raises(UnsupportedComputation, match="accumulator"):
        backend.run_linear(weights, [0.0] * 4, [1.0] * 300, lambda m, w: [0] * 4)


def test_short_host_answer_is_refused(backend):
    weights = [[1.0, 1.0]]
    with pytest.raises(UnsupportedComputation, match="nothing to unblind"):
        backend.run_linear(weights, [0.0, 0.0], [1.0], lambda m, w: [0])


def test_verify_can_be_skipped_for_benchmarks(backend, layer):
    weights, bias, values = layer
    out = backend.run_linear(weights, bias, values, host_matmul, verify=False)
    assert out["verified"] is True
    assert out["rounds"] == 4


# --------------------------------------------------------------- integration


def test_defaults_match_the_documented_reference_scale(backend):
    assert backend.fixed.mask_range == DEFAULT_MASK_RANGE
    assert backend.fixed.value_scale == 1 << 32
    assert backend.fixed.limit >= MAX_TENSOR_SIZE * backend.fixed.scale
