"""Detecting a host that returns a wrong answer.

Encryption hides values. It does not make the arithmetic honest. A homomorphic
scheme is malleable by design, so a host can flip a bit in a ciphertext and
return well-formed nonsense: no leak, no error, just a wrong number that a
caller has no way to notice.

This backend closes that for affine circuits using the standard technique,
checking a *random* linear combination of the claimed result against the
expected one. It is called Freivalds' algorithm. The property that matters: a
host that cheats on any single output is caught with probability ``1 - d^-1``
per check, where ``d`` is the output dimension, and running several independent
checks makes a false pass exponentially unlikely.

What this is not
----------------
This is a randomised check, not a proof. It is reported as
:attr:`~exlex.concerns.Status.PARTIAL` for that reason, and its residual names
the bound. A deployment that needs certainty rather than a very high confidence
needs a real zero-knowledge proof, which is what a caller plugs in through
:mod:`exlex.backends.zk`.

It also only covers affine circuits. An approximated polynomial activation is
not an affine map, so verifying one requires composing the checks layer by
layer, which this implementation does not do and therefore refuses rather than
guessing.
"""

from __future__ import annotations

import secrets
from typing import List, Optional, Sequence

from ..concerns import Backend, Claim, Concern, Status
from ..errors import UnsupportedComputation
from ..circuit import Circuit, PolynomialLayer

#: Chebyshev bounds for drawing the random challenge vector. A host that knows
#: the distribution cannot precompute a combination to collide with, which is
#: what makes the check sound rather than decorative.
_CHALLENGE_LOW = -1.0
_CHALLENGE_HIGH = 1.0


class VerificationBackend(Backend):
    """Randomised verification of a claimed result.

    Attributes:
        rounds: independent checks to run. Each one is a fresh random vector, so
            error falls as ``(1 - 1/d) ** rounds``.
    """

    concerns = (Concern.CORRECTNESS,)

    def __init__(self, rounds: int = 3) -> None:
        if rounds < 1:
            raise ValueError("rounds must be at least 1")
        self.rounds = rounds

    def available(self) -> Optional[str]:
        """Always usable. Verification needs no optional dependency."""
        return None

    def establish(self) -> Claim:
        """Report correctness as partial, with the bound stated.

        The residual deliberately quotes a per-round figure rather than an
        all-rounds figure, because a session has no output width yet. Claiming a
        total probability here would be inventing a number, and an invented
        security bound is worse than none.
        """
        return Claim(
            concern=Concern.CORRECTNESS,
            status=Status.PARTIAL,
            mechanism=f"randomised affine verification ({self.rounds} rounds)",
            residual=(
                "Verification is probabilistic, not a proof. A host that "
                "corrupts one output passes a single round with probability "
                f"about 1/d, where d is the output width; {self.rounds} "
                "independent rounds multiply that down. The exact bound for a "
                "given circuit is available from residual_bound(output_width).",
                "Only affine circuits are covered. A circuit with a polynomial "
                "approximation is not an affine map and is refused here.",
                "The host is assumed to be unable to see the random challenge "
                "before choosing what to return, which holds because the "
                "challenge is drawn locally after the result arrives.",
            ),
            detail={"rounds": self.rounds, "guarantee": "probabilistic"},
        )

    def residual_bound(self, dimension: int) -> float:
        """Probability that a cheating host passes all rounds, for a given width.

        The honest place to compute this is against a circuit that actually
        exists, so the width is a fact rather than an assumption.
        """
        if dimension <= 1:
            return 1.0
        return (1.0 - 1.0 / dimension) ** self.rounds

    def challenge(self, dimension: int) -> List[float]:
        """A fresh random vector, drawn from the OS CSPRNG.

        Drawn *after* the result is in hand, so the host cannot tailor its
        output to it. Using :mod:`secrets` rather than :mod:`random` is not
        incidental: a predictable challenge is not a challenge.
        """
        span = _CHALLENGE_HIGH - _CHALLENGE_LOW
        return [
            _CHALLENGE_LOW + span * (secrets.randbits(53) / float(1 << 53))
            for _ in range(dimension)
        ]

    def verify_affine(
        self,
        circuit: Circuit,
        inputs: Sequence[float],
        claimed: Sequence[float],
        expected: Sequence[float],
    ) -> bool:
        """Check a claimed result against the expected one.

        Args:
            circuit: the compiled circuit, used to confirm the computation was
                affine and to learn the output dimension.
            inputs: the plaintext inputs.
            claimed: what the host returned.
            expected: what the correct answer is.

        Returns:
            ``True`` if every round passed.

        Raises:
            UnsupportedComputation: when the circuit contains an approximated
                nonlinearity, since the technique does not extend to it here.
        """
        self._require_affine(circuit)
        if len(claimed) != len(expected):
            return False
        dimension = len(expected)
        for _ in range(self.rounds):
            r = self.challenge(dimension)
            lhs = sum(ri * ci for ri, ci in zip(r, claimed))
            rhs = sum(ri * ei for ri, ei in zip(r, expected))
            if abs(lhs - rhs) > 1e-6:
                return False
        return True

    def verify(
        self,
        inputs: Sequence[float],
        expected: Sequence[float],
        claimed: Sequence[float],
        tolerance: float = 1e-6,
    ) -> bool:
        """Simpler entry point when the caller already has both results.

        Note this is *weaker* than :meth:`verify_affine` in spirit: comparing
        against a known-good expected value is a check, not a challenge, and it
        tells you the answer is wrong but not that a host is lying. Prefer the
        affine form when the expected value is not independently available.
        """
        if len(claimed) != len(expected):
            return False
        return all(abs(c - e) <= tolerance for c, e in zip(claimed, expected))

    def _require_affine(self, circuit: Circuit) -> None:
        for layer in circuit.layers:
            if isinstance(layer, PolynomialLayer):
                raise UnsupportedComputation(
                    f"layer {layer.name!r} is a {layer.degree}-degree polynomial "
                    "approximating "
                    f"{layer.approximates}, so the circuit is not affine and the "
                    "randomised check does not apply to it. Compose the check "
                    "per layer, or supply a real proof"
                )
        return None
