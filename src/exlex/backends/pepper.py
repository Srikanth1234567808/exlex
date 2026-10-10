"""Batched affine verification, after the Pepper project.

Pepper (Setty et al., NYU/UT Austin) made verifiable outsourced computation
practical by amortising the check: one random challenge covers a whole batch
of instances, so per-instance verification cost drops and the client breaks
even against local execution sooner. The proving half needed a constraint
compiler (Pequin) and is not reproduced here; what this module takes is the
batching idea, applied to the randomised affine check in
:mod:`exlex.backends.verify`.

What this is: given a batch of claimed host results and locally computed
expected results, draw one shared challenge vector per round plus fresh
batch coefficients, and compare a single combined scalar per round. A batch
in which any instance is wrong passes all rounds with probability at most
``(1 - 1/d) ** rounds`` where ``d`` is the output width -- batching
amortises challenge cost, it does not strengthen the bound, and the residual
says so.

What this is not: a proof system. The verifier still recomputes the expected
results locally, so there is no asymptotic saving yet, only fewer random
draws and dot products per instance. A deployment that needs the verifier to
do less work than the execution needs a real SNARK behind
:mod:`exlex.backends.zk`, not this module. The claim is therefore
:attr:`~exlex.concerns.Status.PARTIAL`, exactly as the single-instance check.
"""

from __future__ import annotations

import secrets
from typing import List, Optional, Sequence

from ..circuit import Circuit, PolynomialLayer
from ..concerns import Backend, Claim, Concern, Status
from ..errors import UnsupportedComputation

_CHALLENGE_LOW = -1.0
_CHALLENGE_HIGH = 1.0


class PepperBackend(Backend):
    """One challenge for a whole batch of affine results.

    Args:
        rounds: independent combined checks. More rounds multiply a cheating
            batch's pass probability down, at one combined dot product each.
    """

    concerns = (Concern.CORRECTNESS,)

    def __init__(self, rounds: int = 3) -> None:
        if rounds < 1:
            raise ValueError("rounds must be at least 1")
        self.rounds = rounds

    def available(self) -> Optional[str]:
        """Always usable. Batching needs no optional dependency."""
        return None

    def establish(self) -> Claim:
        """Report correctness as partial, with the batching caveat stated."""
        return Claim(
            concern=Concern.CORRECTNESS,
            status=Status.PARTIAL,
            mechanism=f"Pepper-style batched affine verification ({self.rounds} rounds)",
            residual=(
                "Verification is probabilistic, not a proof. A batch with any "
                "corrupted instance passes all rounds with probability at most "
                "(1 - 1/d) ** rounds, where d is the output width; batching "
                "amortises challenge cost, it does not strengthen the bound.",
                "Only affine circuits are covered. A circuit with a polynomial "
                "approximation is refused here.",
                "The verifier still recomputes every expected result locally. "
                "The saving over per-instance checks is fewer challenges and "
                "one combined comparison per round, not less-than-execution "
                "verification. That needs a real proof via exlex.backends.zk.",
                "Challenges and batch coefficients are drawn locally after all "
                "results arrive, so the host cannot tailor answers to them.",
            ),
            detail={"rounds": self.rounds, "guarantee": "probabilistic-batched"},
        )

    def residual_bound(self, dimension: int) -> float:
        """Pass probability for a cheating batch, given the output width."""
        if dimension <= 1:
            return 1.0
        return (1.0 - 1.0 / dimension) ** self.rounds

    def challenge(self, dimension: int) -> List[float]:
        """A fresh shared challenge vector from the OS CSPRNG."""
        span = _CHALLENGE_HIGH - _CHALLENGE_LOW
        return [
            _CHALLENGE_LOW + span * (secrets.randbits(53) / float(1 << 53))
            for _ in range(dimension)
        ]

    def batch_coefficients(self, batch_size: int) -> List[float]:
        """Fresh per-round batch weights, drawn after results arrive."""
        span = _CHALLENGE_HIGH - _CHALLENGE_LOW
        return [
            _CHALLENGE_LOW + span * (secrets.randbits(53) / float(1 << 53))
            for _ in range(batch_size)
        ]

    def verify_batch(
        self,
        circuit: Circuit,
        batch_claimed: Sequence[Sequence[float]],
        batch_expected: Sequence[Sequence[float]],
        *,
        tolerance: float = 1e-6,
    ) -> bool:
        """Check a whole batch with one combined comparison per round.

        Args:
            circuit: the compiled circuit, used to refuse non-affine work.
            batch_claimed: what the host returned, one result per instance.
            batch_expected: locally recomputed references, same layout.
            tolerance: combined-scalar comparison tolerance.

        Returns:
            ``True`` if every round passed.

        Raises:
            UnsupportedComputation: when the circuit is not affine.
        """
        self._require_affine(circuit)
        if len(batch_claimed) != len(batch_expected) or not batch_claimed:
            return False
        dimension = len(batch_expected[0])
        if dimension == 0:
            return False
        for claimed, expected in zip(batch_claimed, batch_expected):
            if len(claimed) != dimension or len(expected) != dimension:
                return False
        for _ in range(self.rounds):
            r = self.challenge(dimension)
            s = self.batch_coefficients(len(batch_claimed))
            lhs = sum(
                si * sum(ri * ci for ri, ci in zip(r, claimed))
                for si, claimed in zip(s, batch_claimed)
            )
            rhs = sum(
                si * sum(ri * ei for ri, ei in zip(r, expected))
                for si, expected in zip(s, batch_expected)
            )
            if abs(lhs - rhs) > tolerance:
                return False
        return True

    def verify_affine(
        self,
        circuit: Circuit,
        inputs: Sequence[float],
        claimed: Sequence[float],
        expected: Sequence[float],
    ) -> bool:
        """Single-instance check, as a batch of one. Prefer :meth:`verify_batch`."""
        del inputs  # reference is supplied directly, as in VerificationBackend
        return self.verify_batch(circuit, [claimed], [expected])

    def _require_affine(self, circuit: Circuit) -> None:
        for layer in circuit.layers:
            if isinstance(layer, PolynomialLayer):
                raise UnsupportedComputation(
                    f"layer {layer.name!r} is a {layer.degree}-degree polynomial "
                    "approximating "
                    f"{layer.approximates}, so the circuit is not affine and the "
                    "batched check does not apply to it. Supply a real proof"
                )
        return None
