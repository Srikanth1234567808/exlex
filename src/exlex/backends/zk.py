"""Output correctness: detecting a host that returns a wrong answer.

A homomorphic scheme is malleable by design, which is what makes evaluation
possible and also lets a host flip a bit in your ciphertext and hand back
well-formed nonsense. The numbers stay secret; the *result* is not trustworthy.
Closing that needs a zero-knowledge proof that the evaluation was the one you
asked for.

This backend is a thin, honest shell. The proving and verifying backends
underneath are young, and a security library that presents an unverified prover
as verified is the exact failure this project exists to avoid. So
:meth:`establish` reports :attr:`~exlex.concerns.Status.UNAVAILABLE` until a
backend is configured, and the composed report says correctness is unverified
rather than implying the arithmetic was checked.

Users who need it today should wire a backend of their own: subclass, implement
:meth:`verify`, and register it. See :mod:`exlex.report` for registration.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from ..concerns import Backend, Claim, Concern, Status

#: Signature of a verification function. Returns ``True`` when the output is
#: proven to come from evaluating ``expected`` on ``inputs``.
Verifier = Callable[[Sequence[float], Sequence[float], Any], bool]


class ZKBackend(Backend):
    """Verifiable evaluation, delegated to a caller-supplied prover.

    Attributes:
        verify: a callable that verifies a proof and returns a bool. Required;
            without it this backend has nothing to establish, and the session
            reports correctness as unverified.
        scheme: a label for reports, e.g. ``"halo2"``. Purely descriptive, so
            that a reader can see what was actually in play.
    """

    concerns = (Concern.CORRECTNESS,)

    def __init__(self, verify: Optional[Verifier] = None, scheme: str = "unconfigured") -> None:
        self.verify = verify
        self.scheme = scheme

    def available(self) -> Optional[str]:
        """Unavailable until a verifier is supplied.

        Deliberately not probing for a prover library. A prover that is
        installed but unconfigured, or configured with wrong parameters, is
        indistinguishable from a working one at import time, and reporting that
        as available would be a false claim about the one concern this backend
        exists to close.
        """
        if self.verify is None:
            return (
                "no verifier configured; pass verify=<callable> to prove that "
                "outputs are the true result, otherwise a host can return "
                "well-formed wrong answers"
            )
        return None

    def establish(self) -> Claim:
        """Report correctness as protected once a verifier exists.

        Exists to mark the concern as deliberately covered, not to run a
        protocol. The actual check happens in :meth:`check`.
        """
        unavailable = self.available()
        if unavailable is not None:
            return Claim(
                concern=Concern.CORRECTNESS,
                status=Status.UNAVAILABLE,
                residual=(unavailable,),
            )
        return Claim(
            concern=Concern.CORRECTNESS,
            status=Status.PROTECTED,
            mechanism=f"zero-knowledge verification ({self.scheme})",
            detail={"verifier_configured": True},
        )

    def check(
        self,
        inputs: Sequence[float],
        expected: Sequence[float],
        proof: Any,
    ) -> bool:
        """Verify a proof, returning whether the output is genuine.

        Args:
            inputs: the plaintext inputs, held locally.
            expected: the plaintext result, held locally.
            proof: whatever the prover emitted, passed through unchanged.

        Returns:
            ``True`` if the proof verifies. ``False`` on a failed check, so the
            caller decides whether that raises.

        Raises:
            MissingDependency: when no verifier is configured, because silently
                returning ``False`` would read as a detected forgery.
        """
        if self.verify is None:
            from ..errors import MissingDependency

            raise MissingDependency(self.available() or "no verifier configured")
        return bool(self.verify(inputs, expected, proof))

    def state(self) -> dict:
        """Serialisable session state, holding no key material."""
        return {"scheme": self.scheme, "verifier_configured": self.verify is not None}
