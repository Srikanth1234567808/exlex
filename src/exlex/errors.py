"""Exception hierarchy.

Every failure in exlex is fail-closed: an unverified environment raises rather
than silently degrading. Callers who genuinely want to tolerate a missing
control must opt in explicitly via ``session(..., on_missing="downgrade")``.
"""

from __future__ import annotations

from typing import List, Optional


class ExlexError(Exception):
    """Base class for all exlex errors."""


class MissingDependency(ExlexError):
    """An optional dependency required for the requested feature is absent."""


class ResidencyViolation(ExlexError):
    """Secret data was about to cross the host boundary.

    Attributes:
        method: the blocked call, e.g. ``"Tensor.cpu"``.
        where: 1-based line of the offending call site, when discoverable.
    """

    def __init__(self, method: str, where: Optional[int] = None) -> None:
        self.method = method
        self.where = where
        location = f" at line {where}" if where else ""
        super().__init__(
            f"exlex blocked {method!r}{location}: this would copy secret data out of "
            "GPU-resident memory into host RAM, where the operator can read it. "
            "If the transfer is genuinely safe, wrap it in exlex.allow(...) and add a "
            "justification so reviewers can see it."
        )


class AttestationError(ExlexError):
    """Attestation could not be established.

    This is raised for *all* attestation problems, including tooling absence.
    Absence of an attestor is a failure, not a pass.
    """


class HardeningError(ExlexError):
    """The host environment failed a required hardening check."""


class AuditError(ExlexError):
    """Static analysis found a policy violation in the audited source."""


class PolicyError(ExlexError):
    """The session was asked for a level it cannot currently reach."""


class UnsupportedComputation(ExlexError):
    """The requested computation cannot be evaluated under the active scheme.

    Raised instead of falling back to plaintext. A backend that quietly
    downgrades to an unprotected path turns a ``PROTECTED`` claim into a false
    one at the exact moment someone runs something it cannot handle, which is
    the failure mode this project exists to prevent.
    """


class AggregateError(ExlexError):
    """Several independent checks failed.

    Attributes:
        errors: the individual failures, in discovery order.
    """

    def __init__(self, errors: List[ExlexError]) -> None:
        self.errors = list(errors)
        detail = "\n".join(f"  - {e}" for e in self.errors)
        super().__init__(f"{len(self.errors)} check(s) failed:\n{detail}")
