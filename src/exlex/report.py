"""Composing backends into one honest answer.

The point of this module is a report that can be trusted, which means it is
built to be pessimistic. Three rules do the work:

1. A concern no backend claims is :attr:`~exlex.concerns.Status.UNPROTECTED`,
   never absent. Silence would read as coverage.
2. The weakest claim for a concern wins. Two backends cannot average into
   better protection, so a ``PARTIAL`` alongside a ``PROTECTED`` reports
   ``PARTIAL``, and the residual from the weaker one is carried forward.
3. Standing residuals are appended rather than selected, because they apply
   regardless of which backends ran. Model shape leaking under FHE is not a
   property of a particular backend, it is a property of sending a circuit to
   someone else.

The result is deliberately boring to read. A caller should be able to look at
one line per concern and know exactly what they have.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .concerns import (
    HOPELESS,
    REPORT_ORDER,
    Backend,
    Claim,
    Concern,
    Status,
)
from .backends.local import LocalBackend

#: Ordering used when two backends disagree, weakest first. Composition picks
#: the minimum, so this must be ordered from weakest to strongest.
_STATUS_RANK = {
    Status.UNPROTECTED: 0,
    Status.UNAVAILABLE: 1,
    Status.PARTIAL: 2,
    Status.PROTECTED: 3,
}

#: Residuals that hold for any remote computation, regardless of mechanism. The
#: host has to run the code to return a result, and running code is observable.
REMOTE_RESIDUALS: Tuple[str, ...] = (
    "The host learns the circuit structure: operation sequence, layer count, and "
    "every tensor shape. That is often enough to identify a model family.",
    "Tensor sizes, byte counts, operation counts, timing, and GPU utilisation "
    "are visible. No backend in this package hides metadata, and none can.",
    "The host can withhold a result, return it slowly, or return it at all.",
)


@dataclass
class ProtectionReport:
    """What is protected, per concern, and what is not.

    Attributes:
        claims: the winning claim for each concern, keyed by concern.
        remote: whether a remote host is involved. False means everything ran
            locally, which changes several residuals.
        unaddressed: concerns no backend covered, for the caveat line.
    """

    claims: Dict[Concern, Claim] = field(default_factory=dict)
    remote: bool = False
    unaddressed: Tuple[Concern, ...] = ()

    def status(self, concern: Concern) -> Status:
        """The composed status for ``concern``, defaulting to unprotected."""
        claim = self.claims.get(concern)
        return claim.status if claim is not None else Status.UNPROTECTED

    def is_protected(self, concern: Concern) -> bool:
        """Whether ``concern`` is fully protected, not merely raised."""
        return self.status(concern) is Status.PROTECTED

    def to_dict(self) -> dict:
        return {
            "remote": self.remote,
            "claims": {c.value: claim.to_dict() for c, claim in self.claims.items()},
            "unaddressed": [c.value for c in self.unaddressed],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def residuals(self) -> List[str]:
        """Every residual, deduplicated, in report order."""
        seen: Dict[str, None] = {}
        for concern in REPORT_ORDER:
            claim = self.claims.get(concern)
            if claim is not None:
                for item in claim.residual:
                    seen.setdefault(item, None)
            if concern in HOPELESS:
                seen.setdefault(
                    "Metadata is not protected in any configuration this library "
                    "offers: sizes, shapes, timing, and duration stay visible to "
                    "the host.",
                    None,
                )
        if self.remote:
            for item in REMOTE_RESIDUALS:
                seen.setdefault(item, None)
        return list(seen)

    def explain(self) -> str:
        """A short human summary, one line per concern."""
        lines = [
            f"exlex protection report ({'remote' if self.remote else 'local'}):"
        ]
        for concern in REPORT_ORDER:
            claim = self.claims.get(concern)
            status = claim.status.value if claim else Status.UNPROTECTED.value
            mechanism = f"  {claim.mechanism}" if claim and claim.mechanism else ""
            lines.append(f"  {status:<12} {concern.value}{mechanism}")
            if claim is not None and claim.residual:
                for item in claim.residual:
                    lines.append(f"               - {item}")
            elif claim is None:
                lines.append("               - no backend in this session covers this")
        if self.remote:
            lines.append("  any remote computation, regardless of mechanism:")
            for item in REMOTE_RESIDUALS:
                lines.append(f"    - {item}")
        return "\n".join(lines)


def _collect(backend: Backend) -> List[Claim]:
    """Normalise a backend's output into a list of claims.

    A backend may return one claim or several: a concern-keyed mechanism like
    :class:`~exlex.backends.local.FHEBackend` has one, while a posture covering
    several concerns has several. Both are legitimate, so both are accepted.
    """
    if isinstance(backend, LocalBackend):
        return list(backend.claims())
    single = backend.establish()
    if isinstance(single, (list, tuple)):
        return list(single)
    return [single]


def compose(
    backends: Sequence[Backend],
    *,
    remote: bool = True,
) -> ProtectionReport:
    """Build a :class:`ProtectionReport` from a set of backends.

    Args:
        backends: mechanisms to run, in any order. Unavailable ones are
            recorded as unavailable rather than raised, so the report still
            describes the host accurately.
        remote: whether a remote host is involved. Adds the standing remote
            residuals.

    Returns:
        The composed report. A concern with no claim is reported unprotected.

    Raises:
        MissingDependency: never, by design. Availability problems are data.
    """
    per_concern: Dict[Concern, Claim] = {}

    for backend in backends:
        covered = backend.covers()
        if not covered:
            # Nothing to establish, so nothing to ask. A backend that declares
            # no concerns may legitimately leave the protocol methods abstract.
            continue
        reason = backend.available()
        if reason is not None:
            for concern in covered:
                candidate = Claim(
                    concern=concern,
                    status=Status.UNAVAILABLE,
                    residual=(reason,),
                )
                _merge(per_concern, candidate)
            continue
        for claim in _collect(backend):
            _merge(per_concern, claim)

    unaddressed = tuple(c for c in REPORT_ORDER if c not in per_concern)
    return ProtectionReport(claims=per_concern, remote=remote, unaddressed=unaddressed)


def _merge(into: Dict[Concern, Claim], candidate: Claim) -> None:
    """Keep the weakest claim per concern, carrying its residual forward.

    Protection does not average. If any backend leaves a concern partially open,
    the concern is partially open, and a reader needs the weaker backend's reason
    rather than the stronger one's.
    """
    current = into.get(candidate.concern)
    if current is None:
        into[candidate.concern] = candidate
        return
    if _STATUS_RANK[candidate.status] >= _STATUS_RANK[current.status]:
        return
    merged_residual = current.residual + tuple(
        item for item in candidate.residual if item not in current.residual
    )
    into[candidate.concern] = Claim(
        concern=candidate.concern,
        status=candidate.status,
        mechanism=candidate.mechanism or current.mechanism,
        residual=merged_residual,
        detail=candidate.detail or current.detail,
    )
