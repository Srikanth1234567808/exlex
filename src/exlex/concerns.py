"""The independent things a workload might need protected.

The level ladder in :mod:`exlex.levels` answers "how much hardening is active".
It cannot answer "is my model confidential on this host", because that is several
unrelated questions and a workload usually cares about a different subset of them
on each host. FHE on a rented 3090 protects data and leaves the model outline
visible; a TEE protects the outline and the data and still leaks timing. Neither
is a rung above the other, so modelling them as one would force a lie in one
direction or the other.

So this module is a second, orthogonal view: the *concerns* are the axes, the
*backends* are the mechanisms, and :mod:`exlex.report` composes them. A concern
that no backend covers reports as such. There is no partial credit and no
ordering between concerns, because there is nothing to order them by.

What no concern here can reach
------------------------------
Every axis in this module is about **content**. None of it touches metadata:
tensor shapes, byte counts, operation counts, timing, and GPU utilisation are
visible to a host in every configuration, because observing that work happened
is unavoidable. Metadata is modelled explicitly as
:attr:`Concern.METADATA` so that it is reported rather than assumed, and so a
caller can see at a glance that no backend in this package protects it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Optional, Tuple


class Concern(str, Enum):
    """One independently answerable question about a workload.

    Members:

    SCRIPTS:
        Does your source code leave your machine in readable form?
    DATA:
        Are inputs and model weights unreadable to the host?
    CORRECTNESS:
        Can the host return a wrong answer and have you believe it?
    OPERATIONS:
        Does the host learn the structure of what you compute, even without
        seeing the values?
    METADATA:
        Can the host learn the size, shape, or timing of the work?

    Every concern is evaluated separately because a backend may answer one and
    not the others. Enumerating them makes an uncovered axis visible instead of
    leaving it to an unwritten assumption.
    """

    SCRIPTS = "scripts"
    DATA = "data"
    CORRECTNESS = "correctness"
    OPERATIONS = "operations"
    METADATA = "metadata"


class Status(str, Enum):
    """How well a concern is covered, in the words the report will use.

    Members:

    PROTECTED:
        The threat is closed by something active in this session.
    PARTIAL:
        Meaningfully raised, with a named gap. Never used to imply safety.
    UNPROTECTED:
        Nothing is protecting this, and that is a real answer, not a failure to
        configure anything. It is what a single host reports for operations.
    UNAVAILABLE:
        A backend for this concern exists but could not run here, usually a
        missing dependency or absent hardware.
    """

    PROTECTED = "protected"
    PARTIAL = "partial"
    UNPROTECTED = "unprotected"
    UNAVAILABLE = "unavailable"


#: Concerns, in the order reports print them. Data and scripts first because
#: they are what people are usually asking about; metadata last because it is
#: the one nobody protects and it should not lead.
REPORT_ORDER: Tuple[Concern, ...] = (
    Concern.SCRIPTS,
    Concern.DATA,
    Concern.CORRECTNESS,
    Concern.OPERATIONS,
    Concern.METADATA,
)

#: Concerns with no protection available in any configuration shipped here.
#: Used by the report to print a standing caveat, and by the tests to assert we
#: never quietly start claiming one of them.
HOPELESS: FrozenSet[Concern] = frozenset({Concern.METADATA})


@dataclass(frozen=True)
class Claim:
    """What one backend says about one concern.

    A backend returns a Claim rather than a boolean so that a partial answer can
    name its own gap. A backend that cannot protect something says
    :attr:`Status.UNPROTECTED` with the reason filled in, instead of returning
    nothing and letting the report infer optimism.

    Attributes:
        concern: the concern this claim addresses.
        status: the verdict, in report language.
        mechanism: what is doing the protecting, for the report's attribution
            line. Empty when nothing is.
        residual: what remains exposed despite this claim. Never empty for
            :attr:`Status.PARTIAL`, because a partial answer without its gap is
            a false summary.
        detail: free-form extras for ``to_dict`` consumers. Backend-specific and
            not interpreted by exlex.
    """

    concern: Concern
    status: Status
    mechanism: str = ""
    residual: Tuple[str, ...] = ()
    detail: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status is Status.PARTIAL and not self.residual:
            raise ValueError(
                f"partial claim on {self.concern.value!r} must name its residual; "
                "a partial answer without its gap is a false summary"
            )

    def to_dict(self) -> dict:
        return {
            "concern": self.concern.value,
            "status": self.status.value,
            "mechanism": self.mechanism,
            "residual": list(self.residual),
            "detail": dict(self.detail),
        }


class Backend:
    """A mechanism that claims one or more concerns.

    Not a ``typing.Protocol``, deliberately: backends are registered by
    subclassing this and are instantiated by the report, so an explicit base
    class gives a single place to document the contract and lets a subclass that
    forgets a method fail at review rather than at the call site.

    A backend must be honest about failure. :meth:`available` returns a reason
    instead of raising, so an absent dependency or missing TEE hardware becomes a
    reportable :attr:`Status.UNAVAILABLE` rather than a traceback. A backend that
    cannot determine its own status must say
    :attr:`Status.UNPROTECTED` and explain, never :attr:`Status.PROTECTED`.
    """

    #: Concerns this backend addresses. Declared by the subclass.
    concerns: Tuple[Concern, ...] = ()

    def available(self) -> Optional[str]:
        """Return ``None`` if usable here, else why not.

        Read-only and side-effect free. Called by the report before anything
        else so that a backend never half-runs.
        """
        raise NotImplementedError

    def establish(self) -> Claim:
        """Activate the mechanism and return what it now protects.

        Only called when :meth:`available` returned ``None``. Must be safe to
        call more than once; the report may run a session twice.
        """
        raise NotImplementedError

    def covers(self) -> Tuple[Concern, ...]:
        """Concerns this backend addresses, in report order."""
        declared = set(self.concerns)
        return tuple(c for c in REPORT_ORDER if c in declared)
