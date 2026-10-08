"""The posture where the host is not involved: everything stays in-process.

Not a no-op. Working locally is the strongest answer available for scripts, data,
and operations simultaneously, because there is no second party to leak to. Any
comparison of "secure computation" against "just run it here" has to start from
this, because a local run is genuinely protected and is free.

It is not a general solution, which is the reason this is a backend rather than
the default: the machine still has to belong to someone, and a container does
not stop that machine's owner. What it does establish is the correct baseline,
so that a session's report can distinguish "protected because the data never
left" from "protected because of cryptography".
"""

from __future__ import annotations

from typing import Optional

from ..concerns import Backend, Claim, Concern, Status


class LocalBackend(Backend):
    """Local execution, with no remote involved.

    Always available: it requires nothing, which is also why it is only a real
    answer when the compute genuinely fits on the machine you own.
    """

    concerns = (
        Concern.SCRIPTS,
        Concern.DATA,
        Concern.OPERATIONS,
    )

    def available(self) -> Optional[str]:
        """Always usable."""
        return None

    def establish(self) -> Claim:
        """Return one claim per concern this backend covers."""
        raise NotImplementedError(
            "LocalBackend contributes several claims; call claims() instead"
        )

    def claims(self):
        """Yield the per-concern claims this posture supports.

        Returns:
            One :class:`~exlex.concerns.Claim` per covered concern. Scripts and
            data are protected because they never leave; operations is
            protected for the same reason, which is the one claim a remote
            computation cannot make.
        """
        yield Claim(
            concern=Concern.SCRIPTS,
            status=Status.PROTECTED,
            mechanism="local execution",
            detail={"remote": False},
        )
        yield Claim(
            concern=Concern.DATA,
            status=Status.PROTECTED,
            mechanism="local execution",
            detail={"remote": False},
        )
        yield Claim(
            concern=Concern.OPERATIONS,
            status=Status.PROTECTED,
            mechanism="local execution",
            detail={"remote": False},
        )
