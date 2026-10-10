"""The public entry point for the concerns view.

:func:`protect` is the counterpart to :func:`exlex.session`. Where a session
answers "how much hardening is active on this host", a protection report answers
the questions a caller actually has when deciding whether to run on a machine
they do not own: is my data confidential, are my scripts confidential, can this
host lie to me, and what is visible regardless.

Both views are kept because they answer different questions and neither implies
the other. A host can be fully hardened and still hand your model to everyone;
a session can report ``attested`` while your data crosses the bus in the clear.

Example::

    import exlex

    report = exlex.protect()
    print(report.explain())

    if not report.is_protected(exlex.Concern.DATA):
        raise SystemExit("refusing to run: data is not confidential on this host")
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from .backends import (
    FHEBackend,
    LocalBackend,
    PepperBackend,
    SlalomBackend,
    VerificationBackend,
    ZKBackend,
)
from .concerns import Concern, Status
from .report import ProtectionReport, compose


def protect(
    backends: Optional[Sequence[Any]] = None,
    *,
    remote: bool = True,
    include_local: bool = False,
    zk_verify: Optional[Any] = None,
) -> ProtectionReport:
    """Run the backends and report what is protected.

    Args:
        backends: mechanisms to run. Defaults to homomorphic evaluation,
            randomised result verification, and an unconfigured zero-knowledge
            verifier, which is the honest starting posture for running on a host
            you do not control.
        remote: whether a remote host is involved. Adds the standing remote
            residuals, which are not specific to any mechanism.
        include_local: also include the local-execution posture. Only makes
            sense when the compute genuinely runs on your own machine, where
            there is no host to leak to.
        zk_verify: a verifier callable, forwarded to
            :class:`~exlex.backends.zk.ZKBackend`. Without it, correctness
            reports the randomised check only, and the residual says a real
            proof is still absent.

    Returns:
        A :class:`~exlex.report.ProtectionReport`. Every concern appears; none
        is ever silently omitted.

    Raises:
        ValueError: on an unrecognised backend type, rather than ignoring it.
    """
    chosen: List[Any] = list(backends) if backends is not None else []
    if backends is None:
        chosen = [FHEBackend(), VerificationBackend(), ZKBackend(verify=zk_verify)]
    else:
        for backend in chosen:
            if not isinstance(
                backend,
                (
                    FHEBackend,
                    ZKBackend,
                    LocalBackend,
                    PepperBackend,
                    SlalomBackend,
                    VerificationBackend,
                ),
            ):
                raise ValueError(
                    f"unsupported backend {type(backend).__name__}; subclass "
                    "exlex.concerns.Backend to add one"
                )
    if include_local:
        chosen.append(LocalBackend())
    return compose(chosen, remote=remote)


def protect_local() -> ProtectionReport:
    """Report for work that runs entirely on your own machine.

    The strongest and cheapest posture available: nothing leaves, so scripts,
    data, and operations are all protected without cryptography. Included as an
    explicit function because "just run it locally" is the honest baseline any
    remote comparison should be measured against.
    """
    return compose([LocalBackend()], remote=False)


__all__ = ["Concern", "Status", "protect", "protect_local"]
