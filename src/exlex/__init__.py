"""exlex: run ML workloads on GPU hosts you do not control.

    import exlex

    with exlex.session(level="resident"):
        @exlex.secret
        def train(model, batch):
            return model(batch).mean()

A note on what this library is. It is defence in depth against a host operator
who is not actively writing a hypervisor exploit: it keeps secrets in device
memory, hardens the guest against routine snooping, and verifies the machine's
identity where a TEE is available. It is not a TEE, it does not encrypt GPU
memory, and it does not make a consumer 3090 or 4090 safe from a determined
attacker. The residual risks at every level are enumerated in
:mod:`exlex.levels` and printed by ``exlex doctor``; read them before relying on
any of this.

What actually reaches a strong guarantee is well documented: SEV-SNP or TDX with
pinned measurements, which excludes hypervisor-level memory reads. Everything
below that is a cost/benefit trade against a specific class of attacker, not a
guarantee.
"""

from __future__ import annotations

from . import audit
from .backends import (
    FHEBackend,
    FixedPoint,
    LocalBackend,
    PepperBackend,
    PinnedWeights,
    SlalomBackend,
    VerificationBackend,
    ZKBackend,
)
from .circuit import AffineLayer, BiasLayer, Circuit, PolynomialLayer, compile_model
from .concerns import (
    REPORT_ORDER,
    Backend,
    Claim,
    Concern,
    Status,
)
from .errors import (
    AggregateError,
    AttestationError,
    AuditError,
    ExlexError,
    HardeningError,
    MissingDependency,
    PolicyError,
    ResidencyViolation,
    UnsupportedComputation,
)
from .levels import Level, NAMES, spec, weakest
from .protect import protect, protect_local
from .report import ProtectionReport
from .residency import ResidencyGuard
from .session import (
    Session,
    SessionReport,
    allow,
    doctor,
    get_guard,
    resident,
    secret,
    session,
    set_guard,
)

__version__ = "0.2.0"

__all__ = [
    "AffineLayer",
    "AggregateError",
    "AttestationError",
    "AuditError",
    "Backend",
    "BiasLayer",
    "Circuit",
    "Claim",
    "Concern",
    "ExlexError",
    "FHEBackend",
    "FixedPoint",
    "HardeningError",
    "Level",
    "LocalBackend",
    "MissingDependency",
    "NAMES",
    "PolicyError",
    "PolynomialLayer",
    "PepperBackend",
    "PinnedWeights",
    "ProtectionReport",
    "REPORT_ORDER",
    "ResidencyGuard",
    "ResidencyViolation",
    "Session",
    "SessionReport",
    "SlalomBackend",
    "Status",
    "UnsupportedComputation",
    "VerificationBackend",
    "ZKBackend",
    "allow",
    "audit",
    "compile_model",
    "doctor",
    "get_guard",
    "protect",
    "protect_local",
    "resident",
    "secret",
    "session",
    "set_guard",
    "spec",
    "weakest",
    "__version__",
]
