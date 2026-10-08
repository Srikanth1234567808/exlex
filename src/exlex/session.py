"""Session: the single entry point that ties the controls together.

The intended shape of a call site::

    import exlex

    @exlex.secret
    def train(model, batch):
        return model(batch).mean()

    with exlex.session(level=exlex.Level.ATTESTED, pin={"mr_td": "..."}):
        train(model, secret_batch)

or, once, at the top of a job::

    exlex.session(level="resident").start()

    @exlex.secret
    def train(...): ...

A session reports the *lowest* level any enabled control actually reached.
Requesting a level never grants it. If attestation is unavailable, the session
says so and drops to what it can honestly support.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from . import attest as _attest
from . import harden as _harden
from .errors import AggregateError, PolicyError
from .levels import Level, NAMES, spec, weakest
from .residency import ResidencyGuard, _GUARD

LevelLike = Union[Level, str, int]


def _coerce_level(value: LevelLike) -> Level:
    """Accept a Level, its name, or an int. Strings are forgiving on purpose."""
    if isinstance(value, Level):
        return value
    if isinstance(value, int):
        return Level(value)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in NAMES.values():
            return next(lvl for lvl, name in NAMES.items() if name == key)
        if key.isdigit():
            return Level(int(key))
    raise PolicyError(
        f"unknown level {value!r}; expected one of {sorted(NAMES.values())}"
    )


@dataclass
class SessionReport:
    """What a session actually achieved, and what it did not.

    Attributes:
        requested: the level asked for.
        achieved: the lowest level any enabled control reached. The ladder is
            cumulative, so a skipped or failed rung caps the result: residency
            without passing hardening reports HARDENED, not RESIDENT.
        controls: per-control status, keyed by control name.
        attestation: the attestation result, when it was attempted.
        hardening: the hardening probe result, when it was attempted.
        residency: the guard's coverage and violation record.
        residual: the combined residual risk of the achieved level.
        downgrade_reason: why ``achieved`` fell short of ``requested``.
    """

    requested: Level
    achieved: Level
    controls: Dict[str, bool] = field(default_factory=dict)
    attestation: Optional[dict] = None
    hardening: Optional[dict] = None
    residency: Optional[dict] = None
    residual: List[str] = field(default_factory=list)
    downgrade_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "requested": NAMES[self.requested],
            "achieved": NAMES[self.achieved],
            "controls": dict(self.controls),
            "attestation": self.attestation,
            "hardening": self.hardening,
            "residency": self.residency,
            "residual": list(self.residual),
            "downgrade_reason": self.downgrade_reason,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def explain(self) -> str:
        """A short human summary, including the downgrade if there was one."""
        lines = [
            f"exlex session: requested={NAMES[self.requested]} "
            f"achieved={NAMES[self.achieved]}",
        ]
        for name, ok in sorted(self.controls.items()):
            lines.append(f"  [{'ok  ' if ok else 'FAIL'}] {name}")
        if self.downgrade_reason:
            lines.append(f"  downgraded: {self.downgrade_reason}")
        if self.residual:
            lines.append("  residual risk at this level:")
            lines.extend(f"    - {r}" for r in self.residual)
        return "\n".join(lines)


class Session:
    """Holds the active configuration and the resulting report.

    Do not instantiate directly; use :func:`exlex.session`.
    """

    def __init__(
        self,
        level: LevelLike = Level.HARDENED,
        *,
        pin: Optional[Dict[str, str]] = None,
        harden: bool = True,
        attest: bool = True,
        guard: Optional[ResidencyGuard] = None,
        on_missing: str = "downgrade",
        require_hardening: bool = False,
        environ: Optional[Dict[str, str]] = None,
    ) -> None:
        if on_missing not in ("downgrade", "raise", "ignore"):
            raise ValueError("on_missing must be 'downgrade', 'raise', or 'ignore'")
        self.level = _coerce_level(level)
        self.pin = dict(pin or {})
        self.do_harden = harden
        self.do_attest = attest
        self.guard = guard or _GUARD
        self.on_missing = on_missing
        self.require_hardening = require_hardening
        self.environ = environ
        self.report: Optional[SessionReport] = None
        self._started = False

    # ------------------------------------------------------------------ API

    def start(self) -> SessionReport:
        """Run the enabled checks, install the guard, and build the report.

        Returns:
            The session report. Also available as ``.report``.

        Raises:
            AggregateError: when ``on_missing="raise"`` and a required control
                for the requested level did not pass.
        """
        if self._started:
            return self.report  # type: ignore[return-value]

        controls: Dict[str, bool] = {}
        failures: List[str] = []
        reached: List[Level] = []
        reasons: List[str] = []

        if self.do_harden and self.level >= Level.HARDENED:
            probe = _harden.probe()
            controls["hardening"] = probe.ok
            if not probe.ok:
                reasons.append(
                    "host hardening incomplete: " + ", ".join(probe.names())
                )
                if self.require_hardening or self.on_missing == "raise":
                    failures.extend(probe.names())
            else:
                reached.append(Level.HARDENED)
            self._hardening = probe
        else:
            self._hardening = None
            controls["hardening"] = False
            reasons.append("hardening was not requested")

        if self.level >= Level.RESIDENT:
            self.guard.install()
            controls["residency"] = True
            # RESIDENT is defined as HARDENED plus residency, so it can only be
            # claimed when hardening actually passed above.
            if controls.get("hardening"):
                reached.append(Level.RESIDENT)
            else:
                reasons.append(
                    "residency is enabled but cannot be claimed without passing "
                    "hardening, since mlock and swap-off underpin it"
                )
        else:
            controls["residency"] = False
            reasons.append("residency was not requested")

        attestation_dict = None
        if self.do_attest and self.level >= Level.ATTESTED:
            result = _attest.detect(pin=self.pin)
            attestation_dict = result.to_dict()
            controls["attestation"] = result.ok
            if result.ok:
                reached.append(Level.ATTESTED)
            else:
                reasons.append(
                    "attestation unavailable or unverified: " + "; ".join(result.reasons)
                )
                if self.on_missing == "raise":
                    failures.append("attestation")
        else:
            controls["attestation"] = False
            reasons.append("attestation was not requested")

        achieved = weakest(reached)
        if achieved < self.level and not reasons:
            reasons.append("no control reached the requested level")

        if failures and self.on_missing == "raise":
            raise AggregateError([PolicyError(f) for f in failures])

        if self.require_hardening and self._hardening is not None and not self._hardening.ok:
            _harden.require(self._hardening)

        if self.environ is not None:
            self._apply_environ()

        self.report = SessionReport(
            requested=self.level,
            achieved=achieved,
            controls=controls,
            attestation=attestation_dict,
            hardening=self._hardening.to_dict() if self._hardening else None,
            residency=self.guard.coverage() | self.guard.summary(),
            residual=list(spec(achieved).residual),
            downgrade_reason="; ".join(reasons) if achieved < self.level else None,
        )
        self._started = True
        return self.report

    def stop(self) -> None:
        """Uninstall the guard. Idempotent, and safe after a failure."""
        self.guard.uninstall()
        self._started = False

    def __enter__(self) -> "Session":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # ------------------------------------------------------------- internals

    def _apply_environ(self) -> None:
        """Set process env vars that stop common libraries from leaking.

        Best effort and non-fatal: a missing knob must not be the reason a job
        refuses to start, so failures here are silent but the session still
        reports the level it can support.
        """
        env = self.environ
        if env is None:
            return
        for key, value in (
            ("PYTORCH_NO_CUDA_MEMORY_CACHING", "1"),
            ("TOKENIZERS_PARALLELISM", "false"),
            ("HF_HUB_DISABLE_TELEMETRY", "1"),
            ("WANDB_MODE", "offline"),
            ("CUDA_MODULE_LOADING", "LAZY"),
        ):
            try:
                env[key] = value
            except (TypeError, AttributeError):
                pass
        try:
            import torch  # noqa: PLC0415

            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def session(
    level: LevelLike = Level.HARDENED,
    **kwargs: Any,
) -> Session:
    """Create a :class:`Session`. The single entry point for the library.

    Args:
        level: desired assurance, as a :class:`Level`, its name, or an int.
        **kwargs: forwarded to :class:`Session`. See its docstring for
            ``pin``, ``on_missing``, ``guard``, and the rest.

    Returns:
        A :class:`Session`, usable as a context manager.
    """
    return Session(level, **kwargs)


def get_guard() -> ResidencyGuard:
    """Return the process-wide guard used by ``@secret``, ``allow``, ``resident``.

    The package attribute ``exlex.session`` is the factory function, not the
    module of the same name, so this is the supported way to reach the guard.
    """
    return _GUARD


def set_guard(guard: ResidencyGuard) -> ResidencyGuard:
    """Replace the process-wide guard.

    Intended for tests and for embedders that own the torch module, such as
    training frameworks with their own patched tensor subclass. Returns the
    previous guard so callers can restore it.
    """
    global _GUARD
    previous = _GUARD
    _GUARD = guard
    return previous


# --------------------------------------------------------------------- API

def secret(fn: Optional[Callable] = None, **kwargs: Any) -> Any:
    """Mark a function as handling secret data, and run it under the guard.

    Usable bare or with arguments::

        @exlex.secret
        def train(model, batch): ...

        @exlex.secret(strict=True)
        def export(model): ...

    Args:
        fn: the function, when used without parentheses.
        **kwargs: forwarded to :func:`exlex.residency.scope`.

    Returns:
        The wrapped function, or a decorator.
    """

    def decorate(target: Callable) -> Callable:
        from functools import wraps  # noqa: PLC0415

        @wraps(target)
        def wrapper(*args: Any, **kwds: Any) -> Any:
            with _GUARD.scope():
                return target(*args, **kwds)

        wrapper.__exlex_secret__ = True  # type: ignore[attr-defined]
        return wrapper

    return decorate(fn) if fn is not None else decorate


@contextlib.contextmanager
def allow(reason: str) -> Iterator[None]:
    """Permit host transfers inside the block, with a recorded justification.

    Args:
        reason: why the transfer is safe. Required and non-empty; an
            unexplained bypass is a bug, not a feature.
    """
    with _GUARD.allow(reason):
        yield


@contextlib.contextmanager
def resident() -> Iterator[None]:
    """Open a secret region without decorating a function."""
    with _GUARD.scope():
        yield


def doctor(*, as_json: bool = False) -> str:
    """Summarise what is active on this host right now.

    Read-only. Safe to run anywhere, including on a laptop, which is the point:
    the output says what a session here would and would not be able to claim.
    """
    probe = _harden.probe()
    result = _attest.detect()
    reached: List[Level] = []
    reasons: List[str] = []
    if probe.ok:
        reached.append(Level.HARDENED)
        reached.append(Level.RESIDENT)
    else:
        reasons.append("host hardening incomplete: " + ", ".join(probe.names()))
    if result.ok:
        reached.append(Level.ATTESTED)
    else:
        reasons.append("attestation: " + "; ".join(result.reasons))

    report = SessionReport(
        requested=Level.ATTESTED,
        achieved=weakest(reached),
        controls={
            "hardening": probe.ok,
            "residency": True,
            "attestation": result.ok,
        },
        attestation=result.to_dict(),
        hardening=probe.to_dict(),
        residency=_GUARD.coverage(),
        residual=list(spec(weakest(reached)).residual),
        downgrade_reason="; ".join(reasons) or None,
    )
    if as_json:
        return report.to_json()
    lines = [report.explain(), "", "GPUs:"]
    for device in _harden.gpu_inventory():
        lines.append(f"  {device}")
    return "\n".join(lines)
