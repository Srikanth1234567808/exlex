"""Attestation, with a deliberate bias toward refusing to continue.

Three properties matter more than convenience here:

1. **Fail closed.** A missing tool, an unparsable quote, an unreachable
   revocation endpoint, or an unexpected measurement are all failures. It is
   easy to write attestation code that passes when it cannot tell, and that
   inversion is the whole vulnerability.
2. **You pin, not us.** The caller supplies the expected measurement, ideally
   one they obtained out of band. A quote is only as good as the reference
   value it is compared against, so :meth:`Attestation.verify` takes it from
   the caller and never defaults to "whatever the machine said".
3. **Root of trust is visible.** The residual trust in the CPU vendor's
   firmware signing key is reported rather than buried.

The verification of the certificate chain and signature is delegated to the
vendor tooling (``snpguest``, Intel ``tdx_attestation``) rather than
reimplemented here. A hand-rolled parser of a quote format is a liability, and
the community should not depend on one to make a security claim.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import AttestationError


@dataclass
class Attestation:
    """The result of one attestation attempt.

    Attributes:
        platform: ``"sev-snp"``, ``"tdx"``, or ``"none"``.
        ok: True only if every enabled check passed.
        backend: the tool that produced the quote, for attribution.
        measurements: the launch measurements observed, for pinning.
        checks: per-field check outcomes.
        reasons: why ``ok`` is False, in plain language.
        residual: what remains true even when ``ok`` is True. Never empty.
    """

    platform: str = "none"
    ok: bool = False
    backend: str = "none"
    measurements: Dict[str, str] = field(default_factory=dict)
    checks: Dict[str, bool] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    residual: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "ok": self.ok,
            "backend": self.backend,
            "measurements": dict(self.measurements),
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
            "residual": list(self.residual),
        }

    def __bool__(self) -> bool:
        return self.ok


_RESIDUAL_BASE = (
    "Attestation roots in the CPU vendor's firmware signing key, which is a "
    "single point of trust larger than any cloud operator.",
    "GPU VRAM is not covered. A driver-level attacker on a passed-through "
    "device can DMA secret data; attestation does not prevent this.",
    "Workload shape (model size, VRAM footprint, duration) stays visible to "
    "the provider for billing, and no control here changes that.",
    "Microarchitectural side channels are out of scope: cache, timing, and "
    "port contention.",
)


def _run(cmd: Sequence[str], timeout: float = 30.0) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            list(cmd), capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


class SNPAttestor:
    """SEV-SNP attestation via the ``snpguest`` CLI.

    Args:
        binary: override the tool path, mainly for tests.
        runner: callable taking ``cmd`` and returning ``(rc, out, err)``.
    """

    name = "sev-snp"

    def __init__(self, binary: str = "snpguest-attestation", runner=None) -> None:
        self._binary = binary
        self._run = runner or _run

    def available(self) -> bool:
        """Whether the backend exists. Absence is a failure, not a pass."""
        if shutil.which(self._binary) or _path_exists(self._binary):
            return True
        return self._run([self._binary, "--help"])[0] == 0

    def report(self) -> dict:
        """Fetch and verify a raw attestation report as JSON."""
        if not self.available():
            raise AttestationError(
                f"{self._binary} not found. Install the snpguest tooling; "
                "exlex will not pretend a missing attestor means 'verified'."
            )
        rc, out, err = self._run([self._binary, "report", "--output", "json"])
        if rc != 0:
            raise AttestationError(f"snpguest report failed (rc={rc}): {err or out}")
        try:
            return json.loads(out)
        except json.JSONDecodeError as exc:
            raise AttestationError(f"unparsable attestation report: {exc}") from exc

    def attest(self, pin: Optional[Dict[str, str]] = None) -> Attestation:
        """Run a full check.

        Args:
            pin: expected measurement values, e.g.
                ``{"mr_td": "<hex>", "host_data": "<hex>"}``. Compared with
                constant-time equality. When empty, the machine's own values are
                recorded but nothing is pinned, and that is reported.
        """
        result = Attestation(platform=self.name, backend=self._binary)
        result.residual = list(_RESIDUAL_BASE)

        try:
            data = self.report()
        except AttestationError as exc:
            result.reasons.append(str(exc))
            return result

        result.measurements = _extract_measurements(data)
        result.checks["report_retrieved"] = True

        chain = data.get("certificate_chain")
        result.checks["chain_verified"] = bool(chain) and _chain_complete(chain)
        if not result.checks["chain_verified"]:
            result.reasons.append(
                "certificate chain absent or incomplete; a status field alone is "
                "not evidence that the quote was verified, so this is a failure"
            )

        backend_status = data.get("verifier_status")
        result.checks["backend_ok"] = backend_status in (None, "valid", "verified", True)
        if not result.checks["backend_ok"]:
            result.reasons.append(f"backend reported status {backend_status!r}")

        revoked = _revocation_state(data)
        result.checks["not_revoked"] = revoked is not False
        if revoked is False:
            result.reasons.append("attestation report is revoked")

        result.checks["measurement_pinned"] = _check_pin(result.measurements, pin or {})
        if not result.checks["measurement_pinned"]:
            if pin:
                result.reasons.append("launch measurement does not match the pinned value")
            else:
                result.reasons.append(
                    "no pinned measurement: the machine's own values were recorded, "
                    "so this proves the TEE works but not that it is the machine "
                    "you intended. Pass pin=... with an out-of-band value."
                )

        result.ok = all(result.checks.values())
        return result


class TDXAttestor:
    """Intel TDX attestation via ``tdx_attestation``."""

    name = "tdx"

    def __init__(self, binary: str = "tdx_attestation", runner=None) -> None:
        self._binary = binary
        self._run = runner or _run

    def available(self) -> bool:
        if shutil.which(self._binary) or _path_exists(self._binary):
            return True
        return self._run([self._binary, "--help"])[0] == 0

    def report(self) -> dict:
        if not self.available():
            raise AttestationError(
                f"{self._binary} not found. Install Intel TDX attestation tooling."
            )
        rc, out, err = self._run([self._binary, "get_quote", "--report_raw", "-"])
        if rc != 0:
            raise AttestationError(f"tdx_attestation failed (rc={rc}): {err or out}")
        try:
            payload = json.loads(out)
        except json.JSONDecodeError as exc:
            raise AttestationError(f"unparsable TDX quote: {exc}") from exc
        return payload.get("quote", payload) if isinstance(payload, dict) else {}

    def attest(self, pin: Optional[Dict[str, str]] = None) -> Attestation:
        result = Attestation(platform=self.name, backend=self._binary)
        result.residual = list(_RESIDUAL_BASE) + [
            "TDX side channels include the internal RTE/VRBT channels documented "
            "by Intel, which are not mitigated by this library.",
        ]
        try:
            data = self.report()
        except AttestationError as exc:
            result.reasons.append(str(exc))
            return result

        result.measurements = {
            "mrtd": str(data.get("mrtd", "")).lower(),
            "rt_mr": str(data.get("rt_mr", "")).lower(),
        }
        result.checks["report_retrieved"] = True
        result.checks["quote_signature"] = bool(data.get("signature") or data.get("status"))
        if not result.checks["quote_signature"]:
            result.reasons.append("TDX quote signature not validated by the backend")
        result.checks["measurement_pinned"] = _check_pin(result.measurements, pin or {})
        if not result.checks["measurement_pinned"]:
            result.reasons.append(
                "TDX measurement is not pinned to an out-of-band value"
                if pin
                else "no pinned TDX measurement supplied"
            )
        result.ok = all(result.checks.values())
        return result


def _chain_complete(chain: Any) -> bool:
    """True if a certificate chain looks structurally usable.

    A report that omits the chain, or carries only a leaf, has not been anchored
    to the vendor root and proves nothing on its own.
    """
    if not isinstance(chain, (list, tuple)) or not chain:
        return False
    return len(chain) >= 2


def _path_exists(path: str) -> bool:
    from pathlib import Path  # noqa: PLC0415

    return Path(path).exists()


def _constant_time_eq(a: str, b: str) -> bool:
    """Compare measurements without an early-exit timing signal."""
    import hmac  # noqa: PLC0415

    return hmac.compare_digest(str(a).lower(), str(b).lower())


def _check_pin(observed: Dict[str, str], pin: Dict[str, str]) -> bool:
    if not pin:
        return False
    for key, expected in pin.items():
        actual = observed.get(key, "")
        if not actual or not _constant_time_eq(actual, expected):
            return False
    return True


def _extract_measurements(data: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key in ("mr_td", "mr_ld", "host_data", "report_data", "chip_id"):
        value = data.get(key)
        if isinstance(value, str) and value:
            out[key] = value.lower()
    return out


def _revocation_state(data: Dict[str, Any]) -> Optional[bool]:
    """Read revocation status. Unknown is not the same as not-revoked."""
    for key in ("verifier_status", "revoked", "is_revoked"):
        if key in data:
            value = data[key]
            if isinstance(value, bool):
                return not value if key == "revoked" else value
            if isinstance(value, str):
                lowered = value.lower()
                if lowered in ("true", "revoked"):
                    return False
                if lowered in ("false", "not_revoked", "valid"):
                    return True
    return None


def detect(
    attestors: Optional[Sequence] = None,
    pin: Optional[Dict[str, str]] = None,
) -> Attestation:
    """Try each backend in order and return the first passing result.

    Args:
        attestors: backends to try, in order. Defaults to SNP then TDX.
        pin: expected launch measurements, forwarded to each backend. Note that
            without a pin no backend can report success, by design: a quote
            proves the TEE is real, not that it is the machine you intended.

    Returns:
        The first passing :class:`Attestation`, or a failed one describing why.
        Never raises, so callers can report a reason instead of a traceback.
    """
    backends = list(attestors) if attestors is not None else [SNPAttestor(), TDXAttestor()]
    pin = dict(pin or {})
    last = Attestation()
    for backend in backends:
        candidate = backend.attest(pin=pin)
        if candidate.ok:
            return candidate
        last = candidate
    if not last.reasons:
        last.reasons.append("no attestation backend available on this host")
    return last


def require(
    result: Optional[Attestation] = None,
    pin: Optional[Dict[str, str]] = None,
) -> Attestation:
    """Return a passing attestation or raise.

    Args:
        result: reuse a prior result instead of re-attesting.
        pin: expected measurements, forwarded when re-attesting.
    """
    result = result if result is not None else detect(pin=pin)
    if not result.ok:
        detail = "\n".join(f"  - {r}" for r in result.reasons)
        raise AttestationError(f"attestation failed:\n{detail}")
    return result
