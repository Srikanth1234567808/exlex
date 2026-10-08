"""Attestation tests. The important ones are the fail-closed cases."""

from __future__ import annotations

import json

import pytest

from exlex.attest import (
    Attestation,
    SNPAttestor,
    TDXAttestor,
    _check_pin,
    _constant_time_eq,
    _revocation_state,
    detect,
    require,
)
from exlex.errors import AttestationError

GOOD_REPORT = {
    "mr_td": "AA" * 32,
    "mr_ld": "BB" * 32,
    "host_data": "CC" * 32,
    "chip_id": "01",
    "certificate_chain": ["leaf", "intermediate", "ark"],
    "verifier_status": "valid",
}


def _runner_for(payload, rc=0, stderr=""):
    def runner(cmd):
        if "--help" in cmd:
            return 0, "usage", ""
        if rc != 0:
            return rc, "", stderr
        return 0, json.dumps(payload), ""

    return runner


def test_snp_success_requires_pin_to_match():
    backend = SNPAttestor(runner=_runner_for(GOOD_REPORT))
    result = backend.attest(pin={"mr_td": GOOD_REPORT["mr_td"]})
    assert result.ok is True
    assert result.checks["measurement_pinned"] is True
    assert result.residual


def test_snp_without_pin_is_a_failure():
    backend = SNPAttestor(runner=_runner_for(GOOD_REPORT))
    result = backend.attest()
    assert result.ok is False
    assert any("no pinned measurement" in r for r in result.reasons)


def test_snp_wrong_pin_fails_closed():
    backend = SNPAttestor(runner=_runner_for(GOOD_REPORT))
    result = backend.attest(pin={"mr_td": "FF" * 32})
    assert result.ok is False
    assert result.checks["measurement_pinned"] is False


def test_missing_backend_is_a_failure_not_a_pass():
    def runner(cmd):
        return 127, "", "not found"

    result = SNPAttestor(runner=runner).attest()
    assert result.ok is False
    assert any("not found" in r or "snpguest" in r for r in result.reasons)


def test_unparsable_report_fails_closed():
    def runner(cmd):
        if "--help" in cmd:
            return 0, "usage", ""
        return 0, "this is not json", ""

    result = SNPAttestor(runner=runner).attest(pin={"mr_td": "AA"})
    assert result.ok is False
    assert any("unparsable" in r for r in result.reasons)


def test_revoked_report_fails():
    payload = dict(GOOD_REPORT, verifier_status="revoked")
    backend = SNPAttestor(runner=_runner_for(payload))
    result = backend.attest(pin={"mr_td": GOOD_REPORT["mr_td"]})
    assert result.ok is False
    assert result.checks["not_revoked"] is False


def test_missing_chain_fails_closed():
    payload = {k: v for k, v in GOOD_REPORT.items() if k != "certificate_chain"}
    backend = SNPAttestor(runner=_runner_for(payload))
    result = backend.attest(pin={"mr_td": GOOD_REPORT["mr_td"]})
    assert result.ok is False
    assert any("chain" in r for r in result.reasons)


def test_require_raises_with_reasons():
    with pytest.raises(AttestationError) as excinfo:
        require(Attestation(reasons=["no tee here"]))
    assert "no tee here" in str(excinfo.value)


def test_detect_falls_through_to_failure():
    def runner(cmd):
        return 127, "", "absent"

    result = detect(attestors=[SNPAttestor(runner=runner), TDXAttestor(runner=runner)])
    assert result.ok is False
    assert result.reasons


def test_detect_returns_first_passing_backend():
    good = SNPAttestor(runner=_runner_for(GOOD_REPORT))

    def broken(cmd):
        return 127, "", "absent"

    result = detect(
        attestors=[TDXAttestor(runner=broken), good],
        pin={"mr_td": GOOD_REPORT["mr_td"]},
    )
    assert result.ok is True
    assert result.platform == "sev-snp"


def test_detect_without_pin_never_succeeds():
    """A quote with nothing to compare against proves nothing, so it fails."""
    good = SNPAttestor(runner=_runner_for(GOOD_REPORT))
    result = detect(attestors=[good])
    assert result.ok is False
    assert any("pinned" in r for r in result.reasons)


def test_pin_comparison_is_case_insensitive():
    assert _check_pin({"mr_td": "AABB"}, {"mr_td": "aabb"}) is True
    assert _check_pin({"mr_td": "AABB"}, {"mr_td": "aabbcc"}) is False
    assert _check_pin({}, {"mr_td": "aabb"}) is False
    assert _check_pin({"mr_td": "AABB"}, {}) is False


def test_constant_time_eq_basic():
    assert _constant_time_eq("AA", "aa") is True
    assert _constant_time_eq("AA", "AB") is False


def test_revocation_state_unknown_is_none():
    assert _revocation_state({}) is None
    assert _revocation_state({"verifier_status": "valid"}) is True
    assert _revocation_state({"revoked": True}) is False
    assert _revocation_state({"revoked": False}) is True


def test_tdx_success_and_residual_mentions_rte():
    payload = {"mrtd": "DD" * 32, "rt_mr": "EE" * 32, "signature": "ab", "status": "ok"}
    backend = TDXAttestor(runner=_runner_for(payload))
    result = backend.attest(pin={"mrtd": "DD" * 32})
    assert result.ok is True
    assert any("RTE" in r or "side channel" in r for r in result.residual)


def test_tdx_without_pin_fails():
    payload = {"mrtd": "DD" * 32, "signature": "ab"}
    result = TDXAttestor(runner=_runner_for(payload)).attest()
    assert result.ok is False


def test_attestation_bool_and_dict_roundtrip():
    result = Attestation(ok=True, platform="sev-snp")
    assert bool(result) is True
    assert result.to_dict()["platform"] == "sev-snp"
    assert Attestation().to_dict()["ok"] is False
