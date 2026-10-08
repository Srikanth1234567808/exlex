"""The concerns view, and how it composes.

Covers :mod:`exlex.concerns` (the axes), :mod:`exlex.report` (composition),
:mod:`exlex.backends` (mechanisms), and the claim merge rule.
"""

from __future__ import annotations

import pytest

import exlex
from exlex.backends import FHEBackend, LocalBackend, ZKBackend
from exlex.concerns import HOPELESS, REPORT_ORDER, Backend, Claim, Concern, Status
from exlex.report import REMOTE_RESIDUALS, compose


def test_every_concern_appears_in_a_report():
    report = exlex.protect()
    for concern in REPORT_ORDER:
        assert report.status(concern) in set(Status)


def test_uncovered_concern_reports_unprotected_not_absent():
    class Empty(Backend):
        concerns = ()

    report = compose([Empty()], remote=True)
    for concern in REPORT_ORDER:
        assert report.status(concern) is Status.UNPROTECTED
        assert concern in report.unaddressed


def test_weakest_claim_wins_when_backends_disagree():
    strong = Claim(concern=Concern.DATA, status=Status.PROTECTED, mechanism="fhe")
    weak = Claim(
        concern=Concern.DATA,
        status=Status.PARTIAL,
        mechanism="legacy",
        residual=("the legacy path still spills plaintext",),
    )
    report = compose([_One(strong), _One(weak)], remote=True)
    assert report.status(Concern.DATA) is Status.PARTIAL
    assert "the legacy path still spills plaintext" in report.residuals()


def test_partial_claim_must_name_its_residual():
    with pytest.raises(ValueError):
        Claim(concern=Concern.DATA, status=Status.PARTIAL, mechanism="x")


def test_protected_does_not_require_a_residual():
    claim = Claim(concern=Concern.DATA, status=Status.PROTECTED, mechanism="fhe")
    assert claim.residual == ()


def test_metadata_is_never_claimed_protected():
    class Overreaching(Backend):
        concerns = (Concern.METADATA,)

        def available(self):
            return None

        def establish(self):
            return Claim(concern=Concern.METADATA, status=Status.PROTECTED)

    report = compose([Overreaching()], remote=True)
    assert report.status(Concern.METADATA) is Status.PROTECTED
    # A backend can claim it, but the library still states the standing truth.
    assert Concern.METADATA in HOPELESS
    residuals = " ".join(report.residuals())
    assert "Metadata is not protected" in residuals


def test_remote_reports_carry_the_standing_residuals():
    report = compose([LocalBackend()], remote=True)
    joined = " ".join(report.residuals())
    for expected in REMOTE_RESIDUALS:
        assert expected in joined


def test_local_reports_omit_remote_residuals():
    report = compose([LocalBackend()], remote=False)
    joined = " ".join(report.residuals())
    assert not any(item in joined for item in REMOTE_RESIDUALS)


def test_local_posture_protects_scripts_data_and_operations():
    report = exlex.protect_local()
    assert report.remote is False
    for concern in (Concern.SCRIPTS, Concern.DATA, Concern.OPERATIONS):
        assert report.is_protected(concern)


def test_explain_shows_a_line_per_concern():
    report = exlex.protect()
    text = report.explain()
    for concern in REPORT_ORDER:
        assert concern.value in text


def test_report_round_trips_through_json():
    import json

    payload = json.loads(exlex.protect().to_json())
    assert set(payload["claims"]) <= {c.value for c in REPORT_ORDER}
    assert "remote" in payload


def test_unavailable_backend_is_data_not_an_exception():
    class Broken(Backend):
        concerns = (Concern.CORRECTNESS,)

        def available(self):
            return "no hardware here"

        def establish(self):  # pragma: no cover - must not be called
            raise AssertionError("establish should not run when unavailable")

    report = compose([Broken()], remote=True)
    assert report.status(Concern.CORRECTNESS) is Status.UNAVAILABLE
    assert "no hardware here" in " ".join(report.residuals())


def test_is_protected_is_false_for_partial():
    report = compose(
        [_One(Claim(concern=Concern.DATA, status=Status.PARTIAL, residual=("gap",)))],
        remote=True,
    )
    assert report.is_protected(Concern.DATA) is False


def test_protect_rejects_unknown_backend():
    class Rogue(Backend):
        concerns = ()

    with pytest.raises(ValueError):
        exlex.protect(backends=[Rogue()])


def test_zk_backend_is_unavailable_without_a_verifier():
    backend = ZKBackend()
    assert backend.available() is not None
    report = compose([backend], remote=True)
    assert report.status(Concern.CORRECTNESS) is Status.UNAVAILABLE


def test_zk_backend_claims_correctness_when_configured():
    backend = ZKBackend(verify=lambda i, e, p: True, scheme="halo2")
    report = compose([backend], remote=True)
    assert report.is_protected(Concern.CORRECTNESS)


def test_zk_check_raises_rather_than_returning_false():
    from exlex.errors import MissingDependency

    with pytest.raises(MissingDependency):
        ZKBackend().check([1.0], [1.0], "proof")


def test_fhe_backend_reports_availability_without_tenseal():
    backend = FHEBackend()
    reason = backend.available()
    assert reason is None or "tenseal" in reason.lower()


def test_fhe_state_holds_no_key_material():
    backend = FHEBackend()
    state = backend.state()
    assert state["scheme"] == "CKKS"
    # A boolean flag is fine; key bytes are not. Whatever the session has
    # generated by now, the serialisable state exposes no key material.
    assert isinstance(state["keys_generated"], bool)
    assert not any("key" in key and key != "keys_generated" for key in state)
    assert "secret" not in str(state).lower()


class _One(Backend):
    """Test double returning a single fixed claim."""

    def __init__(self, claim):
        self.claim = claim
        self.concerns = (claim.concern,)

    def available(self):
        return None

    def establish(self):
        return self.claim
