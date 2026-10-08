"""The FHE path, exercised against real TenSEAL when it is installed.

Skipped rather than failed when the extra is absent, because the library is
designed to work without it and CI should stay cheap. The tests that matter
most are :func:`test_public_context_cannot_decrypt` and
:func:`test_ciphertext_repr_leaks_nothing`: they assert the property the whole
design rests on, rather than that the arithmetic runs.
"""

from __future__ import annotations

import pytest

from exlex.backends import FHEBackend
from exlex.errors import MissingDependency, UnsupportedComputation

pytest.importorskip("tenseal", reason="requires the fhe extra: pip install 'exlex[fhe]'")

VALUES = [1.0, 2.0, 3.0]
WEIGHTS = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
BIAS = [0.5, 0.5]
EXPECTED = [4.5, 5.5]


@pytest.fixture(scope="module")
def backend():
    instance = FHEBackend()
    if instance.available() is not None:
        pytest.skip(instance.available())
    instance.establish()
    return instance


def _remote_mm(ciphertext, weights, bias):
    """A host's entire view: a ciphertext, and public parameters."""
    return ciphertext.mm(weights) + list(bias)


def test_round_trip_matches_plaintext(backend):
    result = backend.run_remote(_remote_mm, VALUES, WEIGHTS, BIAS)
    assert result == pytest.approx(EXPECTED, abs=1e-2)


def test_public_context_cannot_decrypt(backend):
    """The property the design depends on."""
    public = backend.public_context()
    assert public.is_public() is True
    assert public.has_secret_key() is False

    ciphertext = ts_encrypt(backend)
    # TenSEAL raises ValueError when a tensor's context holds no secret key and
    # none is supplied. Asserting the message, not just the type, so that a
    # different failure cannot be mistaken for the protection working.
    with pytest.raises(ValueError, match="secret_key"):
        ciphertext.decrypt()


def test_ciphertext_repr_leaks_nothing(backend):
    ciphertext = ts_encrypt(backend)
    text = repr(ciphertext)
    for value in VALUES:
        assert str(value) not in text


def test_secret_key_is_not_on_the_public_context(backend):
    public = backend.public_context()
    assert not public.has_secret_key()


def test_encrypt_before_establish_raises():
    with pytest.raises(MissingDependency):
        FHEBackend().encrypt([1.0])


def test_establish_is_idempotent(backend):
    before = backend.state()["keys_generated"]
    backend.establish()
    assert backend.state()["keys_generated"] is before is True


def test_state_carries_no_key_material(backend):
    text = str(backend.state()).lower()
    assert "secret" not in text
    assert "private" not in text


def test_unsupported_shapes_raise_rather_than_fall_back(backend):
    backend.check_supported((3, 2), (2,))
    with pytest.raises(UnsupportedComputation):
        backend.check_supported((3,), (2,))
    with pytest.raises(UnsupportedComputation):
        backend.check_supported((3, 2), (5,))


def test_result_is_approximate_not_exact(backend):
    """Documents the precision caveat rather than hiding it.

    A caller comparing with ``==`` should be surprised here, which is the
    intended way to discover the approximation.
    """
    noisy = [0.1, 0.2, 0.3]
    identity = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    result = backend.run_remote(_remote_mm, noisy, identity, [0.0, 0.0])
    assert result == pytest.approx(noisy[:2], abs=1e-2)


def ts_encrypt(backend):
    """Encrypt with the public context, as a host would receive."""
    return backend.encrypt(VALUES)
