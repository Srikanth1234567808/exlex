"""Residency guard: the control that keeps secret tensors off the host."""

from __future__ import annotations

import pytest

from exlex.errors import MissingDependency, ResidencyViolation
from exlex.residency import ResidencyGuard, _is_device_tensor

from conftest import FakeTensor


def test_blocks_cpu_transfer_inside_scope(guard):
    tensor = FakeTensor(3.0, is_cuda=True)
    with guard.scope():
        with pytest.raises(ResidencyViolation) as excinfo:
            tensor.cpu()
    assert "Tensor.cpu" in str(excinfo.value)


def test_allows_transfer_outside_scope(guard):
    tensor = FakeTensor(3.0, is_cuda=True)
    tensor.cpu()
    assert tensor.transfers == ["cpu"]


def test_to_cpu_blocked_but_to_cuda_allowed(guard):
    tensor = FakeTensor(1.0, is_cuda=True)
    with guard.scope():
        with pytest.raises(ResidencyViolation):
            tensor.to("cpu")
        moved = tensor.to("cuda:0")
    assert moved.is_cuda is True


def test_to_without_argument_is_treated_as_suspicious(guard):
    tensor = FakeTensor(1.0, is_cuda=True)
    with guard.scope():
        with pytest.raises(ResidencyViolation):
            tensor.to()


def test_host_tensor_untouched_inside_scope(guard):
    tensor = FakeTensor(1.0, is_cuda=False)
    with guard.scope():
        tensor.item()
    assert tensor.transfers == ["item"]


def test_allow_scope_permits_with_justification(guard):
    tensor = FakeTensor(2.0, is_cuda=True)
    with guard.scope():
        with guard.allow("aggregate metric for the dashboard"):
            tensor.item()
    assert tensor.transfers == ["item"]
    assert "aggregate metric for the dashboard" in guard.summary()["allow_scopes"]


def test_allow_requires_justification(guard):
    with pytest.raises(ValueError, match="non-empty justification"):
        with guard.allow("   "):
            pass


def test_strict_ignores_allow_scopes(fake_torch):
    strict = ResidencyGuard(torch_module=fake_torch, strict=True)
    strict.install()
    try:
        tensor = FakeTensor(1.0, is_cuda=True)
        with strict.scope():
            with strict.allow("should not matter"):
                with pytest.raises(ResidencyViolation):
                    tensor.cpu()
    finally:
        strict.uninstall()


def test_log_mode_records_instead_of_raising(fake_torch, capsys):
    logger = ResidencyGuard(torch_module=fake_torch, on_violation="log")
    logger.install()
    try:
        tensor = FakeTensor(1.0, is_cuda=True)
        with logger.scope():
            tensor.cpu()
        assert logger.violations and logger.violations[0]["method"] == "Tensor.cpu"
        assert "allowed by policy" in capsys.readouterr().err
    finally:
        logger.uninstall()


def test_uninstall_restores_originals(fake_torch):
    pristine = fake_torch.Tensor.cpu
    instance = ResidencyGuard(torch_module=fake_torch)
    instance.install()
    assert fake_torch.Tensor.cpu is not pristine
    instance.uninstall()
    assert fake_torch.Tensor.cpu is pristine


def test_install_is_idempotent(fake_torch):
    pristine = fake_torch.Tensor.cpu
    instance = ResidencyGuard(torch_module=fake_torch)
    instance.install()
    once = fake_torch.Tensor.cpu
    instance.install()
    assert fake_torch.Tensor.cpu is once
    instance.uninstall()
    assert fake_torch.Tensor.cpu is pristine


def test_scope_depth_restores_on_exception(guard):
    with pytest.raises(ValueError):
        with guard.scope():
            raise ValueError("boom")
    assert guard.active is False
    FakeTensor(1.0).cpu()


def test_explicit_taint_marks_host_objects(fake_torch):
    instance = ResidencyGuard(
        torch_module=fake_torch, taint_predicate=lambda obj: False
    )
    instance.install()
    try:
        tensor = FakeTensor(1.0, is_cuda=False)
        instance.taint(tensor)
        with instance.scope():
            with pytest.raises(ResidencyViolation):
                tensor.item()
    finally:
        instance.uninstall()


def test_custom_taint_predicate(fake_torch):
    """A predicate replacing the default must still catch tainted objects."""
    instance = ResidencyGuard(
        torch_module=fake_torch, taint_predicate=lambda obj: getattr(obj, "secret", False)
    )
    instance.install()
    try:
        tensor = FakeTensor(1.0, is_cuda=False)
        tensor.secret = True
        with instance.scope():
            with pytest.raises(ResidencyViolation):
                tensor.item()
    finally:
        instance.uninstall()


def test_missing_torch_raises_clear_error():
    """With no torch available and no override, the error names the dependency."""
    with pytest.raises(MissingDependency, match="PyTorch"):
        ResidencyGuard(torch_module=None).install()


def test_injected_module_without_tensor_rejected():
    class Empty:
        pass

    with pytest.raises(MissingDependency, match="Tensor"):
        ResidencyGuard(torch_module=Empty()).install()


def test_scope_as_decorator(guard):
    @guard.scope()
    def compute():
        return FakeTensor(5.0).cpu()

    with pytest.raises(ResidencyViolation):
        compute()


def test_scope_decorator_runs_body_inside_the_region(guard):
    @guard.scope()
    def compute():
        return FakeTensor(5.0)

    tensor = compute()
    # The scope closed when compute returned, so a later transfer is allowed.
    tensor.cpu()
    assert tensor.transfers == ["cpu"]


def test_nested_scopes_unwind_together(guard):
    tensor = FakeTensor(1.0, is_cuda=True)
    with guard.scope():
        with guard.scope():
            with pytest.raises(ResidencyViolation):
                tensor.numpy()
    # Both scopes have exited, so the guard is no longer active.
    tensor.numpy()


def test_untaint_clears_the_explicit_mark_only(fake_torch):
    """Documented behaviour: the default predicate still protects device tensors.

    ``untaint`` is not an escape hatch. It removes the explicit mark added by
    :meth:`ResidencyGuard.taint`, but a tensor on a GPU still matches the default
    taint predicate, so it stays blocked. Use ``allow(...)`` for a scoped,
    justified bypass.
    """
    instance = ResidencyGuard(torch_module=fake_torch)
    instance.install()
    try:
        tensor = FakeTensor(1.0, is_cuda=True)
        instance.taint(tensor)
        instance.untaint(tensor)
        with instance.scope():
            with pytest.raises(ResidencyViolation):
                tensor.numpy()
    finally:
        instance.uninstall()


def test_violation_records_callsite(guard):
    tensor = FakeTensor(1.0, is_cuda=True)
    with guard.scope():
        with pytest.raises(ResidencyViolation) as excinfo:
            tensor.cpu()
    assert isinstance(excinfo.value.where, int)
    assert excinfo.value.where > 0


def test_violation_callsite_points_at_the_offending_line(guard):
    """The reported line must be the caller's, or the error is unactionable.

    Guards against a wrong ``skip`` in ``_callsite``. The failure mode is
    pointing into residency.py or at the frame above, which a weaker assertion
    such as ``where > 0`` would happily accept.
    """
    import inspect

    source_lines, start = inspect.getsourcelines(
        test_violation_callsite_points_at_the_offending_line
    )
    # The line that performs the leaking call, found by matching the call and
    # not the surrounding comment.
    offset = next(
        i
        for i, line in enumerate(source_lines)
        if line.strip().startswith("tensor.cpu()")
    )
    expected = start + offset

    tensor = FakeTensor(1.0, is_cuda=True)
    reported = -1
    with guard.scope():
        try:
            tensor.cpu()
        except ResidencyViolation as exc:
            reported = exc.where

    assert reported == expected, (
        f"reported line {reported} is not the offending line {expected}"
    )


def test_zero_copy_dlpack_export_is_blocked(guard):
    """A DLPack consumer can pull a buffer sharing VRAM with no copy involved."""
    tensor = FakeTensor(1.0, is_cuda=True)
    with guard.scope():
        with pytest.raises(ResidencyViolation):
            tensor.__dlpack__()


def test_untyped_storage_is_blocked(guard):
    """A storage handle bypasses every other tensor method."""
    tensor = FakeTensor(1.0, is_cuda=True)
    with guard.scope():
        with pytest.raises(ResidencyViolation):
            tensor.untyped_storage()


def test_torch_save_is_blocked_in_secret_region(fake_torch):
    """torch.save is a module function, so tensor methods cannot reach it.

    This is the vector that wrote model weights to disk in plaintext, and it was
    declared in SERIALISERS without ever being enforced.
    """
    written = []
    fake_torch.save = lambda obj, *a, **k: written.append(obj)
    guard = ResidencyGuard(torch_module=fake_torch)
    guard.install()
    try:
        tensor = FakeTensor(1.0, is_cuda=True)
        with guard.scope():
            with pytest.raises(ResidencyViolation) as excinfo:
                fake_torch.save(tensor)
        assert written == [], "the checkpoint must not be written"
        assert "Tensor" not in excinfo.value.method
    finally:
        guard.uninstall()


def test_torch_save_allowed_outside_secret_region(fake_torch):
    written = []
    original = getattr(fake_torch, "save", None)
    fake_torch.save = lambda obj, *a, **k: written.append(obj)
    guard = ResidencyGuard(torch_module=fake_torch)
    guard.install()
    try:
        fake_torch.save(FakeTensor(1.0, is_cuda=True))
        assert len(written) == 1
    finally:
        guard.uninstall()
        if original is not None:
            fake_torch.save = original


def test_torch_save_allowable_with_justification(fake_torch):
    written = []
    fake_torch.save = lambda obj, *a, **k: written.append(obj)
    guard = ResidencyGuard(torch_module=fake_torch)
    guard.install()
    try:
        with guard.scope():
            with guard.allow("release checkpoint is encrypted at rest"):
                fake_torch.save(FakeTensor(1.0, is_cuda=True))
        assert len(written) == 1
    finally:
        guard.uninstall()


def test_uninstall_restores_serialiser_functions(fake_torch):
    original = lambda obj, *a, **k: None  # noqa: E731
    fake_torch.save = original
    guard = ResidencyGuard(torch_module=fake_torch)
    guard.install()
    assert fake_torch.save is not original
    guard.uninstall()
    assert fake_torch.save is original


def test_coverage_admits_its_limits(guard):
    coverage = guard.coverage()
    assert coverage["not_enforced"]
    joined = " ".join(coverage["not_enforced"]).lower()
    assert "compiled" in joined and "side channel" in joined


def test_is_device_tensor_predicate():
    assert _is_device_tensor(FakeTensor(1.0, is_cuda=True)) is True
    assert _is_device_tensor(FakeTensor(1.0, is_cuda=False)) is False
    assert _is_device_tensor(None) is False
    assert _is_device_tensor("string") is False
