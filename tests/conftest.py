"""Shared fixtures.

The key trick here is :class:`FakeTensor`, a minimal stand-in for
``torch.Tensor`` with the same method surface exlex instruments. It lets the
residency guard be tested end to end, including the monkeypatch and restore
logic, on a machine with no GPU and possibly no torch at all.
"""

from __future__ import annotations

import types
from typing import Any, List

import pytest


class FakeTensor:
    """A tensor-like object that records whether it was copied to host."""

    def __init__(self, value: Any = 0.0, is_cuda: bool = True) -> None:
        self._value = value
        self.is_cuda = is_cuda
        self.transfers: List[str] = []
        self.secret: bool = False

    def _mark(self, name: str) -> "FakeTensor":
        self.transfers.append(name)
        return self

    def cpu(self) -> "FakeTensor":
        self._mark("cpu")
        return FakeTensor(self._value, is_cuda=False)

    def numpy(self) -> list:
        self._mark("numpy")
        return [self._value]

    def item(self) -> Any:
        self._mark("item")
        return self._value

    def tolist(self) -> list:
        self._mark("tolist")
        return [self._value]

    def to(self, *args: Any, **kwargs: Any) -> "FakeTensor":
        self._mark("to")
        target = args[0] if args else kwargs.get("device", "cpu")
        name = target if isinstance(target, str) else str(getattr(target, "type", "?"))
        return FakeTensor(self._value, is_cuda=not name.startswith("cpu"))

    def tobytes(self) -> bytes:
        self._mark("tobytes")
        return repr(self._value).encode()

    def pin_memory(self) -> "FakeTensor":
        self._mark("pin_memory")
        return self

    def share_memory_(self) -> "FakeTensor":
        self._mark("share_memory_")
        return self

    def __array__(self) -> list:
        self._mark("__array__")
        return [self._value]

    def __dlpack__(self):
        self._mark("__dlpack__")
        return object()

    def __dlpack_device__(self):
        self._mark("__dlpack_device__")
        return (1, 0)

    def untyped_storage(self):
        self._mark("untyped_storage")
        return object()

    def storage(self):
        self._mark("storage")
        return object()

    def __repr__(self) -> str:
        return f"FakeTensor({self._value!r}, is_cuda={self.is_cuda})"

    def cuda(self, *_args: Any, **_kwargs: Any) -> "FakeTensor":
        self.is_cuda = True
        return self

    def sum(self) -> "FakeTensor":
        return FakeTensor(self._value, is_cuda=self.is_cuda)


def make_fake_torch() -> types.ModuleType:
    """Build a stand-in module exposing just Tensor, matching the torch surface.

    A plain function rather than only a fixture, so other test modules can call
    it directly without triggering pytest's fixture machinery.
    """
    module = types.ModuleType("fake_torch")
    module.Tensor = FakeTensor
    return module


@pytest.fixture
def fake_torch() -> types.ModuleType:
    """A stand-in module exposing just Tensor, matching the torch surface used."""
    return make_fake_torch()


@pytest.fixture
def guard(fake_torch):
    """A guard bound to the fake torch, installed and torn down per test."""
    from exlex.residency import ResidencyGuard

    instance = ResidencyGuard(torch_module=fake_torch)
    instance.install()
    yield instance
    instance.uninstall()


@pytest.fixture
def clean_torch_class(fake_torch):
    """Ensure FakeTensor methods are pristine, in case a test patched them."""
    yield fake_torch.Tensor
    for name, attr in list(vars(FakeTensor).items()):
        if callable(attr) and getattr(attr, "_exlex_wrapped", False):
            setattr(FakeTensor, name, getattr(attr, "_exlex_original", attr))
