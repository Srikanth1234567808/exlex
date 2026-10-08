"""Control of the boundary where secret data would cross into host RAM.

A tensor that leaves the GPU is a tensor the host can read, so the boundary is
the security event. This module makes that crossing a deliberate, visible act
instead of an accident.

The guard is deliberately narrow. It instruments the tensor methods that move
data out of device memory and refuses them while a secret region is open. It
does not attempt to police what the GPU does internally, and it cannot see
through a compiled extension or a subprocess. Those limits are documented on
:func:`ResidencyGuard.coverage` and are surfaced in reports.

Design constraints this satisfies:

* Fail closed. A blocked call raises :class:`~exlex.errors.ResidencyViolation`.
* Opt-out is explicit and named. ``with exlex.allow("reason")`` is a visible,
  greppable statement that a reviewer can challenge.
* No implicit monkeypatching at import time. Patching happens only when a guard
  is installed, and :meth:`ResidencyGuard.uninstall` restores the originals.
* Testable without a GPU. The torch module is injectable.
"""

from __future__ import annotations

import contextlib
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .errors import MissingDependency, ResidencyViolation

#: Tensor methods that copy data out of device memory. The guard replaces each
#: of these on the live torch class while a guard is installed.
#:
#: ``__dlpack__`` and ``untyped_storage`` are here because they are zero-copy
#: escapes: a consumer can pull a buffer that shares VRAM with no tensor method
#: involved, so a method allowlist that omits them is trivially bypassed.
TRANSFER_METHODS: Tuple[str, ...] = (
    "cpu",
    "numpy",
    "item",
    "tolist",
    "to",
    "tobytes",
    "pin_memory",
    "share_memory_",
    "__array__",
    "__repr__",
    "__dlpack__",
    "__dlpack_device__",
    "untyped_storage",
    "storage",
)

#: Methods whose return value is always device-resident, so they are never a
#: boundary crossing. ``to`` is deliberately absent: it is handled by inspecting
#: the target device, since ``t.to("cuda")`` is safe and ``t.to("cpu")`` is not.
SAFE_DEVICE_METHODS = frozenset({"detach", "clone", "contiguous", "cuda", "t"})

#: Module-level functions that serialise a secret object graph to disk or to a
#: stream. These are free functions on the torch module, so instrumenting tensor
#: methods cannot reach them.
SERIALISERS: Tuple[str, ...] = ("save", "savez", "to_pickle")


def _load_torch(override: Optional[Any] = None) -> Any:
    """Return a torch-like module, or raise a clear error.

    Args:
        override: a module to use instead of importing torch. Tests pass a fake
            here; production code leaves it ``None``.
    """
    if override is not None:
        return override
    try:
        import torch  # noqa: PLC0415 - deferred so import cost is opt-in
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise MissingDependency(
            "exlex.residency needs PyTorch. Install torch, or pass "
            "torch_module=... to inject a compatible module."
        ) from exc
    return torch


def _is_device_tensor(obj: Any) -> bool:
    """Best-effort test for 'a tensor living on an accelerator'."""
    if obj is None:
        return False
    if getattr(obj, "is_cuda", False):
        return True
    # Other accelerators (XPU, NPU, ROCm reports is_cuda too on some builds).
    return any(bool(getattr(obj, attr, False)) for attr in ("is_xpu", "is_npu", "is_mps"))


def _mentions_host_device(args: Sequence[Any], kwargs: Dict[str, Any]) -> bool:
    """True if a ``.to(...)`` call may target host memory.

    A call with no device argument at all is treated as a boundary crossing.
    Torch returns ``self`` in that case, so this is stricter than the runtime
    semantics require, but it matches what :mod:`exlex.audit` flags and erring
    toward refusing is the right side to err on here.
    """
    candidates = list(args[:2]) + [kwargs.get("device")] + list(kwargs.values())
    if not any(c is not None for c in candidates):
        return True
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, str):
            if candidate.split(":")[0] in ("cpu", ""):
                return True
        elif candidate.__class__.__name__ == "device":
            if str(getattr(candidate, "type", "")).lower() == "cpu":
                return True
    return False


@dataclass
class _Taint:
    """Set of object ids currently considered secret, plus the region depth."""

    ids: set = field(default_factory=set)
    depth: int = 0
    notes: List[str] = field(default_factory=list)

    def add(self, obj: Any) -> None:
        if isinstance(obj, (int, str, bytes, bool, float, complex, type(None))):
            return
        self.ids.add(id(obj))

    def contains(self, obj: Any) -> bool:
        return id(obj) in self.ids

    def discard(self, obj: Any) -> None:
        self.ids.discard(id(obj))


class ResidencyGuard:
    """Blocks secret data from crossing into host memory.

    Instantiate one per process; the default singleton is :data:`_GUARD`.

    Args:
        torch_module: torch-compatible module. Injectable for tests.
        taint_predicate: given an object, return True if it holds secret data.
            The default treats every device tensor as secret, which is the safe
            reading of "unknown".
        on_violation: ``"raise"`` (default) or ``"log"``. ``"log"`` records and
            continues, and is reported in the session summary so a downgrade is
            never invisible.
        strict: when True, ``allow()`` scopes are ignored and every transfer is
            refused. Useful for CI.
    """

    def __init__(
        self,
        torch_module: Optional[Any] = None,
        taint_predicate: Optional[Callable[[Any], bool]] = None,
        on_violation: str = "raise",
        strict: bool = False,
    ) -> None:
        if on_violation not in ("raise", "log"):
            raise ValueError("on_violation must be 'raise' or 'log'")
        self._torch = torch_module
        self._predicate = taint_predicate or _is_device_tensor
        self._on_violation = on_violation
        self._strict = strict
        self._local = threading.local()
        self._taint = _Taint()
        self._originals: Dict[Tuple[Any, str], Any] = {}
        self._installed = False
        self.violations: List[dict] = []

    # ------------------------------------------------------------------ state

    @property
    def installed(self) -> bool:
        """Whether instrumentation is currently active on the torch class."""
        return self._installed

    @property
    def active(self) -> bool:
        """Whether a secret region is open on the current thread."""
        return getattr(self._local, "depth", 0) > 0 or self._taint.depth > 0

    @property
    def allow_depth(self) -> int:
        """Nesting depth of open ``allow()`` scopes on this thread."""
        return getattr(self._local, "allow", 0)

    def _allowing(self) -> bool:
        return bool(getattr(self._local, "allow", 0)) and not self._strict

    # ---------------------------------------------------------------- install

    def install(self) -> "ResidencyGuard":
        """Instrument the torch tensor class. Idempotent.

        Returns:
            self, so this can be chained.
        """
        if self._installed:
            return self
        torch = _load_torch(self._torch)
        tensor_cls = getattr(torch, "Tensor", None)
        if tensor_cls is None:
            raise MissingDependency("the injected module has no Tensor attribute")

        for name in TRANSFER_METHODS:
            original = getattr(tensor_cls, name, None)
            if original is None or not callable(original):
                continue
            key = (tensor_cls, name)
            if key in self._originals:
                continue
            self._originals[key] = original
            setattr(tensor_cls, name, self._wrap(name, original))

        self._torch_module = torch
        self._installed = True
        self._wrap_serialisers()
        return self

    def _wrap_serialisers(self) -> None:
        """Instrument the module-level functions that write secrets to disk.

        ``torch.save`` and friends are free functions, not tensor methods, so the
        tensor method list cannot reach them. A guard that claims to keep secrets
        in device memory but happily writes them to a checkpoint file is worse
        than no guard, because the user believes otherwise.
        """
        torch = self._torch_module
        for name in SERIALISERS:
            original = getattr(torch, name, None)
            if original is None or not callable(original):
                continue
            key = (torch, name)
            if key in self._originals:
                continue
            self._originals[key] = original
            setattr(torch, name, self._wrap_module_fn(name, original))

    def _wrap_module_fn(self, name: str, original: Callable) -> Callable:
        guard = self

        def wrapper(*args, **kwargs):
            if guard.active and not guard._allowing():
                for arg in list(args) + list(kwargs.values()):
                    if guard._taint.contains(arg) or _is_device_tensor(arg):
                        guard._handle_violation(arg, name, surface="module")
            return original(*args, **kwargs)

        wrapper.__name__ = f"exlex_guarded_{name}"
        wrapper._exlex_wrapped = True
        wrapper._exlex_original = original
        return wrapper

    def uninstall(self) -> "ResidencyGuard":
        """Restore the original tensor methods exactly.

        Restoring unconditionally, rather than only on clean exit, means a
        failed guard cannot leave a process permanently patched.
        """
        for (owner, name), original in self._originals.items():
            setattr(owner, name, original)
        self._originals.clear()
        self._installed = False
        return self

    def __enter__(self) -> "ResidencyGuard":
        return self.install()

    def __exit__(self, *exc_info) -> None:
        self.uninstall()

    # ------------------------------------------------------------- instrument

    def _wrap(self, name: str, original: Callable) -> Callable:
        """Build the instrumented method.

        The fast path is hoisted into the closure rather than delegated to a
        method, because this runs on every tensor transfer in a training loop.
        Reading thread-local state is a dict lookup, so the common case (no open
        region) costs one attribute read and a comparison.
        """
        guard = self
        local = self._local
        is_to = name == "to"

        def wrapper(self_tensor, *args, **kwargs):
            depth = getattr(local, "depth", 0)
            if not depth and not guard._taint.depth:
                return original(self_tensor, *args, **kwargs)
            if getattr(local, "allow", 0) and not guard._strict:
                return original(self_tensor, *args, **kwargs)
            taint = guard._taint
            if id(self_tensor) not in taint.ids and not guard._predicate(self_tensor):
                return original(self_tensor, *args, **kwargs)
            if is_to:
                if not _mentions_host_device(args, kwargs):
                    return original(self_tensor, *args, **kwargs)
            elif name in SAFE_DEVICE_METHODS:
                return original(self_tensor, *args, **kwargs)
            guard._handle_violation(self_tensor, name)
            return original(self_tensor, *args, **kwargs)

        wrapper.__name__ = f"exlex_guarded_{name}"
        wrapper.__doc__ = f"exlex-guarded wrapper around Tensor.{name}."
        wrapper._exlex_wrapped = True
        wrapper._exlex_original = original
        return wrapper

    def _should_block(self, obj: Any, name: str, args: Sequence, kwargs: Dict) -> bool:
        """Reference implementation of the decision, kept for testability.

        :meth:`_wrap` inlines this for speed; the two must agree, and
        ``test_wrapper_matches_should_block`` checks that they do.
        """
        if not self.active or self._allowing():
            return False
        if not self._taint.contains(obj) and not self._predicate(obj):
            return False
        if name in SAFE_DEVICE_METHODS:
            return False
        if name == "to":
            return _mentions_host_device(args, kwargs)
        return True

    def _handle_violation(self, obj: Any, name: str, surface: str = "method") -> None:
        # _callsite starts at its own caller, so from here: 0 = this method,
        # 1 = the tensor wrapper, 2 = the line of user code that leaked.
        where = _callsite(skip=2)
        record = {
            "method": name if surface == "module" else f"Tensor.{name}",
            "type": type(obj).__name__,
            "lineno": where,
            "policy": "log" if self._on_violation == "log" else "raise",
        }
        self.violations.append(record)
        if self._on_violation == "raise":
            raise ResidencyViolation(record["method"], where)
        _emit(record)

    # ----------------------------------------------------------------- scopes

    @contextlib.contextmanager
    def scope(self, fn: Optional[Callable] = None):
        """Open a secret region for the duration of the block.

        Usable directly::

            with guard.scope():
                loss = model(batch)

        or as a decorator on ``fn``.
        """
        self._taint.depth += 1
        previous = getattr(self._local, "depth", 0)
        self._local.depth = previous + 1
        try:
            yield self
        finally:
            self._local.depth = previous
            self._taint.depth -= 1

    @contextlib.contextmanager
    def allow(self, reason: str):
        """Temporarily permit host transfers.

        Args:
            reason: recorded in the session report. An empty or whitespace-only
                reason is a programming error and raises, because an unexplained
                bypass defeats the point of the guard.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("exlex.allow requires a non-empty justification string")
        previous = getattr(self._local, "allow", 0)
        self._local.allow = previous + 1
        self._taint.notes.append(reason.strip())
        try:
            yield self
        finally:
            self._local.allow = previous

    # ------------------------------------------------------------- helpers

    def taint(self, *objs: Any) -> None:
        """Explicitly mark objects as secret, beyond the default predicate."""
        for obj in objs:
            self._taint.add(obj)

    def untaint(self, *objs: Any) -> None:
        """Remove the explicit secret mark from ``objs``.

        This is not a bypass. It drops the mark set by :meth:`taint`, but the
        taint predicate still applies, so a device tensor remains blocked. For a
        deliberate, scoped, and recorded bypass use ``allow(reason)``.
        """
        for obj in objs:
            self._taint.discard(obj)

    def zeroize(self, obj: Any) -> None:
        """Best-effort overwrite of a host buffer's contents.

        Only useful on real buffers. The compiler is free to elide writes to a
        dead object, so this raises the cost of recovery without guaranteeing it.
        """
        try:
            import ctypes  # noqa: PLC0415

            data = memoryview(obj).cast("B")
            length = data.nbytes
            ctypes.memset(
                (ctypes.c_char * length).from_buffer(data), 0, length
            )
        except (TypeError, ValueError, BufferError, AttributeError):
            pass

    def coverage(self) -> dict:
        """Describe exactly what is and is not enforced.

        Returned verbatim in session reports, so a user reading a report sees
        the holes alongside the guarantees. The ``not_enforced`` list is derived
        from what ``bench/probe_gaps.py`` actually demonstrates as open, not
        from a guess about what might fail.
        """
        return {
            "enforced": [
                f"Tensor.{name} calls on secret tensors" for name in TRANSFER_METHODS
            ]
            + [f"torch.{name} of a secret tensor" for name in SERIALISERS],
            "not_enforced": [
                "compiled CUDA kernels or custom ops that DMA on their own",
                "torch.load(map_location='cpu'), which lands weights in host RAM "
                "before any tensor method is called",
                "pickling to DataLoader workers or any subprocess: a child process "
                "has its own guard state and none is installed there",
                "data_ptr() and similar address disclosure, which leaks a device "
                "pointer but not tensor contents",
                "data written by the driver into pinned or managed buffers",
                "anything outside a scope() region or without the taint predicate",
                "VRAM contents read directly by a kernel-level attacker",
                "side channels such as timing, cache, or port contention",
            ],
        }

    def summary(self) -> dict:
        """Report violations and allow-scopes for the session record."""
        return {
            "installed": self._installed,
            "violations": list(self.violations),
            "allow_scopes": list(getattr(self._taint, "notes", [])),
        }


_GUARD = ResidencyGuard()


def _callsite(skip: int = 2) -> Optional[int]:
    """Return the 1-based line number ``skip`` frames up the stack.

    Uses direct frame walking rather than :func:`traceback.extract_stack`, which
    materialises a FrameSummary for every frame in the stack. On the refusal
    path that dominated the cost at roughly 27 microseconds per call; walking
    ``f_back`` to the frame we want and stopping is far cheaper and allocates
    nothing. The frames between here and the target are not read at all, so the
    limit is only relevant for very deep stacks.
    """
    try:
        frame = sys._getframe(1)
    except (AttributeError, ValueError):  # pragma: no cover - exotic runtimes
        return None
    for _ in range(skip):
        if frame is None:
            return None
        frame = frame.f_back
    if frame is None:
        return None
    return frame.f_lineno


def _emit(record: dict) -> None:
    """Write a violation to stderr, on one line, without importing logging."""
    detail = record["method"]
    if record.get("lineno"):
        detail = f"{detail} (line {record['lineno']})"
    sys.stderr.write(f"exlex: host transfer allowed by policy: {detail}\n")
