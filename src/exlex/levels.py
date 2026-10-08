"""Security levels and, more importantly, their limits.

Each level states two things: what an adversary can no longer do, and what they
still can. The second list is the one that matters. A level is only meaningful
if its residual risk is understood, so nothing here is described as "secure" in
the absolute sense.

Ordering is total: ``RESIDENT < HARDENED < ATTESTED``, and a session reports the
*lowest* level every enabled control actually reached. Raising a requested level
never raises the achieved one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import FrozenSet, Tuple


class Level(IntEnum):
    """Assurance level, ordered weakest to strongest."""

    NONE = 0
    """No exlex control active. Baseline OS behaviour."""

    HARDENED = 1
    """Host configured to resist routine and opportunistic access."""

    RESIDENT = 2
    """HARDENED, plus secret tensors kept out of host RAM in audited regions."""

    ATTESTED = 3
    """RESIDENT, plus the guest's cryptographic identity verified before use."""


#: Human-readable level name, used in reports and CLI output.
NAMES = {
    Level.NONE: "none",
    Level.HARDENED: "hardened",
    Level.RESIDENT: "resident",
    Level.ATTESTED: "attested",
}


@dataclass(frozen=True)
class LevelSpec:
    """What a level means, stated in the most sceptical terms available.

    Attributes:
        level: the level this describes.
        prevents: actions the adversary can no longer take. Each is a concrete
            action, not an adjective.
        residual: what a motivated adversary retains. Never empty.
        requires: what the level depends on to hold.
    """

    level: Level
    prevents: FrozenSet[str]
    residual: Tuple[str, ...]
    requires: Tuple[str, ...]

    @property
    def name(self) -> str:
        return NAMES[self.level]


def _specs() -> "dict":
    return {
        Level.NONE: LevelSpec(
            level=Level.NONE,
            prevents=frozenset(),
            residual=(
                "Everything a host operator can normally do, including reading "
                "guest RAM with root, dumping VRAM, and reading your disk.",
            ),
            requires=(),
        ),
        Level.HARDENED: LevelSpec(
            level=Level.HARDENED,
            prevents=frozenset(
                {
                    "reading guest RAM via /proc/kcore, guestmem, or a crash dump",
                    "reading secrets from swap, hibernation, or core files",
                    "attaching with ptrace or reading secret memory of another uid",
                    "exfiltration over the network, by blocking egress",
                    "cloud metadata credential theft via IMDS",
                    "cross-tenant access from a shared host",
                    "local log/support staff snooping through installed agents",
                },
            ),
            residual=(
                "A malicious kernel with root still reads guest RAM directly. "
                "Hardening is configuration, not isolation, so it is defeatable "
                "by anyone who can change the boot or kernel parameters.",
                "GPU VRAM is unencrypted on consumer parts and remains readable "
                "by a driver-level attacker.",
                "A privileged operator can observe workload shape: model size, "
                "VRAM footprint, duration, and timing.",
            ),
            requires=(
                "control of the guest boot chain, so an operator who can boot a "
                "different kernel can disable the hardening entirely",
            ),
        ),
        Level.RESIDENT: LevelSpec(
            level=Level.RESIDENT,
            prevents=frozenset(
                {
                    "accidental or incidental host-RAM copies from a library "
                    "the audit missed",
                    "plaintext tensor material in coredumps, swap, or /dev/shm",
                    "round-tripping secrets through .cpu(), .numpy(), or .item()",
                    "serialising secret tensors to disk or to a log",
                },
            ),
            residual=(
                "Residency is enforced in the Python process you run, so it "
                "binds the interpreter, not the GPU. A kernel-level attacker "
                "reads VRAM without asking Python's permission.",
                "Anything that legitimately must produce a host-side output "
                "(metrics, a preview, a checkpoint) still leaks that output.",
                "Coverage is bounded by the audit. Unmarked code is unenforced.",
                "A GPU or driver bug that spills to host memory is not caught.",
            ),
            requires=(
                "every secret-handling function being marked with @exlex.secret, "
                "since unmarked code runs unguarded",
                "a non-hostile kernel, since residency has no teeth against root",
            ),
        ),
        Level.ATTESTED: LevelSpec(
            level=Level.ATTESTED,
            prevents=frozenset(
                {
                    "reading guest DRAM, even with kernel root, because the "
                    "memory is encrypted by the CPU's memory controller",
                    "substituting a different kernel, bootloader, or firmware "
                    "before launch, via measurement pinning",
                    "being quietly downgraded to a platform without a TEE",
                    "rolling back the guest to a previously valid state",
                },
            ),
            residual=(
                "The CPU vendor's firmware signing key. Attestation is rooted in "
                "the vendor root of trust, which is a larger single point of "
                "trust than any cloud operator.",
                "Physical access to the board before the memory controller "
                "initialises, which can read DRAM during a cold boot.",
                "GPU VRAM is still unencrypted. A driver-level attacker with the "
                "passed-through device can DMA your data, and this is the "
                "strongest realistic residual for GPU work.",
                "Microarchitectural side channels: cache, timing, port contention.",
                "Workload shape and billing metadata remain visible. No amount of "
                "attestation changes what the provider can meter.",
                "Supply-chain interdiction of your own hardware or model files.",
            ),
            requires=(
                "SEV-SNP or TDX hardware, and a provider that exposes it",
                "a pinned measurement you chose and verified out of band",
                "your own hardware, or a facility that has never opened the machine",
            ),
        ),
    }


SPECS = _specs()


def spec(level: Level) -> LevelSpec:
    """Return the claim description for ``level``."""
    return SPECS[level]


def weakest(levels) -> Level:
    """Return the lowest level in ``levels``.

    Session reporting uses this so that one strong control cannot mask a weak
    or missing one. An empty iterable yields ``Level.NONE``.
    """
    ordered = sorted(set(levels))
    return ordered[0] if ordered else Level.NONE


def weakest_name(levels) -> str:
    """Return the name of the lowest level in ``levels``."""
    return NAMES[weakest(levels)]
