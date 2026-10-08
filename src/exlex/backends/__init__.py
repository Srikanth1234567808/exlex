"""Backend implementations, one module per mechanism.

Each backend addresses a different concern set and has different prerequisites:

* :mod:`exlex.backends.fhe` -- data confidentiality via homomorphic encryption.
* :mod:`exlex.backends.zk` -- output correctness verification.
* :mod:`exlex.backends.local` -- the local-compute posture, where scripts and
  data never leave the process at all and the host is not involved.

* :mod:`exlex.backends.slalom` -- blinded delegation of a linear layer, so a
  host does the matmul and sees only a one-time pad, with the result checked
  against a local reference.

A backend that cannot run reports why, rather than raising, so the composed
report stays honest on a laptop, in CI, and on a GPU host. Nothing here installs
anything at import time; the cryptographic extras are resolved on first use.
"""

from __future__ import annotations

from .fhe import FHEBackend
from .local import LocalBackend
from .slalom import FixedPoint, SlalomBackend, host_matmul
from .verify import VerificationBackend
from .zk import ZKBackend

__all__ = [
    "FHEBackend",
    "FixedPoint",
    "LocalBackend",
    "SlalomBackend",
    "VerificationBackend",
    "ZKBackend",
    "host_matmul",
]
