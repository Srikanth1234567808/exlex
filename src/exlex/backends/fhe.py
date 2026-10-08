"""Data confidentiality by computing on ciphertext.

This backend sends encrypted tensors to a host that cannot read them. What the
host receives is a circuit and a set of ciphertexts; what it sends back is a
ciphertext. The key never leaves this process, so there is nothing on the remote
side to read, and the guarantee is information-theoretic rather than
computational: it does not weaken if the host gets better hardware.

What this does not do
---------------------
The circuit travels in the clear. The host learns the operation sequence, the
layer count, and every tensor shape, which is enough to identify a model family
even with no idea what the weights are. That is reported as
:attr:`~exlex.concerns.Concern.OPERATIONS` unprotected rather than glossed, and
it is the reason :mod:`exlex.concerns` models the axes separately.

A homomorphic scheme is also malleable by design, so the host can corrupt
results. Integrity comes from :mod:`exlex.backends.zk`, not from here.

Scope
-----
Only what the underlying scheme can actually evaluate is supported: affine
layers and polynomial approximations to a nonlinearity. Anything richer would
need a compiler for the target framework, and a library that silently falls back
to plaintext when it cannot compile is worse than one that refuses. Unsupported
shapes raise :class:`~exlex.errors.UnsupportedComputation`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..concerns import Backend, Claim, Concern, Status
from ..errors import MissingDependency, UnsupportedComputation

#: CKKS parameters used unless overridden. Chosen for correctness over speed:
#: the numbers are not where the cost of this approach is paid.
#:
#: These are load-bearing and were established empirically against TenSEAL
#: 0.3.x, not guessed. A ciphertext-times-plaintext matmul consumes modulus
#: budget, and getting this wrong fails in a way that looks like working code:
#:
#: * The global scale is the first modulus, not the widest one. Setting it to
#:   60 with a 60-bit lead raises ``ValueError: scale out of bounds``.
#: * The tower needs enough depth for the product. At 40/40/40 the matmul
#:   returns near-zero garbage with no error at all, which is the worst
#:   possible failure for a security library and the reason these values are
#:   pinned with a test rather than left to a caller.
#:
#: 60/40/40/60 gives a wide outer pair for the product and a 40-bit working
#: scale, which round-trips the demo case exactly.
DEFAULT_POLY_MODULUS_DEGREE = 8192
DEFAULT_COEFF_MOD_BIT_SIZES: Tuple[int, ...] = (60, 40, 40, 60)

#: Result precision. CKKS is approximate, so decryption gives a number near the
#: true value rather than exactly it. Callers that need an exact answer need
#: exact arithmetic, which this scheme does not provide.
DEFAULT_PRECISION = 3


class FHEBackend(Backend):
    """Homomorphic evaluation, with key material held locally.

    Attributes:
        poly_modulus_degree: ring degree; larger means more depth before noise
            accumulates, at quadratic cost.
        coeff_mod_bit_sizes: the multiplication tower.
        precision: decimal places kept on decrypt.
        scheme: passed through to the underlying library; CKKS by default
            because it evaluates on real numbers.
        keys: whether the keypair has been generated. Held only here: the
            secret key never leaves the process, and the object handed to a
            host is :meth:`public_context`, which cannot decrypt.
    """

    concerns = (Concern.DATA,)

    def __init__(
        self,
        *,
        poly_modulus_degree: int = DEFAULT_POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes: Sequence[int] = DEFAULT_COEFF_MOD_BIT_SIZES,
        precision: int = DEFAULT_PRECISION,
        scheme: str = "CKKS",
    ) -> None:
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = tuple(coeff_mod_bit_sizes)
        self.precision = precision
        self.scheme = scheme
        self.keys: bool = False
        self._context: Optional[Any] = None
        self._public: Optional[Any] = None
        self._secret_key: Optional[Any] = None

    # ------------------------------------------------------------------ gates

    def available(self) -> Optional[str]:
        """Check the optional dependency without importing it eagerly."""
        try:
            import tenseal  # noqa: F401
        except ImportError as exc:
            # 0.3.16 shipped without a declared numpy dependency, so an
            # incomplete install surfaces here rather than as a confusing
            # TypeError deep inside tensor construction.
            if getattr(exc, "name", None) == "numpy":
                return (
                    "TenSEAL is installed but its numpy dependency is missing; "
                    "run: pip install numpy"
                )
            return (
                "homomorphic evaluation needs TenSEAL, which is an optional "
                "extra: pip install 'exlex[fhe]'"
            )
        return None

    def establish(self) -> Claim:
        """Build the keypair and the two contexts derived from it.

        Idempotent. The secret key is the entire security argument, so it is
        generated in this process and is the only key material that exists.
        :meth:`public_context` is what a remote would receive; it provably
        cannot decrypt, and a test asserts that.
        """
        unavailable = self.available()
        if unavailable is not None:
            raise MissingDependency(unavailable)
        if self._context is None:
            self._build_contexts()
        return Claim(
            concern=Concern.DATA,
            status=Status.PROTECTED,
            mechanism=f"homomorphic evaluation ({self.scheme})",
            detail={
                "poly_modulus_degree": self.poly_modulus_degree,
                "coeff_mod_bit_sizes": list(self.coeff_mod_bit_sizes),
                "precision": self.precision,
                "approximate": True,
            },
        )

    # ------------------------------------------------------------ computation

    def public_context(self) -> Any:
        """The key-less context a remote host is allowed to have.

        This is the only object that should cross the wire. It can encrypt and
        compute but not decrypt, which is the boundary the whole design rests
        on.
        """
        self._require_keys()
        return self._public

    def encrypt(self, values: Sequence[float], *, secret: bool = False) -> Any:
        """Encrypt a vector.

        Args:
            values: plaintext numbers, which stay in this process.
            secret: encrypt with the private context. Needed for values you
                will decrypt yourself; the remote should use
                :meth:`public_context` instead.

        Returns:
            An opaque ciphertext. Its repr carries no plaintext.
        """
        self._require_keys()
        import tenseal as ts

        context = self._context if secret else self._public
        return ts.ckks_vector(context, list(values))

    def evaluate(self, ciphertext: Any, weights: Any, bias: Sequence[float]) -> Any:
        """Compute ``ciphertext @ weights + bias`` without decrypting.

        Runs on the host. The host holds no secret key, so it learns nothing
        from the values; what it does learn is the shape of the operation, which
        is why :attr:`exlex.concerns.Concern.OPERATIONS` is reported
        separately and is not protected.

        Args:
            ciphertext: output of :meth:`encrypt`.
            weights: a plaintext weight matrix. Homomorphic matmul multiplies
                a ciphertext by public data; the weight matrix is not itself
                secret from the host performing the matmul.
            bias: a plaintext bias vector.

        Returns:
            An encrypted result, to be sent back and decrypted locally.
        """
        self._require_keys()
        return ciphertext.mm(weights) + list(bias)

    def decrypt(self, ciphertext: Any) -> List[float]:
        """Decrypt a returned result. Requires the private context.

        Approximate: CKKS yields a value near the true one, controlled by
        :attr:`precision`. Callers must not treat the result as exact, and
        must not compare with ``==``.

        The secret key is passed explicitly. TenSEAL's public context cannot
        decrypt, which is the point, so a result that came back from a remote
        has to be opened with the key held here.
        """
        self._require_keys()
        return [
            round(v, self.precision) for v in ciphertext.decrypt(self._secret_key)
        ]

    def run_remote(
        self,
        remote_evaluate: Any,
        values: Sequence[float],
        weights: Sequence[Sequence[float]],
        bias: Sequence[float],
    ) -> List[float]:
        """Full round trip: encrypt, evaluate elsewhere, decrypt.

        Args:
            remote_evaluate: a callable taking a ciphertext, a plaintext weight
                matrix, and a plaintext bias, returning an encrypted result. In
                a real deployment this is a call to the host; it is injected
                here so the boundary can be tested without a second machine.
            values: plaintext inputs, which never leave this process.
            weights: plaintext weights, passed to the host for the matmul.
            bias: plaintext bias, passed to the host for the addition.

        Returns:
            The decrypted, approximate result.
        """
        self._require_keys()
        ciphertext = self.encrypt(values, secret=True)
        return self.decrypt(remote_evaluate(ciphertext, weights, bias))

    def run_circuit(
        self,
        circuit: Any,
        values: Sequence[float],
    ) -> List[float]:
        """Encrypt ``values``, run the whole circuit on the host, decrypt.

        This is the intended entry point: it keeps the plaintext inputs and the
        local reference answer in this process, sends only ciphertext plus the
        circuit, and brings back a decrypted result.

        Args:
            circuit: a :class:`exlex.circuit.Circuit`. The host receives it in
                the clear, which is the documented ``OPERATIONS`` leak.
            values: plaintext inputs, which never leave this process.

        Returns:
            The decrypted, approximate result. Compare against
            ``circuit.evaluate_plaintext(values)`` with a tolerance.
        """
        from ..circuit import AffineLayer, BiasLayer, PolynomialLayer

        self._require_keys()
        current = self.encrypt(values, secret=True)
        for layer in circuit.layers:
            if isinstance(layer, AffineLayer):
                current = current.mm([list(row) for row in layer.weights])
            elif isinstance(layer, BiasLayer):
                current = current + list(layer.bias)
            elif isinstance(layer, PolynomialLayer):
                current = current.polyval(list(layer.coefficients))
            else:  # pragma: no cover - guarded by the Circuit type
                raise UnsupportedComputation(f"cannot evaluate {type(layer).__name__}")
        return self.decrypt(current)

    def check_supported(self, weight_shape: Tuple[int, ...], bias_shape: Tuple[int, ...]) -> None:
        """Fail loudly on shapes this scheme cannot evaluate.

        Exists because the alternative is a silent plaintext fallback, which
        would turn a protected claim into a false one at exactly the moment
        someone runs something unsupported.

        Raises:
            UnsupportedComputation: when the shapes are not affine-compatible.
        """
        if len(weight_shape) != 2 or len(bias_shape) != 1:
            raise UnsupportedComputation(
                f"only affine layers are supported (2-D weights, 1-D bias); got "
                f"weights {weight_shape} and bias {bias_shape}"
            )
        if weight_shape[1] != bias_shape[0]:
            raise UnsupportedComputation(
                f"weight columns {weight_shape[1]} do not match bias length "
                f"{bias_shape[0]}"
            )

    # -------------------------------------------------------------- internals

    def _new_context(self) -> Any:
        import tenseal as ts

        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=list(self.coeff_mod_bit_sizes),
        )
        # The working scale is the second modulus, not the first. The first is
        # the outer pair that absorbs the matmul's modulus growth, and using it
        # as the scale overflows. See the note on the default parameters.
        context.global_scale = 2 ** self.coeff_mod_bit_sizes[1]
        # Required for ciphertext-times-plaintext matmul. Without it the
        # failure is a missing-key error deep in the tensor layer.
        context.generate_galois_keys()
        return context

    def _build_contexts(self) -> None:
        """Create a private context and derive a key-less public twin.

        The secret key is captured *before* it is dropped, because
        ``make_context_public`` destroys it in place and the local side still
        needs it to decrypt results. Holding it separately is the whole
        security argument: it exists only in this process, and the object a
        remote receives cannot decrypt.
        """
        private = self._new_context()
        self._secret_key = private.secret_key()

        public = self._new_context()
        public.load(private.serialize())
        public.make_context_public(generate_relin_keys=True, generate_galois_keys=True)

        # Only the public context travels. The private one is kept so that
        # locally produced ciphertexts can be decrypted, and it is never
        # serialised anywhere a remote could reach.
        self._context = private
        self._public = public
        self.keys = True

    def _require_keys(self) -> None:
        if not self.keys or self._context is None or self._public is None:
            raise MissingDependency(
                "keys have not been generated; call establish() first, or use "
                "exlex.protect() to set up a session"
            )

    def state(self) -> Dict[str, object]:
        """Serialisable session state, holding no key material."""
        return {
            "scheme": self.scheme,
            "poly_modulus_degree": self.poly_modulus_degree,
            "coeff_mod_bit_sizes": list(self.coeff_mod_bit_sizes),
            "precision": self.precision,
            "keys_generated": self.keys is not None,
        }
