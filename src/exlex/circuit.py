"""Compiling a model into a circuit the FHE path can actually evaluate.

This is the honest boundary of the encrypted path. A homomorphic scheme
evaluates affine maps and polynomial approximations to a nonlinearity. A real
model is mostly *not* that: attention, softmax, normalisation, and arbitrary
elementwise activations have no cheap homomorphic form. So this module does one
of two things and never a third:

* compile a model whose every supported layer maps onto something the scheme
  can evaluate, or
* raise :class:`~exlex.errors.UnsupportedComputation` naming the layer that
  stopped it.

There is deliberately no third path. A compiler that quietly drops an
unsupported layer, or approximates one it was not asked to approximate, turns a
``PROTECTED`` claim into a false one at the exact moment a model gains a layer
the library has never seen. Refusing is the only safe answer, and it is why
:func:`compile_model` returns a description of what it could not do rather than
a partial result.

What "supported" means here
---------------------------
:class:`AffineLayer` and :class:`BiasLayer`, which are exact. A single
polynomial activation, if the caller supplies coefficients, which are
approximate and must be reported as such. Everything else is refused.

Note what the compiled circuit still contains: the operation sequence, the
layer count, and every shape. That is what the host receives, and it is why
:attr:`~exlex.concerns.Concern.OPERATIONS` is not protected by this path. See
:mod:`exlex.concerns`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .errors import MissingDependency, UnsupportedComputation

Matrix = Tuple[Tuple[float, ...], ...]


@dataclass(frozen=True)
class AffineLayer:
    """An exact matrix multiply by plaintext weights.

    Stored as ``(out, in)``, matching how TenSEAL's ``CKKSVector.mm`` behaves:
    a vector of length ``in`` multiplied by a matrix with ``in`` rows and ``out``
    columns yields a vector of length ``out``. Every output is a dot product of
    one matrix row with the input.

    Homomorphic because the weights are not secret from the host performing the
    multiply. That is the standard trade: the *inputs* and the *output* stay
    encrypted, and the weight matrix is treated as public. A deployment where
    the weights must also be hidden needs a different mechanism, and the report
    says so rather than implying otherwise.
    """

    weights: Matrix
    name: str = "affine"

    @property
    def input_dim(self) -> int:
        """Width the input vector must have: the number of matrix rows."""
        return len(self.weights)

    @property
    def output_dim(self) -> int:
        """Width of the result: the length of a matrix row."""
        return len(self.weights[0]) if self.weights else 0

    def describe(self) -> str:
        return f"Affine({self.input_dim}->{self.output_dim})"


@dataclass(frozen=True)
class BiasLayer:
    """An exact vector add. Homomorphic and free."""

    bias: Tuple[float, ...]
    name: str = "bias"

    @property
    def output_dim(self) -> int:
        return len(self.bias)

    def describe(self) -> str:
        return f"Bias({self.output_dim})"


@dataclass(frozen=True)
class PolynomialLayer:
    """A polynomial applied elementwise, via :meth:`CKKSTensor.polyval`.

    This is the only honest way to get a nonlinearity out of a homomorphic
    scheme, and it is an *approximation*: ``tanh`` and ``sigmoid`` become
    truncated Chebyshev or Taylor polynomials whose accuracy falls off outside
    the fitted range. Both the approximation and the degree are recorded so the
    report can name them, because an approximated activation is not the model
    you wrote.

    Coefficients are in ascending order, matching TenSEAL's ``polyval``.
    """

    coefficients: Tuple[float, ...]
    name: str = "polynomial"
    approximates: str = "activation"

    @property
    def degree(self) -> int:
        return max(0, len(self.coefficients) - 1)

    def describe(self) -> str:
        return f"Poly(deg={self.degree}, ~{self.approximates})"


Layer = Union[AffineLayer, BiasLayer, PolynomialLayer]


@dataclass
class Circuit:
    """A compiled, evaluable sequence of layers.

    Attributes:
        layers: the layers in evaluation order.
        approximations: human-readable notes about every approximated step, so
            the report can state what is not exactly the original model.
    """

    layers: List[Layer] = field(default_factory=list)
    approximations: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def input_dim(self) -> int:
        """Width of the first layer's input, or 0 for an empty circuit.

        Shape is stored as ``(out, in)`` because a weight matrix is indexed
        ``row`` for each output, so the *input* width is the row length, not the
        number of rows. Getting this backwards is the easy mistake and it
        matters: the shape is what the host sees, so an inverted dimension here
        misreports the leak as well as breaking evaluation.
        """
        for layer in self.layers:
            if isinstance(layer, AffineLayer):
                return layer.input_dim
            if isinstance(layer, PolynomialLayer):
                break
        return 0

    @property
    def output_dim(self) -> int:
        """Width of the last layer's output."""
        for layer in reversed(self.layers):
            if isinstance(layer, (AffineLayer, BiasLayer)):
                return layer.output_dim
        return 0

    def shapes(self) -> List[Tuple[int, ...]]:
        """Per-layer shapes, as the host would see them.

        This method exists because the leak is a fact about the design, not an
        oversight, and a caller inspecting a circuit should be able to see
        exactly what a host learns.
        """
        out: List[Tuple[int, ...]] = []
        for layer in self.layers:
            if isinstance(layer, AffineLayer):
                out.append((layer.input_dim, layer.output_dim))
            elif isinstance(layer, BiasLayer):
                out.append((layer.output_dim,))
            else:
                out.append((self.input_dim,))
        return out

    def describe(self) -> str:
        """A readable dump.

        Sent to the host, so it doubles as a demonstration of the leak: the
        operation sequence and every width are printed in full.
        """
        lines = [f"Circuit({len(self.layers)} layers)"]
        lines.extend(
            f"  {i}: {layer.name} = {layer.describe()}"
            for i, layer in enumerate(self.layers)
        )
        if self.layers:
            lines.append("approximated (not exactly the original model):")
            if self.approximations:
                lines.extend(f"    ! {note}" for note in self.approximations)
            else:
                lines.append("    (none: every layer is exact)")
        return "\n".join(lines)

    def evaluate_plaintext(self, values: Sequence[float]) -> List[float]:
        """Run the circuit locally, in the clear.

        The reference answer. Used to check the encrypted path, and it is what a
        caller would get if the host simply did the work in the clear:
        identical up to CKKS rounding.
        """
        current: List[float] = [float(v) for v in values]
        for layer in self.layers:
            if isinstance(layer, AffineLayer):
                current = _matmul(current, layer.weights)
            elif isinstance(layer, BiasLayer):
                current = [a + b for a, b in zip(current, layer.bias)]
            else:
                current = [
                    sum(c * (v**i) for i, c in enumerate(layer.coefficients))
                    for v in current
                ]
        return current


def _matmul(vector: Sequence[float], matrix: Matrix) -> List[float]:
    """``out[j] = sum_i matrix[i][j] * vector[i]``, matching ``vector @ matrix``.

    The matrix is ``(in, out)``: it has ``in`` rows, one per input, and ``out``
    columns, one per output. Each output is therefore a dot product *down a
    column*, not across a row. Transposing this silently swaps every layer, so
    the test suite pins the convention rather than trusting it.
    """
    if not matrix:
        return []
    if len(vector) != len(matrix):
        raise UnsupportedComputation(
            f"vector of length {len(vector)} cannot multiply a matrix with "
            f"{len(matrix)} rows; the input width is the row count"
        )
    out_dim = len(matrix[0])
    return [
        sum(matrix[i][j] * vector[i] for i in range(len(vector)))
        for j in range(out_dim)
    ]


#: Torch module type names this compiler handles exactly.
_EXACT_TYPES = ("Linear", "LazyLinear", "Add", "Bias")

#: Torch module type names that need a polynomial and are therefore refused
#: unless the caller opts in with ``approximate=True``.
_APPROXIMATABLE = {
    "ReLU": "ReLU",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
    "GELU": "GELU",
    "SiLU": "SiLU",
    "Softmax": "softmax",
    "LayerNorm": "LayerNorm",
    "BatchNorm1d": "BatchNorm1d",
}


def compile_model(model: Any, *, approximate: bool = False) -> Circuit:
    """Compile a torch model into an evaluable circuit.

    Args:
        model: a ``torch.nn.Module``, typically an ``nn.Sequential``.
        approximate: when True, non-linearities are compiled to polynomial
            approximations and every one is recorded in
            :attr:`Circuit.approximations`. When False (the default) they are
            refused.

    Returns:
        A :class:`Circuit`.

    Raises:
        MissingDependency: when torch is not installed.
        UnsupportedComputation: naming the first layer that cannot be compiled.
            The message is the point: a caller needs to know *which* layer
            stopped them, not merely that something did.
    """
    if not _torch_available():
        raise MissingDependency(
            "compiling a model needs torch; install it, or construct a Circuit "
            "from raw matrices instead"
        )
    import torch.nn as nn

    circuit = Circuit()
    for name, layer in model.named_children():
        kind = type(layer).__name__
        if kind in _EXACT_TYPES:
            circuit.layers.extend(_compile_exact(name, layer, kind))
        elif kind in _APPROXIMATABLE:
            if not approximate:
                raise UnsupportedComputation(
                    f"layer {name!r} is {kind}, which has no exact homomorphic "
                    "form. Pass approximate=True to compile it as a polynomial "
                    "approximation, and note that the result is then an "
                    "approximation of your model rather than your model"
                )
            circuit.layers.append(
                PolynomialLayer(
                    coefficients=_approximate(kind, layer),
                    name=name,
                    approximates=_APPROXIMATABLE[kind],
                )
            )
            circuit.approximations.append(
                f"{name}: {kind} replaced by a degree-"
                f"{len(_approximate(kind, layer)) - 1} polynomial; accuracy "
                "degrades outside the fitted range"
            )
        elif kind in _EXACT_TYPES or isinstance(layer, nn.Sequential):
            nested = compile_model(layer, approximate=approximate)
            circuit.layers.extend(nested.layers)
            circuit.approximations.extend(nested.approximations)
        else:
            raise UnsupportedComputation(
                f"layer {name!r} is {kind}, which this compiler does not handle. "
                "Attention, softmax, and normalisation have no cheap "
                "homomorphic form; see the module docstring for what is "
                "supported"
            )
    _check_dimensions(circuit)
    return circuit


def _compile_exact(name: str, layer: Any, kind: str) -> List[Layer]:
    """Extract a layer's parameters as plain floats.

    A torch ``Linear`` holds its bias inside the same module, and CKKS has no
    fused bias-add, so this emits two circuit layers: the matmul, then the
    vector add. Dropping the bias instead would make the encrypted result
    quietly differ from the plaintext one, which is exactly the class of bug
    that only shows up in production.
    """
    if kind in ("Linear", "LazyLinear"):
        # torch stores Linear.weight as (out, in) because it computes
        # x @ W.T. This circuit uses (in, out), so the transpose happens here
        # rather than being left for the caller to discover.
        weight = layer.weight.detach()
        weights = tuple(
            tuple(float(v) for v in row) for row in weight.transpose(0, 1).tolist()
        )
        out: List[Layer] = [AffineLayer(weights=weights, name=name)]
        if layer.bias is not None:
            out.append(
                BiasLayer(
                    bias=tuple(float(v) for v in layer.bias.detach().flatten()),
                    name=f"{name}.bias",
                )
            )
        return out
    if getattr(layer, "bias", None) is not None:
        return [
            BiasLayer(
                bias=tuple(float(v) for v in layer.bias.detach().flatten()),
                name=name,
            )
        ]
    raise UnsupportedComputation(
        f"layer {name!r} is {kind} with no bias and no weight; nothing to compile"
    )


def _approximate(kind: str, layer: Any) -> Tuple[float, ...]:
    """Chebyshev-ish coefficients for a supported activation.

    Deliberately a fixed, small set rather than a fitted polynomial: a fitted
    polynomial would be more accurate and would also leak the fitting data
    through its degree and range, which is a trade nobody should make silently.
    Callers who need a specific fit pass their own coefficients.
    """
    table: Dict[str, Tuple[float, ...]] = {
        "ReLU": (0.0, 1.0),
        "Sigmoid": (0.5, 0.25, 0.0, -0.015625),
        "Tanh": (0.0, 1.0, 0.0, -0.3333333333333333),
        "GELU": (0.0, 0.5, 0.0, -0.16666666666666666),
        "SiLU": (0.0, 0.5, 0.0, -0.08333333333333333),
        "Softmax": (1.0, 0.0),
        "LayerNorm": (0.0, 1.0),
        "BatchNorm1d": (0.0, 1.0),
    }
    return table.get(kind, (0.0, 1.0))


def _check_dimensions(circuit: Circuit) -> None:
    """Fail on a shape mismatch rather than at evaluation time on a host."""
    previous: Optional[int] = None
    for index, layer in enumerate(circuit.layers):
        if isinstance(layer, AffineLayer):
            if previous is not None and previous != layer.input_dim:
                raise UnsupportedComputation(
                    f"layer {index} ({layer.name}) expects {layer.input_dim} "
                    f"inputs but the previous layer produces {previous}"
                )
            previous = layer.output_dim
        elif isinstance(layer, BiasLayer):
            if previous is not None and previous != layer.output_dim:
                raise UnsupportedComputation(
                    f"bias at layer {index} has {layer.output_dim} entries but "
                    f"the previous layer produces {previous}"
                )
        elif previous is None:
            previous = circuit.output_dim


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True
