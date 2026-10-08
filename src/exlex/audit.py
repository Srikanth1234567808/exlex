"""Static analysis: find host-boundary crossings in code before it runs.

The runtime guard in :mod:`exlex.residency` catches transfers that happen. This
catches the ones that are written down, in source, where a reviewer can see
them, and it catches them in CI rather than in production.

The analysis is AST-based, so it has no false negatives from the search text
being split across lines, and it is cheap enough for a pre-commit hook. It does
have real blind spots, all of which are listed in :data:`KNOWN_BLIND_SPOTS` and
surfaced in the CLI output rather than hidden.

Design: the tool reports, it does not rewrite. An autofixer that rewrites
``tensor.cpu()`` into something else usually breaks the computation, and a
silent rewrite in a security tool is worse than no rewrite.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Set, Tuple

#: Method names that move tensor data out of device memory.
TRANSFER_METHODS: Set[str] = {
    "cpu",
    "numpy",
    "item",
    "tolist",
    "to",
    "tobytes",
    "detach",
    "share_memory_",
    "pin_memory",
}

#: Functions that serialise an object graph and may capture secrets.
SERIALISE_CALLS: Set[str] = {
    "save",
    "torch.save",
    "numpy.save",
    "np.save",
    "pickle.dump",
    "pickle.dumps",
    "joblib.dump",
    "marshal.dumps",
}

#: Sinks that move data off the machine.
NETWORK_SINKS: Set[str] = {
    "requests.post",
    "requests.put",
    "requests.get",
    "urllib.request.urlopen",
    "httpx.post",
    "httpx.get",
    "socket.send",
    "socket.sendall",
    "boto3.client",
    "paramiko.SSHClient",
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.check_output",
}

#: Sinks that write to a file, which persists plaintext indefinitely.
FILE_SINKS: Set[str] = {"open", "io.open", "os.open"}

#: Decorators that mark a function as handling secret data.
SECRET_MARKERS: Tuple[str, ...] = ("exlex.secret", "exlex.guard", "secret", "guard")

#: Documented limits of this analysis. Printed by the CLI.
KNOWN_BLIND_SPOTS: Tuple[str, ...] = (
    "Dynamically resolved attribute names (getattr(t, 'cp' + 'u')) are invisible.",
    "Assignments that hide a secret in a container or a dict escape name tracking.",
    "Code executed through exec, eval, importlib, or a notebook cell is not walked.",
    "C extensions and compiled kernels are opaque; only Python is analysed.",
    "A transfer inside an unmarked function is not flagged, because nothing "
    "declares that function secret.",
    "Interprocedural flows are not tracked: a secret passed to an unmarked "
    "helper is not followed into it.",
)


@dataclass
class Finding:
    """One suspected host-boundary crossing.

    Attributes:
        rule: stable identifier, e.g. ``"transfer"`` or ``"network"``.
        severity: ``"high"`` or ``"medium"``.
        lineno: 1-based source line.
        col: 0-based column.
        call: the reconstructed call text, truncated.
        message: what the problem is.
        in_secret: whether it sits inside a function marked as secret-handling.
    """

    rule: str
    severity: str
    lineno: int
    col: int
    call: str
    message: str
    in_secret: bool

    def __str__(self) -> str:
        tag = " [in secret region]" if self.in_secret else ""
        return f"{self.lineno}:{self.col} {self.severity} {self.rule}{tag}: {self.message}"


def _dotted(node: ast.AST) -> str:
    """Reconstruct a dotted name, best effort, for a readable call string."""
    parts: List[str] = []
    current: Optional[ast.AST] = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _call_text(node: ast.Call) -> str:
    try:
        text = ast.unparse(node)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - py<3.9 fallback
        return _dotted(node.func)
    return text if len(text) <= 120 else text[:117] + "..."


def _is_transfer_call(node: ast.Call) -> Optional[str]:
    """Return a reason string if this call moves data to host memory."""
    if not isinstance(node.func, ast.Attribute):
        return None
    name = node.func.attr
    if name not in TRANSFER_METHODS:
        return None
    if name == "to":
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.split(":")[0] in ("cpu", ""):
                    return f".to({arg.value!r}) targets host memory"
        if not node.args and not node.keywords:
            return ".to() with no device argument may default to host"
        return None
    if name == "detach":
        return ".detach() yields a tensor that can be moved freely to host"
    if name == "pin_memory":
        return ".pin_memory() allocates a page-locked host buffer"
    return f".{name}() copies data out of device memory"


class _Visitor(ast.NodeVisitor):
    """Collect findings, tracking whether we are inside a secret region."""

    def __init__(self) -> None:
        self.findings: List[Finding] = []
        self._secret_depth = 0
        self._allow_depth = 0
        self._secret_names: Set[str] = set()

    # ------------------------------------------------------------- functions

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def _visit_function(self, node) -> None:
        marked = _is_secret_marked(node)
        was = self._secret_depth
        saved_names = set(self._secret_names)
        if marked:
            self._secret_depth += 1
        for arg in _all_args(node):
            self._secret_names.add(arg.arg)
        for stmt in node.body:
            self.visit(stmt)
        self._secret_depth = was
        self._secret_names = saved_names

    # ----------------------------------------------------------------- calls

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        in_allow = self._allow_depth > 0
        secret_arg = any(
            isinstance(a, ast.Name) and a.id in self._secret_names for a in node.args
        )

        if not in_allow:
            reason = _is_transfer_call(node)
            if reason:
                self._add(
                    Finding(
                        rule="transfer",
                        severity="high" if self._secret_depth else "medium",
                        lineno=node.lineno,
                        col=node.col_offset,
                        call=_call_text(node),
                        message=reason,
                        in_secret=bool(self._secret_depth),
                    )
                )

            dotted = _dotted(node.func)
            bare = node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if dotted in SERIALISE_CALLS or bare in ("save", "dumps", "dump"):
                if self._secret_depth or secret_arg:
                    self._add(
                        Finding(
                            rule="serialize",
                            severity="high",
                            lineno=node.lineno,
                            col=node.col_offset,
                            call=_call_text(node),
                            message="serialising a secret to disk captures plaintext",
                            in_secret=bool(self._secret_depth),
                        )
                    )
            if dotted in NETWORK_SINKS or dotted.split(".")[0] in (
                "requests",
                "httpx",
                "urllib",
                "socket",
                "boto3",
                "paramiko",
                "subprocess",
            ):
                if self._secret_depth or secret_arg:
                    self._add(
                        Finding(
                            rule="network",
                            severity="high",
                            lineno=node.lineno,
                            col=node.col_offset,
                            call=_call_text(node),
                            message="secret data may cross the network boundary",
                            in_secret=bool(self._secret_depth),
                        )
                    )
            if dotted in FILE_SINKS and (self._secret_depth or secret_arg):
                self._add(
                    Finding(
                        rule="file",
                        severity="medium",
                        lineno=node.lineno,
                        col=node.col_offset,
                        call=_call_text(node),
                        message="secret data may be written to disk in the clear",
                        in_secret=bool(self._secret_depth),
                    )
                )

        self.generic_visit(node)

    # ------------------------------------------------------------ statements

    def visit_With(self, node: ast.With) -> None:  # noqa: N802
        is_allow = any(_context_is_allow(item.context_expr) for item in node.items)
        previous = self._allow_depth
        if is_allow:
            self._allow_depth += 1
        for item in node.items:
            self.visit(item.context_expr)
        for stmt in node.body:
            self.visit(stmt)
        self._allow_depth = previous

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:  # noqa: N802
        is_allow = any(_context_is_allow(item.context_expr) for item in node.items)
        previous = self._allow_depth
        if is_allow:
            self._allow_depth += 1
        for item in node.items:
            self.visit(item.context_expr)
        for stmt in node.body:
            self.visit(stmt)
        self._allow_depth = previous

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for target in node.targets:
            for name in _target_names(target):
                if _secret_looking(name):
                    self._secret_names.add(name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if node.value is None:
            return
        for name in _target_names(node.target):
            if _secret_looking(name):
                self._secret_names.add(name)
        self.generic_visit(node)

    def _add(self, finding: Finding) -> None:
        self.findings.append(finding)


def _all_args(node) -> List[ast.arg]:
    a = node.args
    collected = list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)
    if a.vararg:
        collected.append(a.vararg)
    if a.kwarg:
        collected.append(a.kwarg)
    return collected


def _target_names(node: ast.AST) -> Iterator[str]:
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            yield from _target_names(element)


_SECRET_WORDS = (
    "secret",
    "password",
    "passwd",
    "token",
    "key",
    "credential",
    "weights",
    "checkpoint",
    "ckpt",
    "private",
    "embeds",
)


def _secret_looking(name: str) -> bool:
    lowered = name.lower()
    return any(word in lowered for word in _SECRET_WORDS)


def _context_is_allow(node: ast.AST) -> bool:
    """True if a ``with`` item is an exlex.allow(...) context manager.

    The name to inspect is on the ``Call`` node, not the context expression, so
    both ``with exlex.allow("x")`` and ``with exlex.guard.allow("x")`` are
    matched.
    """
    target = node.func if isinstance(node, ast.Call) else node
    text = _dotted(target)
    return text.split(".")[-1] == "allow" or text.endswith("allow")


def _is_secret_marked(node) -> bool:
    for decorator in node.decorator_list:
        text = _dotted(decorator) or ast.dump(decorator)
        if any(marker in text for marker in SECRET_MARKERS):
            return True
        if isinstance(decorator, ast.Call):
            inner = _dotted(decorator.func)
            if any(marker in inner for marker in SECRET_MARKERS):
                return True
    return False


def audit_source(source: str, filename: str = "<string>") -> List[Finding]:
    """Audit a Python source string. Returns findings sorted by position.

    Raises:
        SyntaxError: if ``source`` does not parse. Not swallowed, because a
            file that fails to parse has not been audited.
    """
    tree = ast.parse(source, filename=filename)
    visitor = _Visitor()
    visitor.visit(tree)
    return sorted(visitor.findings, key=lambda f: (f.lineno, f.col))


def audit_file(path: Path) -> List[Finding]:
    """Audit one file. Non-Python files are skipped by returning an empty list."""
    path = Path(path)
    if path.suffix != ".py":
        return []
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return audit_source(source, filename=str(path))


def audit_path(root: Path) -> List[Finding]:
    """Recursively audit a directory tree, skipping common vendor directories."""
    root = Path(root)
    skip = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist", ".mypy_cache"}
    findings: List[Finding] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in skip for part in path.parts):
            continue
        findings.extend(audit_file(path))
    return findings


def to_dicts(findings: Sequence[Finding]) -> List[dict]:
    """Serialise findings for JSON output or CI annotations."""
    return [
        {
            "rule": f.rule,
            "severity": f.severity,
            "lineno": f.lineno,
            "col": f.col,
            "call": f.call,
            "message": f.message,
            "in_secret": f.in_secret,
        }
        for f in findings
    ]
