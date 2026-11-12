# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Report where the Python lines of solprop go, and check them against ratchets.

``python -m tools.size_report`` prints, for each top-level package, its code,
docstring (public and private), comment and blank lines, and the counts that show
book-keeping: private docstrings with numpydoc sections, docstrings longer than their
body, ``noqa`` comments by rule, private names imported across packages, ``__all__``
lines and an estimate of cloned code. ``--json`` prints the same as JSON; ``--check``
compares ``src/solprop`` with the caps in ``tools/size_budget.toml`` and exits 1 when
one is exceeded. A name is private by exactly the rule that numpydoc's validation
excludes: a single-underscore component in ``module.qualname``, the module named by its
file stem, so that this report and ``poe docstrings`` agree. Only the standard library
is used.
"""

import argparse
import ast
import io
import json
import keyword
import re
import shutil
import subprocess
import sys
import tokenize
import tomllib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["main"]

_REPO = Path(__file__).resolve().parent.parent
_ROOTS = ("src/solprop",)
_EXTRA_ROOTS = ("native/src/solprop_native", "gui/src/solprop_gui")
_BUDGET = _REPO / "tools" / "size_budget.toml"
_PRIVATE = re.compile(r"(^|\.)_[^_]")
_SECTION = re.compile(
    r"^\s*(Parameters|Returns|Yields|Raises|Examples|See Also)\s*\n\s*-{3,}\s*$", re.M
)
_NOQA = re.compile(r"#\s*noqa:\s*([A-Z]+[0-9]+(?:\s*,\s*[A-Z]+[0-9]+)*)")
_WINDOW = 50  # tokens in a clone window
_TOP = 20
_KEYWORDS = frozenset(keyword.kwlist)


@dataclass
class _Package:
    """Line counts for one top-level package (or the root's own modules)."""

    files: int = 0
    total: int = 0
    code: int = 0
    public_doc: int = 0
    private_doc: int = 0
    comment: int = 0
    blank: int = 0
    all_lines: int = 0
    sectioned_private: int = 0
    doc_over_body: int = 0


@dataclass
class _Report:
    """Everything one run measures, keyed for the text and JSON forms."""

    packages: dict[str, _Package] = field(default_factory=dict)
    largest_files: list[tuple[str, int]] = field(default_factory=list)
    largest_functions: list[tuple[str, int]] = field(default_factory=list)
    noqa: dict[str, int] = field(default_factory=dict)
    private_imports: list[str] = field(default_factory=list)
    clone_percent: float = 0.0
    exact_clone_percent: float = 0.0
    clone_pairs: list[tuple[str, str, int]] = field(default_factory=list)
    complexity: dict[str, int] = field(default_factory=dict)
    module_lines: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _Source:
    """One file read, parsed and tokenised once."""

    path: Path
    text: str
    tree: ast.Module
    tokens: list[tokenize.TokenInfo]


def _read(path: Path) -> _Source:
    text = path.read_text(encoding="utf-8")
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    return _Source(path, text, ast.parse(text), tokens)


def _package_of(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    return parts[0] if len(parts) > 1 else "(top-level)"


def _docstring_node(node: ast.AST) -> ast.Constant | None:
    body = getattr(node, "body", None)
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value
    return None


def _walk_definitions(tree: ast.Module, stem: str) -> list[tuple[ast.AST, str, bool]]:
    # (node, dotted name as numpydoc forms it, private), the module first.
    found: list[tuple[ast.AST, str, bool]] = [(tree, stem, bool(_PRIVATE.search(stem)))]

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f"{prefix}.{child.name}"
                found.append((child, name, bool(_PRIVATE.search(name))))
                visit(child, name)
            else:
                visit(child, prefix)

    visit(tree, stem)
    return found


def _line_kinds(tokens: list[tokenize.TokenInfo]) -> tuple[set[int], set[int]]:
    # The lines holding only a comment, and the lines holding any other token.
    comments: set[int] = set()
    code: set[int] = set()
    skip = {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT}
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            comments.add(tok.start[0])
        elif tok.type not in skip and tok.type != tokenize.ENDMARKER:
            code.update(range(tok.start[0], tok.end[0] + 1))
    return comments - code, code


def _tokens(tokens: list[tokenize.TokenInfo], *, names: bool) -> list[str]:
    # Normalised tokens: literals lose their spelling, and names too unless `names`.
    out: list[str] = []
    keep = {tokenize.OP, tokenize.NEWLINE}
    for tok in tokens:
        if tok.type == tokenize.NAME:
            out.append(tok.string if names or tok.string in _KEYWORDS else "N")
        elif tok.type == tokenize.NUMBER:
            out.append("0")
        elif tok.type in {tokenize.STRING, tokenize.FSTRING_START}:
            out.append("S")
        elif tok.type in keep:
            out.append(tok.string or ";")
    return out


def _measure_file(
    source: _Source, package: _Package, report: _Report, label: str
) -> None:
    text, tree = source.text, source.tree
    lines = text.splitlines()
    comment_lines, code_lines = _line_kinds(source.tokens)
    doc_lines: dict[int, bool] = {}  # line -> private
    definitions = _walk_definitions(tree, source.path.stem)
    for node, _, private in definitions:
        doc = _docstring_node(node)
        if doc is None:
            continue
        span = range(doc.lineno, (doc.end_lineno or doc.lineno) + 1)
        doc_lines.update(dict.fromkeys(span, private))
        if private and _SECTION.search(str(doc.value)):
            package.sectioned_private += 1
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            body = (node.end_lineno or node.lineno) - (doc.end_lineno or doc.lineno)
            if len(span) > body:
                package.doc_over_body += 1
    for node, name, _ in definitions:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            size = (node.end_lineno or node.lineno) - node.lineno + 1
            report.largest_functions.append((f"{label}:{name}", size))
    blank = sum(  # a blank line inside a string belongs to the string
        1 for n, line in enumerate(lines, 1) if not line.strip() and n not in code_lines
    )
    package.files += 1
    package.total += len(lines)
    package.blank += blank
    package.comment += len(comment_lines)
    package.public_doc += sum(1 for p in doc_lines.values() if not p)
    package.private_doc += sum(1 for p in doc_lines.values() if p)
    package.code += len(code_lines - doc_lines.keys())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                package.all_lines += (node.end_lineno or node.lineno) - node.lineno + 1
    for match in _NOQA.finditer(text):
        for code in re.split(r"\s*,\s*", match.group(1)):
            report.noqa[code] = report.noqa.get(code, 0) + 1
    report.largest_files.append((label, len(lines)))
    report.module_lines[label] = len(lines)


def _private_imports(source: _Source, root: Path, label: str) -> list[str]:
    # Private names imported from another package: a private name, a name from a private
    # module of a subpackage, or a private module itself (`import solprop.x._y`,
    # `from solprop import _y`). Public names from the root's private modules
    # (`from solprop._process import ...`) are shared by design and do not count.
    top = root.name
    here = _package_of(source.path, root)

    def package(name: str) -> str:
        return name if (root / name).is_dir() else "(top-level)"

    found: list[str] = []
    for node in ast.walk(source.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                crosses = (
                    len(parts) > 1 and parts[0] == top and package(parts[1]) != here
                )
                if crosses and any(_PRIVATE.match(p) for p in parts[1:]):
                    found.append(f"{label}:{node.lineno} {alias.name}")
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            parts = node.module.split(".")
            if parts[0] != top:
                continue
            module_private = any(_PRIVATE.match(p) for p in parts[2:])
            for alias in node.names:
                there = package(parts[1] if len(parts) > 1 else alias.name)
                if there != here and (module_private or _PRIVATE.match(alias.name)):
                    found.append(f"{label}:{node.lineno} {node.module}.{alias.name}")
    return found


def _clones(
    token_lists: dict[str, list[str]],
) -> tuple[float, list[tuple[str, str, int]]]:
    # Rabin-Karp hashes of every window of _WINDOW normalised tokens; a window seen in
    # two places is cloned. Returns the cloned share of tokens and the top file pairs.
    # A place is packed as file index * 2**32 + token index, and a hash keeps its first
    # place alone until a second one turns up, so that memory follows the clones.
    ids: dict[str, int] = {}
    labels = list(token_lists)
    mod, base = (1 << 61) - 1, 1_000_003
    power = pow(base, _WINDOW - 1, mod)
    first: dict[int, int] = {}
    again: dict[int, list[int]] = defaultdict(list)
    total = 0
    for index, label in enumerate(labels):
        values = [ids.setdefault(t, len(ids) + 1) for t in token_lists[label]]
        total += len(values)
        h = 0
        for i, v in enumerate(values):
            if i >= _WINDOW:
                h = (h - values[i - _WINDOW] * power) % mod
            h = (h * base + v) % mod
            if i >= _WINDOW - 1:
                place = (index << 32) | (i - _WINDOW + 1)
                if first.setdefault(h, place) != place:
                    again[h].append(place)
    covered: dict[int, set[int]] = defaultdict(set)
    pairs: Counter[tuple[str, str]] = Counter()
    for h, later in again.items():
        places = [first[h], *later]
        if len(places) > 50:  # boilerplate seen everywhere is not a clone
            continue
        for place in places:
            start = place & 0xFFFFFFFF
            covered[place >> 32].update(range(start, start + _WINDOW))
        files = sorted({labels[place >> 32] for place in places})
        for a in range(len(files)):
            for b in range(a + 1, len(files)):
                pairs[files[a], files[b]] += 1
    percent = 100.0 * sum(map(len, covered.values())) / max(total, 1)
    return percent, [(a, b, n) for (a, b), n in pairs.most_common(10)]


def _complexity(paths: list[str]) -> dict[str, int]:
    ruff = shutil.which("ruff")
    if ruff is None:
        return {}
    run = subprocess.run(
        [ruff, "check", "--select", "C901,PLR0912,PLR0915", "--ignore-noqa",
         "--statistics", "--exit-zero", *paths],
        cwd=_REPO, capture_output=True, text=True, check=False,
    )  # fmt: skip
    counts: dict[str, int] = {}
    for line in run.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].isdigit():
            counts[fields[1]] = int(fields[0])
    return counts


def _measure(roots: list[str], *, full: bool = True) -> _Report:
    # `full=False` (the check) leaves out the clone and complexity passes.
    report = _Report()
    token_lists: dict[str, list[str]] = {}
    exact_lists: dict[str, list[str]] = {}
    for root_name in roots:
        root = _REPO / root_name
        for path in sorted(root.rglob("*.py")):
            label = path.relative_to(_REPO).as_posix()
            key = _package_of(path, root)
            key = key if root_name == _ROOTS[0] else f"{root.name}/{key}"
            package = report.packages.setdefault(key, _Package())
            source = _read(path)
            _measure_file(source, package, report, label)
            report.private_imports += _private_imports(source, root, label)
            if full:
                token_lists[label] = _tokens(source.tokens, names=False)
                exact_lists[label] = _tokens(source.tokens, names=True)
    report.largest_files = sorted(report.largest_files, key=lambda x: -x[1])[:_TOP]
    report.largest_functions = sorted(report.largest_functions, key=lambda x: -x[1])[
        :_TOP
    ]
    report.noqa = dict(sorted(report.noqa.items(), key=lambda x: -x[1]))
    if not full:
        return report
    report.clone_percent, report.clone_pairs = _clones(token_lists)
    report.exact_clone_percent = _clones(exact_lists)[0]
    report.complexity = _complexity(roots)
    return report


def _text(report: _Report) -> str:
    out = io.StringIO()
    heads = ("package", "files", "total", "code", "pub doc", "priv doc", "comment",
             "blank", "__all__", "sect priv", "doc>body")  # fmt: skip
    counts = {
        k: [int(v) for v in asdict(p).values()] for k, p in report.packages.items()
    }
    sums = [sum(column) for column in zip(*counts.values(), strict=True)]
    rows = [(k, *v) for k, v in sorted(counts.items())]
    totals = ("total", *sums)
    widths = [
        max(len(str(r[i])) for r in [heads, *rows, totals]) for i in range(len(heads))
    ]
    for row in [heads, *rows, totals]:
        cells = [
            str(c).rjust(w) if i else str(c).ljust(w)
            for i, (c, w) in enumerate(zip(row, widths, strict=True))
        ]
        out.write("  ".join(cells) + "\n")
    out.write("\nLargest files:\n")
    out.writelines(f"  {n:6d}  {name}\n" for name, n in report.largest_files)
    out.write("\nLargest functions:\n")
    out.writelines(f"  {n:6d}  {name}\n" for name, n in report.largest_functions)
    out.write(
        "\nnoqa by rule: "
        + ", ".join(f"{k} {v}" for k, v in report.noqa.items())
        + "\n"
    )
    out.write(
        f"Private names imported across packages: {len(report.private_imports)}\n"
    )
    out.write(
        f"Clones ({_WINDOW}-token windows): {report.clone_percent:.1f} % of tokens "
        f"with names renamed, {report.exact_clone_percent:.1f} % as written:\n"
    )
    out.writelines(f"  {n:6d}  {a}  ~  {b}\n" for a, b, n in report.clone_pairs)
    if report.complexity:
        out.write("Complexity (ruff, noqa ignored): ")
        out.write(", ".join(f"{k} {v}" for k, v in report.complexity.items()) + "\n")
    return out.getvalue()


def _check(report: _Report, budget: dict[str, Any]) -> list[str]:  # TOML tables
    # The ratchets of tools/size_budget.toml, on the package's own root only.
    failures: list[str] = []
    sectioned = sum(
        p.sectioned_private for k, p in report.packages.items() if "/" not in k
    )
    cap = budget["private_docstrings"]["max_sectioned"]
    if sectioned > cap:
        failures.append(
            f"{sectioned} private docstrings have numpydoc sections (cap {cap})"
        )
    limit: int = budget["modules"]["max_lines"]
    allowed: dict[str, int] = budget["modules"]["allow"]
    for label, n in report.module_lines.items():
        if label.startswith(_ROOTS[0]) and n > allowed.get(label, limit):
            failures.append(f"{label} has {n} lines (cap {allowed.get(label, limit)})")
    imports = [i for i in report.private_imports if i.startswith(_ROOTS[0])]
    cap = budget["private_imports"]["max_cross_package"]
    if len(imports) > cap:
        failures.append(
            f"{len(imports)} private names imported across packages (cap {cap})"
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    """Print the size report, or check it against the budget; return the exit status.

    Parameters
    ----------
    argv : list of str, optional
        The command-line arguments; ``sys.argv[1:]`` by default.

    Returns
    -------
    int
        0, or 1 when ``--check`` finds a cap exceeded.
    """
    parser = argparse.ArgumentParser(
        prog="size_report", description="Report the size of solprop's Python code."
    )
    parser.add_argument(
        "--all", action="store_true", help="add the native adapter and the GUI"
    )
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument(
        "--check", action="store_true", help="check tools/size_budget.toml"
    )
    args = parser.parse_args(argv)
    roots = [*_ROOTS, *(_EXTRA_ROOTS if args.all else ())]
    report = _measure(roots, full=not args.check)
    if args.check:
        failures = _check(report, tomllib.loads(_BUDGET.read_text(encoding="utf-8")))
        sys.stdout.writelines(f"size-check: {f}\n" for f in failures)
        if not failures:
            sys.stdout.write("size-check: within budget\n")
        return 1 if failures else 0
    if args.json:
        payload = asdict(report)
        payload["private_imports"] = len(report.private_imports)
        sys.stdout.write(json.dumps(payload, indent=1) + "\n")
    else:
        sys.stdout.write(_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
