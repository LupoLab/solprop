# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Cut private and nested docstrings down to the house rule (AGENTS.md).

``python -m tools.docstring_diet PATH... [--write] [--check] [--report]
[--rename-closures]``. A definition is private when its numpydoc name (module stem and
qualname) has a single-underscore component, and nested when it sits in a function.
Either loses its ``Parameters``, ``Returns``, ``Yields``, ``Raises``, ``Examples`` and
``See Also`` sections; the summary, the extended summary and the other sections stay
verbatim, and an entry that states a unit, range, condition, convention or shape becomes
a one-line bullet. ``--check`` proves that each file still compiles and that its AST
without docstrings is unchanged. ``--rename-closures`` renames a nested function that
numpydoc validates (a public name in a public function) to ``_name`` in its function.
"""

import argparse
import ast
import functools
import re
import sys
import textwrap
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from numpydoc.docscrape import NumpyDocString

__all__ = ["main"]

_PRIVATE = re.compile(r"(^|\.)_[^_]")
_BULLETED = ("Parameters", "Other Parameters", "Receives", "Returns", "Yields")
_DROP = {*_BULLETED, "Raises", "Warns", "Examples", "See Also"}
_KEEP = {"Notes", "References", "Warnings", "Attributes", "Methods"}
_UNDERLINE = re.compile(r"^\s*-{3,}\s*$")
_QUOTED = re.compile(r"^([rRuU]{0,2})(\"\"\"|''')(.*)\2$", re.DOTALL)
_JAXTYPING = re.compile(
    r"\b(Float|Complex|Int|UInt|Bool|Shaped|Num|Real|Inexact|Integer)\["
)
_SHAPE = re.compile(r",?\s*(of |with |in )?shape\s*(``[^`]*``|\([^)]*\))")
_FACTS = re.compile(
    "|".join((  # a unit; a range, condition or sentinel; a convention or layout
        r"(?<![\w'`^])(rad/s|V/m|W/m\^?2|J/m\^?2|m\^-?\d|m⁻³|m\*\*-?\d|[munpf]?s|"
        r"[mnµμu]?m|cm|W|J|K|Pa|bar|eV|T?Hz|rad|GiB|MiB|(?i:bytes|seconds?|metres?))(?![\w'-])",
        r"\b(must|if|when|where|unless|only|between|positive|non-negative|nonnegative|"
        r"non-zero|nonzero|at least|at most|finite|strictly|clamp\w*|None|empty|"
        r"otherwise|else|left out|omitted)\b",
        r"``[\"']?[\w.-]+[\"']?``,? or ``",  # a choice of literals
        r"[<>≤≥]|\[\s*-?\w+\s*,|\(\s*-?[\d.]+\s*,\s*[\d.∞]",
        r"\b(FFT|DFT|conventions?|sign|normali[sz]\w*|layout|axis|axes|row-major|"
        r"contiguous|interface|internal|conjugat\w*|keyed)\b",
    ))
)  # fmt: skip
_COMPOSITE = re.compile(r"^(tuple|dict|list of tuple|(collections\.abc\.)?Mapping)\b")
_PROPER = {"True", "False", "None", "Kerr", "Raman", "Fourier", "Butcher", "Adams"}
_PROPER |= {"Milne", "Hermite", "Gaussian", "Newton", "Python", "NumPy", "FFTW"}


@dataclass
class _Report:
    """What a run did to one file."""

    removed: int = 0
    examples: list[str] = field(default_factory=list)
    closures: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    flattened: list[str] = field(default_factory=list)


def _exclude(path: Path) -> re.Pattern[str]:
    """Return numpydoc's exclude pattern, from the nearest ``pyproject.toml``."""
    for parent in path.resolve().parents:
        if (toml := parent / "pyproject.toml").is_file():
            with toml.open("rb") as stream:
                config = tomllib.load(stream).get("tool", {})
            patterns = config.get("numpydoc_validation", {}).get("exclude", [])
            return re.compile("|".join(patterns) or r"(?!)")
    return re.compile(r"(?!)")


def _module(path: Path) -> tuple[Path, str]:
    """Return the directory above ``path``'s top package, and its dotted module name."""
    parts, parent = [path.stem], path.resolve().parent
    while (parent / "__init__.py").is_file():
        parts.insert(0, parent.name)
        parent = parent.parent
    return parent, ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


@functools.cache
def _reexported(root: Path) -> frozenset[tuple[str, str]]:
    """Return each (module, name) that a public module lists in its ``__all__``.

    A class or function defined in a private module and re-exported is public API: its
    docstring is the reference page, whatever numpydoc's file-based name says.
    """
    edges: list[tuple[str, str, str, str]] = []  # importer, source, name, alias
    for file in sorted(root.rglob("*.py")):
        tree, module = ast.parse(file.read_bytes()), _module(file)[1]
        listed = {
            c.value
            for node in tree.body
            if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign)
            and "__all__" in ast.unparse(node).partition("=")[0]
            for c in ast.walk(node)
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
        }
        edges += [
            (module, node.module, a.name, a.asname or a.name)
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
            for a in node.names
            if (a.asname or a.name) in listed
        ]
    found: set[tuple[str, str]] = set()
    while True:
        more = {
            (source, name)
            for importer, source, name, alias in edges
            if not _PRIVATE.search(importer) or (importer, alias) in found
        }
        if more <= found:
            return frozenset(found)
        found |= more


def _demands_sections(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether numpydoc asks this function for Parameters, Returns or Yields.

    Mirrors numpydoc's PR01, RT01 and YD01: any argument, a ``return`` with a value
    outside nested functions, or a ``yield`` statement in the body.
    """
    args = node.args
    if args.posonlyargs or args.args or args.kwonlyargs or args.vararg or args.kwarg:
        return True
    stack: list[ast.AST] = list(node.body)
    while stack:
        child = stack.pop()
        value = getattr(child, "value", None)
        if isinstance(child, ast.Return) and not (
            value is None or (isinstance(value, ast.Constant) and value.value is None)
        ):
            return True
        if isinstance(child, ast.Expr) and isinstance(
            child.value, ast.Yield | ast.YieldFrom
        ):
            return True
        if not isinstance(child, ast.FunctionDef):
            stack.extend(ast.iter_child_nodes(child))
    return False


def _bullet(label: str, desc: list[str], indent: str) -> list[str]:
    """Return one entry as a wrapped ``- label: fact.`` bullet."""
    fact = " ".join(" ".join(desc).split())
    first = fact.partition(" ")[0].rstrip(",.:;")
    if first.istitle() and first.isalpha() and first not in _PROPER:
        fact = fact[0].lower() + fact[1:]
    fact = fact.rstrip(",;:") if fact.endswith((",", ";", ":")) else fact
    fact = fact if fact.endswith(".") else f"{fact}."
    return textwrap.wrap(
        f"- {label}: {fact}",
        width=88,
        initial_indent=indent,
        subsequent_indent=f"{indent}  ",
        break_long_words=False,
        break_on_hyphens=False,
    )


def _words(text: str) -> set[str]:
    """Return the lower-case words of ``text``, without the commonest ones."""
    return set(re.findall(r"\w+", text.lower())) - {"the", "a", "an", "of", "in", "to"}


def _bullets(  # noqa: PLR0917 - private, called once
    doc: NumpyDocString,
    node: ast.AST,
    indent: str,
    report: _Report,
    name: str,
    said: set[str],
) -> list[str]:
    """Return the bullets of the entries whose description states a fact."""
    annotations: dict[str, str] = {}
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        arguments = node.args
        every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        every += [a for a in (arguments.vararg, arguments.kwarg) if a is not None]
        annotations = {a.arg: ast.unparse(a.annotation) for a in every if a.annotation}
        annotations["return"] = ast.unparse(node.returns) if node.returns else ""
    lines: list[str] = []
    for section in _BULLETED:
        for entry in doc[section]:
            names = [n.strip() for n in entry.name.split(",") if n.strip()]
            returned = section in {"Returns", "Yields"}
            keys = ["return"] if returned else [n.lstrip("*") for n in names]
            text = " ".join(entry.desc)
            shaped = all(_JAXTYPING.search(annotations.get(k, "")) for k in keys)
            if shaped:  # the annotation carries the shape
                text = _SHAPE.sub("", text)
            composite = _COMPOSITE.match(entry.type) and re.search(r"``|;| and ", text)
            if not (
                (not shaped and "shape" in text) or composite or _FACTS.search(text)
            ):
                continue
            if any(
                not line.strip() or line.lstrip().startswith("..")
                for line in entry.desc
            ):
                report.flattened.append(name)
            label = ", ".join(f"``{n}``" for n in names)
            if returned:
                label = f"{section} {label}" if label and entry.type else section
            if "``" in text or not _words(text) <= said:  # else the summary says it
                lines += _bullet(label, entry.desc, indent)
    return lines


def _slim(
    seg: str, node: ast.AST, column: int, name: str, report: _Report
) -> str | None:
    """Return the slimmed source of one docstring literal, or None to leave it alone."""
    quoted = _QUOTED.match(seg)
    if quoted is None:
        report.skipped.append(f"{name}: not a triple-quoted literal")
        return None
    prefix, quote, content = quoted.groups()
    lines = content.split("\n")
    heads = [
        i
        for i in range(1, len(lines) - 1)
        if _UNDERLINE.match(lines[i + 1]) and lines[i].strip()
    ]
    titles = [lines[i].strip() for i in heads]
    if unknown := [t for t in titles if t not in _DROP | _KEEP]:
        report.skipped.append(f"{name}: unknown sections {unknown}")
        return None
    if not set(titles) & _DROP:
        return None
    try:
        doc = NumpyDocString(textwrap.dedent("\n".join(lines[1:])))
    except Exception as error:
        report.skipped.append(f"{name}: {error}")
        return None
    report.examples += [name] if "Examples" in titles else []
    indent = " " * min(
        (len(line) - len(line.lstrip()) for line in lines[1:] if line.strip()),
        default=column,
    )
    out = _trimmed(lines[: heads[0]])
    said = _words("\n".join(lines[: heads[0]]))
    if bullets := _bullets(doc, node, indent, report, name, said):
        out += ["", *bullets]
    for title, start, end in zip(titles, heads, [*heads[1:], len(lines)], strict=True):
        if title in _KEEP:
            out += ["", *_trimmed(lines[start:end])]
    if len(out) == 1 and column + len(prefix) + 6 + len(out[0]) > 88:
        report.skipped.append(f"{name}: summary too long to stand alone (D200)")
        return None
    text = out[0] if len(out) == 1 else "\n".join(out) + f"\n{indent}"
    return f"{prefix}{quote}{text}{quote}"


def _trimmed(lines: list[str]) -> list[str]:
    """Return ``lines`` without trailing blank lines."""
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines


class _Diet:
    """Plan the docstring and rename edits of one module."""

    def __init__(self, path: Path, source: str, rename: bool) -> None:
        self.path, self.source, self.rename = path, source, rename
        self.lines = source.splitlines(keepends=True)
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))
        self.exclude = _exclude(path)
        root, module = _module(path)
        self.public = {name for m, name in _reexported(root) if m == module}
        self.edits: list[tuple[int, int, str]] = []
        self.renamed: list[tuple[tuple[str, ...], str, str]] = []
        self.report = _Report()

    def offset(self, line: int, col: int) -> int:
        """Return the character offset of an AST position (UTF-8 column)."""
        text = self.lines[line - 1].encode()[:col].decode()
        return self.starts[line - 1] + len(text)

    def visit(
        self, parent: ast.AST, stack: tuple[str, ...], func: bool, seen: bool
    ) -> None:
        """Plan the edits of every definition under ``parent``.

        ``seen`` tracks whether numpydoc's visitor reaches the node: it descends only
        through the direct bodies of modules, classes and functions.
        """
        body = getattr(parent, "body", [])
        for child in ast.iter_child_nodes(parent):
            if not isinstance(
                child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
            ):
                self.visit(child, stack, func, seen=False)
                continue
            reached = seen and any(child is s for s in body)
            qual = (*stack, child.name)
            name = ".".join((self.path.stem, *qual))
            validated = func and reached and not self.exclude.search(name)
            validated = validated and not _PRIVATE.search(name)
            if validated and (
                isinstance(child, ast.ClassDef) or _demands_sections(child)
            ):
                closure = isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef)
                if not (
                    self.rename
                    and closure
                    and not isinstance(child, ast.ClassDef)
                    and self.rename_closure(parent, child, stack)
                ):
                    self.report.closures.append(f"{name} (kept)")
                    self.visit(child, qual, True, seen=True)
                    continue
                qual = (*stack, f"_{child.name}")
                self.report.closures.append(f"{name} -> _{child.name}")
            public = qual[0] in self.public  # re-exported from a private module
            private = _PRIVATE.search(".".join(qual) if public else name)
            if (func or private) and ast.get_docstring(child, clean=False):
                self.slim(child, name)
            self.visit(
                child, qual, func or not isinstance(child, ast.ClassDef), reached
            )

    def slim(self, node: ast.AST, name: str) -> None:
        """Plan the rewrite of one node's docstring."""
        expr = getattr(node, "body")[0]  # noqa: B009 - typed as ast.AST
        start = self.offset(expr.lineno, expr.col_offset)
        end = self.offset(expr.end_lineno, expr.end_col_offset)
        column = start - self.starts[expr.lineno - 1]
        new = _slim(self.source[start:end], node, column, name, self.report)
        if new is not None:
            self.edits.append((start, end, new))

    def rename_closure(
        self,
        outer: ast.FunctionDef | ast.AsyncFunctionDef,
        inner: ast.FunctionDef | ast.AsyncFunctionDef,
        stack: tuple[str, ...],
    ) -> bool:
        """Plan renaming ``inner`` to ``_inner`` throughout ``outer``; False if unsafe.

        Unsafe is any other binding of either name in ``outer``, a ``global`` or
        ``nonlocal`` naming it, the name as a string, or a read of ``__name__``.
        """
        old, new = inner.name, f"_{inner.name}"
        both, names, binders = {old, new}, [], 0
        for node in ast.walk(outer):
            if isinstance(node, ast.Name) and node.id in both:
                if node.id == new or not isinstance(node.ctx, ast.Load):
                    return False
                names.append(node)
            elif isinstance(
                node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
            ):
                if node.name == new:
                    return False
                binders += node.name == old
            elif (
                (isinstance(node, ast.Global | ast.Nonlocal) and both & set(node.names))
                or (isinstance(node, ast.alias) and (node.asname or node.name) in both)
                or (isinstance(node, ast.arg) and node.arg in both)
                or (isinstance(node, ast.ExceptHandler) and node.name in both)
                or (
                    isinstance(node, ast.Attribute)
                    and node.attr in {"__name__", "__qualname__"}
                )
                or (isinstance(node, ast.Constant) and node.value in both)
            ):
                return False
        if binders != 1:
            return False
        line = self.lines[inner.lineno - 1]
        found = re.compile(rf"def\s+({old})\b").search(
            line.encode()[inner.col_offset :].decode()
        )
        if found is None:
            return False
        col = len(line.encode()[: inner.col_offset].decode()) + found.start(1)
        at = self.starts[inner.lineno - 1] + col
        self.edits.append((at, at + len(old), new))
        for node in names:
            at = self.offset(node.lineno, node.col_offset)
            self.edits.append((at, at + len(old), new))
        self.renamed.append((stack, new, old))
        return True

    def apply(self) -> str:
        """Return the source with every planned edit applied, last first."""
        text = self.source
        for start, end, new in sorted(self.edits, reverse=True):
            text = text[:start] + new + text[end:]
        return text


def _stripped(tree: ast.Module, renamed: list[tuple[tuple[str, ...], str, str]]) -> str:
    """Return the dump of ``tree`` without docstrings, closure renames undone."""
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                del body[0]
    for stack, new, old in renamed:
        scope: ast.AST = tree
        for part in stack:
            scope = next(
                n
                for n in ast.walk(scope)
                if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                and n.name == part
            )
        for node in ast.walk(scope):
            if isinstance(node, ast.Name) and node.id == new:
                node.id = old
            if (
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                and node.name == new
            ):
                node.name = old
    return ast.dump(tree)


def main(argv: list[str] | None = None) -> int:
    """Slim the docstrings of the given files or directories; return the exit status.

    Parameters
    ----------
    argv : list of str, optional
        The command-line arguments; ``sys.argv[1:]`` by default.

    Returns
    -------
    int
        0, or 1 when ``--check`` finds a changed AST or a file that does not compile.
    """
    parser = argparse.ArgumentParser(prog="docstring_diet", description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true", help="rewrite the files")
    parser.add_argument("--check", action="store_true", help="prove the AST unchanged")
    parser.add_argument("--report", action="store_true", help="print what changed")
    parser.add_argument(
        "--rename-closures",
        action="store_true",
        help="rename validated nested functions to _name",
    )
    args = parser.parse_args(argv)
    files = [
        f for p in args.paths for f in (sorted(p.rglob("*.py")) if p.is_dir() else [p])
    ]
    status, total, out = 0, 0, sys.stdout
    for path in files:
        source = path.read_text(encoding="utf-8")
        diet = _Diet(path, source, args.rename_closures)
        diet.visit(ast.parse(source), (), func=False, seen=True)
        new = diet.apply()
        report = diet.report
        report.removed = source.count("\n") - new.count("\n")
        total += report.removed
        if args.check and new != source:
            try:
                compile(new, str(path), "exec")
                same = _stripped(ast.parse(source), []) == _stripped(
                    ast.parse(new), diet.renamed
                )
            except SyntaxError as error:
                same = False
                out.write(f"CHECK FAILED {path}: {error}\n")
            if not same:
                status = 1
                out.write(f"CHECK FAILED {path}: the stripped AST changed\n")
                continue
        if args.write and new != source:
            path.write_text(new, encoding="utf-8")
        if args.report and (new != source or report.skipped or report.closures):
            out.write(f"{path}: -{report.removed} lines\n")
            for title, items in (
                ("examples dropped", report.examples),
                ("validated closures", report.closures),
                ("skipped", report.skipped),
                ("flattened", report.flattened),
            ):
                out.writelines(f"  {title}: {item}\n" for item in items)
    if args.report:
        out.write(f"total: -{total} lines in {len(files)} files\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
