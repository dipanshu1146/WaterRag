"""Every regular expression literal in the code base must compile.

The original STP-ID route contained patterns such as ``\\s***\\(**`` that Python's
parser accepts as plain strings but ``re`` rejects at request time, producing
HTTP 500s. This test finds that whole class of bug statically.
"""

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
RE_FUNCTIONS = {"compile", "search", "match", "fullmatch", "findall", "finditer", "sub", "subn", "split"}


def _regex_literals():
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in RE_FUNCTIONS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "re"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                yield pytest.param(node.args[0].value, id=f"{path.name}:{node.lineno}")


@pytest.mark.parametrize("pattern", list(_regex_literals()))
def test_regex_compiles(pattern):
    re.compile(pattern)


def test_no_markdown_artifacts_in_source():
    """'**' and HTML entities inside regex strings are copy/paste corruption."""
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.name.endswith(".bak") or ".backup" in path.name:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "&#x" in line or re.search(r'r["\'].*\*\*\\\(', line):
                offenders.append(f"{path.name}:{number}")
    assert not offenders, offenders
