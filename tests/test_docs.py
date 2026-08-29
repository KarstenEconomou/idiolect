"""Test the Python docstring rules."""

import ast
from pathlib import Path


def public_definitions(
    tree: ast.AST,
) -> list[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return public classes and functions from one tree."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]


def test_source_has_docstrings() -> None:
    """Check that source modules and public definitions have docstrings."""
    root = Path(__file__).parents[1] / "src" / "idiolect"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert ast.get_docstring(tree), f"Module has no docstring: {path}"
        for node in public_definitions(tree):
            assert ast.get_docstring(node), (
                f"Public definition has no docstring: {path}:{node.lineno}"
            )
