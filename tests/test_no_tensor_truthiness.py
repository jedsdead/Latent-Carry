#!/usr/bin/env python3
"""Guard against using a tensor in a boolean context.

Numpy only - no Wan2GP, no torch, no GPU. Pure AST inspection of the source.

WHY THIS EXISTS
---------------
`_patched_decode` contained:

    decoded = _measure_colour(decoded) or decoded

`or` evaluates `bool()` on its left operand, and a torch tensor with more than
one element raises `RuntimeError: Boolean value of Tensor with more than one
element is ambiguous`. The surrounding `except Exception` then caught it,
logged "colour measurement failed, carrying uncorrected", and disabled colour
correction. Every window. The feature never ran once, and the only outward sign
was a log line that read like a cautious fallback rather than a defect.

That is the dangerous shape: a broad `except` around code that must not fail
turns a hard error into a plausible-looking message. So rather than test the
one line, this forbids the idiom across the plugin - any tensor-valued name or
call appearing where Python will call `bool()` on it.

Comparisons are fine (`if tail is None`, `if decoded.shape[2] >= 2`); it is the
bare value that is not.

    python tests/test_no_tensor_truthiness.py
"""

import ast
import pathlib
import sys

FILES = ("patches.py", "colour.py", "plugin.py")

# Names that hold tensors in this plugin.
TENSOR_NAMES = {
    "decoded", "corrected", "latents", "tail", "video", "frames", "sample",
    "incoming", "signature", "signatures", "waveform", "pooled", "block",
    "moved", "unit", "probe", "out", "history_video", "condition", "matrix",
    "shift", "values01", "frame01",
}

# Functions in this plugin that return tensors.
TENSOR_CALLS = {
    "_measure_colour", "_correct_pixels_ramp", "_fingerprint", "_to_unit",
    "_colour_stats", "apply_torch", "apply_numpy",
    "latent_to_signed_rgb",
}

# Attribute calls that are safe to test directly.
SAFE_ATTRS = {"any", "all", "item", "numel", "get", "keys", "values", "items"}

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAILURES = []


def describe(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return f"{func.id}()"
        if isinstance(func, ast.Attribute):
            return f"...{func.attr}()"
    return None


def is_tensorish(node):
    """Would Python call bool() on something tensor-valued here?"""
    if isinstance(node, ast.Name):
        return node.id in TENSOR_NAMES
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id in TENSOR_CALLS
        if isinstance(func, ast.Attribute):
            return func.attr in TENSOR_CALLS and func.attr not in SAFE_ATTRS
    return False


class Visitor(ast.NodeVisitor):
    def __init__(self, path):
        self.path = path

    def flag(self, node, context):
        name = describe(node)
        FAILURES.append(f"{self.path}:{node.lineno}  {name} used in {context}")

    def visit_BoolOp(self, node):
        # Every operand but the last is bool()-tested by `and` / `or`.
        for operand in node.values[:-1]:
            if is_tensorish(operand):
                op = "or" if isinstance(node.op, ast.Or) else "and"
                self.flag(operand, f"`{op}` (calls bool())")
        self.generic_visit(node)

    def _test(self, node, context):
        if is_tensorish(node.test):
            self.flag(node.test, context)

    def visit_If(self, node):
        self._test(node, "`if` test")
        self.generic_visit(node)

    def visit_While(self, node):
        self._test(node, "`while` test")
        self.generic_visit(node)

    def visit_IfExp(self, node):
        self._test(node, "conditional expression")
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not) and is_tensorish(node.operand):
            self.flag(node.operand, "`not` (calls bool())")
        self.generic_visit(node)


print(__doc__.strip().splitlines()[0])
print()

checked = 0
for name in FILES:
    path = ROOT / name
    if not path.exists():
        print(f"  [skip] {name} not present")
        continue
    Visitor(name).visit(ast.parse(path.read_text(encoding="utf-8")))
    checked += 1
    print(f"  scanned {name}")

print()
if FAILURES:
    print(f"{len(FAILURES)} tensor value(s) in a boolean context:")
    for line in FAILURES:
        print(f"  - {line}")
    print("\nUse an explicit comparison instead, e.g.")
    print("    result = f(x)")
    print("    if result is not None:")
    print("        x = result")
    sys.exit(1)

print(f"all checks passed ({checked} files)")
