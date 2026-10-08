#!/usr/bin/env python3
"""Consistency checks on the patch tables in patches.py.

Numpy-free, torch-free, GPU-free: pure AST inspection of the source.

WHY THIS EXISTS
---------------
`_TARGETS`, `_REPLACEMENTS` and the `_ORIGINALS[...]` reads are the only part
of the plugin that runs exactly once, at load, on someone else's machine. A
typo in a key is invisible until then, and the failure arrives as a Wan2GP
plugin-load traceback rather than anything this project can catch. So the
tables are checked against each other, and against the functions they name,
without importing anything.

    python tests/test_install_table.py
"""

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "patches.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

FAILURES = []


def check(label, condition, detail=""):
    print(f"  [{'pass' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def module_assign(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    return None


def defined_functions():
    return {node.name for node in ast.walk(TREE)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def originals_reads():
    """Every literal key read or written as _ORIGINALS["..."]."""
    keys = set()
    for node in ast.walk(TREE):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "_ORIGINALS"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            keys.add(node.slice.value)
    return keys


print(__doc__.strip().splitlines()[0])
print()

targets_node = module_assign("_TARGETS")
replacements_node = module_assign("_REPLACEMENTS")
check("_TARGETS is a module-level literal", targets_node is not None)
check("_REPLACEMENTS is a module-level literal", replacements_node is not None)
if targets_node is None or replacements_node is None:
    sys.exit(1)

targets = ast.literal_eval(targets_node)
replacements = ast.literal_eval(replacements_node)
functions = defined_functions()
reads = originals_reads()

target_keys = [entry[0] for entry in targets]
owner_names = {entry[1] for entry in targets}

print(f"  {len(targets)} targets, {len(replacements)} replacements\n")

print("table shape")
check("every target is a 4-tuple (key, owner, attribute, required)",
      all(len(entry) == 4 for entry in targets))
check("target keys are unique", len(target_keys) == len(set(target_keys)))
check("required flags are booleans",
      all(isinstance(entry[3], bool) for entry in targets))
check("owners are resolved names, not dotted paths",
      all("." not in entry[1] for entry in targets), f"owners {sorted(owner_names)}")
print()

print("targets and replacements agree")
check("every target has a replacement",
      set(target_keys) == set(replacements),
      f"missing {set(target_keys) - set(replacements)}, "
      f"extra {set(replacements) - set(target_keys)}")
for key, name in sorted(replacements.items()):
    check(f"{name} is defined in patches.py", name in functions)
print()

print("runtime reads match the tables")
check("every _ORIGINALS key read is a declared target",
      reads <= set(target_keys), f"undeclared {sorted(reads - set(target_keys))}")
unread = set(target_keys) - reads
check("every declared target is actually used", not unread,
      f"never read: {sorted(unread)}")
print()

print("required targets are the ones the code cannot do without")
required = {entry[0] for entry in targets if entry[3]}
optional = {entry[0] for entry in targets if not entry[3]}
# Video carry and the coordinate correction cannot degrade; audio and Ref2VA can.
check("the video decode, generate, history and plain builder are required",
      {"decode", "generate", "add_video_history", "build_packed_sequence"} <= required,
      f"required {sorted(required)}")
check("audio and Ref2VA are optional",
      {"audio_decode", "encode_audio", "build_ref2va_packed_sequence"} <= optional,
      f"optional {sorted(optional)}")
print()

print("preflight covers every Wan2GP name the runtime needs")


def h3_imports(scope=None):
    """(module, name) pairs imported from models.minimax_h3.* in `scope`."""
    pairs = set()
    tree = scope if scope is not None else TREE
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "models.minimax_h3"):
            for alias in node.names:
                pairs.add((node.module, alias.name))
    return pairs


preflight = next((n for n in ast.walk(TREE)
                  if isinstance(n, ast.FunctionDef) and n.name == "_preflight"), None)
check("_preflight is defined", preflight is not None)

if preflight is not None:
    # Imports read through a helper with a documented fallback do not need to
    # be verified by preflight, because their absence is survivable rather
    # than disqualifying.  Each exemption needs a reason here.
    EXEMPT = {
        # _audio_latent_fps() falls back to 40, which is the rate implied by the
        # autoencoder's 800-sample hop at 32 kHz.
        ("models.minimax_h3.pipeline", "AUDIO_LATENT_FPS"),
    }
    runtime = h3_imports() - h3_imports(preflight) - EXEMPT
    covered = h3_imports(preflight)
    # Names the runtime imports from a module must be verified from that same
    # module by preflight, not probed on a class.  v1.0.0 probed
    # video_latent_frames on MiniMaxH3Pipeline, which never had it.
    missing = {pair for pair in runtime if pair not in covered}
    check("every name imported elsewhere is also imported by _preflight",
          not missing, f"uncovered: {sorted(missing)}")

    # And nothing in preflight may assert an attribute on a class when the
    # runtime takes it from a module namespace.
    module_names = {name for _, name in covered}
    class_probes = set()
    for node in ast.walk(preflight):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and isinstance(node.args[1], ast.Constant)
                and node.args[0].id[:1].isupper()):
            class_probes.add((node.args[0].id, node.args[1].value))
    conflicts = {p for p in class_probes if p[1] in module_names}
    check("no module-level name is probed on a class", not conflicts,
          f"conflicts: {sorted(conflicts)}")
print()

print("install/uninstall symmetry")
src = SOURCE
check("uninstall restores through _ORIGINAL_OWNERS",
      "_ORIGINAL_OWNERS.get(key" in src)
check("install records owners for uninstall", "_ORIGINAL_OWNERS.update" in src)
check("install rolls back on failure", "reversed(applied)" in src)
check("preflight runs before anything is patched",
      src.index("_preflight()") < src.index("setattr(owner, attribute, globals()"))
check("the packing module is resolved once, not hard-imported",
      "from models.minimax_h3.components.packing import" not in src,
      "no hard-coded packing import remains")
print()

if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
