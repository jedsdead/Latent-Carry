#!/usr/bin/env python3
"""Other plugins can find the native bindings behind this plugin's replacements.

Loads patches.py against stubbed Wan2GP modules (no torch, no GPU), runs the
real install() and uninstall(), and checks the __wrapped__ contract.

WHY THIS EXISTS
---------------
The replacements reach Wan2GP's originals through _ORIGINALS["..."], which is a
global lookup nothing outside this module can see.  H3 Latent Continue edits
the native MiniMaxH3Pipeline.generate body in place and locates it by following
__wrapped__ and closure cells from the class attribute.  Before 1.2.3 it found
only _patched_generate and refused with "no reachable native method contains
the required latent insertion points".

WHAT IS BEING CHECKED
---------------------
1. After install(), every replacement's __wrapped__ is the exact object it
   displaced, and inspect.unwrap() walks from the patched attribute through
   Wan2GP's own decorator stack down to the undecorated native body.
2. The native body is still the object in _ORIGINALS, so an in-place edit made
   by another plugin is what this plugin delegates to.
3. The replacements keep their own __name__, so logs still name this plugin.
4. uninstall() and install()'s rollback path remove __wrapped__, leaving no
   stale link to a binding that is no longer patched.

    python tests/test_interop_wrapped.py
"""

import functools
import importlib.util
import inspect
import pathlib
import sys
import types

FAILURES = []


def check(label, condition, detail=""):
    print(f"  [{'pass' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


# --------------------------------------------------------------------------
# Wan2GP stand-ins
# --------------------------------------------------------------------------

def _decorator(method):
    """Stands in for _return_none_on_interrupt / generation_progress / inference_mode."""
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        return method(self, *args, **kwargs)
    return wrapped


def _install_stubs():
    torch = types.ModuleType("torch")
    torch.uint8 = "uint8"
    torch.float32 = "float32"
    torch.nn = types.SimpleNamespace(functional=types.SimpleNamespace())
    sys.modules.setdefault("torch", torch)

    for name in ("models", "models.minimax_h3", "models.minimax_h3.components",
                 "shared", "shared.utils"):
        sys.modules.setdefault(name, types.ModuleType(name))

    class MiniMaxH3Pipeline:
        @_decorator
        @_decorator
        @_decorator
        def generate(self, input_prompt, seed=0):
            return ("native", input_prompt)

        def _add_video_history(self, video, visual_latents, keyframes):
            return None

        def _encode_audio(self, waveform):
            return None

    pipeline = types.ModuleType("models.minimax_h3.pipeline")
    pipeline.MiniMaxH3Pipeline = MiniMaxH3Pipeline
    pipeline._as_video = lambda video: video
    pipeline.video_latent_frames = lambda n: 2 + ((n - 5) // 17) * 5
    sys.modules[pipeline.__name__] = pipeline

    transformer = types.ModuleType("models.minimax_h3.transformer")
    transformer.build_packed_sequence = lambda *a, **k: None
    transformer.build_ref2va_packed_sequence = lambda *a, **k: None
    sys.modules[transformer.__name__] = transformer

    video_vae = types.ModuleType("models.minimax_h3.video_vae")
    video_vae.MiniMaxH3VideoVAE = type("MiniMaxH3VideoVAE", (), {"decode": lambda self, x: x})
    sys.modules[video_vae.__name__] = video_vae

    audio_vae = types.ModuleType("models.minimax_h3.audio_vae")
    audio_vae.MiniMaxH3AudioVAE = type("MiniMaxH3AudioVAE", (), {"decode": lambda self, x: x})
    sys.modules[audio_vae.__name__] = audio_vae

    packing = types.ModuleType("models.minimax_h3.components.packing")
    for helper in ("_unpack_keyframe_anchor", "_frame_grid", "_video_t_grid", "_reference_t_span"):
        setattr(packing, helper, lambda *a, **k: None)
    packing._FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    packing._FRAME_RESCALE = 5.0 / 3.0
    packing.MINIMAX_H3_AUDIO_CHANNELS = 2
    sys.modules[packing.__name__] = packing

    scheduler = types.ModuleType("shared.utils.frame_scheduler")
    scheduler.floor_frame_count = scheduler.normalize_frame_count = lambda *a, **k: 0
    scheduler.normalize_overlap = lambda *a, **k: (0, None)
    sys.modules[scheduler.__name__] = scheduler
    return MiniMaxH3Pipeline, transformer


def _load_patches():
    path = pathlib.Path(__file__).resolve().parent.parent / "patches.py"
    spec = importlib.util.spec_from_file_location("swl_patches_interop", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


print(__doc__.strip().splitlines()[0])
print()

Pipeline, transformer = _install_stubs()
native_attr = Pipeline.__dict__["generate"]
native_body = inspect.unwrap(native_attr)
patches = _load_patches()
patches._log = lambda *a, **k: None

print("install links every replacement to what it displaced")
check("install() succeeds against the stubs", patches.install() is True)
for key, name in sorted(patches._REPLACEMENTS.items()):
    if key not in patches._ORIGINALS:
        continue
    replacement = getattr(patches, name)
    check(f"{name}.__wrapped__ is the displaced binding",
          getattr(replacement, "__wrapped__", None) is patches._ORIGINALS[key])
    check(f"{name} keeps its own __name__", replacement.__name__ == name)
print()

print("the native generate body is reachable from the class attribute")
patched = Pipeline.__dict__["generate"]
check("the class attribute is the replacement", patched is patches._patched_generate)
check("inspect.unwrap() reaches the undecorated native body",
      inspect.unwrap(patched) is native_body)
check("_ORIGINALS still holds the decorated native binding",
      patches._ORIGINALS["generate"] is native_attr)
check("inspect.signature() now reports the native parameters",
      list(inspect.signature(patched).parameters)[:2] == ["self", "input_prompt"])
check("the packing builder unwraps to transformer's native binding",
      inspect.unwrap(transformer.build_packed_sequence)
      is patches._ORIGINALS["build_packed_sequence"])
print()

print("uninstall removes the links")
patches.uninstall()
check("generate is restored", Pipeline.__dict__["generate"] is native_attr)
check("no replacement keeps a stale __wrapped__",
      not any(hasattr(getattr(patches, name), "__wrapped__")
              for name in patches._REPLACEMENTS.values()))
print()

print("a failed install rolls the links back too")
original_setattr = setattr
calls = {"n": 0}


def failing_setattr(owner, attribute, value):
    calls["n"] += 1
    if calls["n"] == 3:
        raise RuntimeError("simulated failure")
    return original_setattr(owner, attribute, value)


patches.setattr = failing_setattr        # module global shadows the builtin
try:
    check("install() reports failure", patches.install() is False)
finally:
    del patches.setattr
check("nothing is left patched", Pipeline.__dict__["generate"] is native_attr)
check("no replacement keeps a stale __wrapped__ after rollback",
      not any(hasattr(getattr(patches, name), "__wrapped__")
              for name in patches._REPLACEMENTS.values()))
print()

if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
