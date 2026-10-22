#!/usr/bin/env python3
"""Join frame from latents: the frame-0 condition rebuilt from the carried latents.

Runs the real patches.py on the numpy torch shim, with a stand-in decoder and a
recorder in place of the pipeline's _add_image_condition.

WHAT IS BEING CHECKED
---------------------
1. With the option on and a carried history pending, the join frame handed on
   is the last frame of the decoded carried latents, and the pending latents
   are used once only.
2. A decoder that outputs twice the size (the x2 VAE upsampler) is resized to
   the native frame's size.
3. With the option off, for a frame other than 0, for an anchored condition,
   or with nothing pending, Wan2GP's own frame passes through untouched.
4. A rebuilt frame too different from the video's, or a decode that raises,
   falls back to the video's frame.

    python tests/test_latent_anchor.py
"""

import importlib.util
import pathlib
import sys
import types

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import _torch_shim  # noqa: E402

torch = _torch_shim.install()
FAILURES = []


def check(label, condition, detail=""):
    print(f"  [{'pass' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


package = types.ModuleType("h3lc_anchor")
package.__path__ = [str(ROOT)]
sys.modules["h3lc_anchor"] = package
for name in ("colour", "store", "patches"):
    spec = importlib.util.spec_from_file_location(f"h3lc_anchor.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
patches = sys.modules["h3lc_anchor.patches"]
patches._log = lambda message: None


def box_resize(x, size, **_):
    a = np.asarray(x.a)
    n, c, h, w = a.shape
    fh, fw = h // size[0], w // size[1]
    return torch.from_numpy(a.reshape(n, c, size[0], fh, size[1], fw).mean(axis=(3, 5)).astype(np.float32))


torch.nn.functional.interpolate = box_resize

H, W = 8, 12
rng = np.random.default_rng(1)
LAST = rng.uniform(-0.9, 0.9, (3, 1, H, W)).astype(np.float32)
CALLS = []
SCALE = [1]
RAISE = [False]


def fake_decode(vae, z, *args, **kwargs):
    if RAISE[0]:
        raise RuntimeError("decoder unavailable")
    s = SCALE[0]
    frames = rng.uniform(-1, 1, (1, 3, 22, H * s, W * s)).astype(np.float32)
    frames[0, :, -1:] = np.repeat(np.repeat(LAST, s, axis=2), s, axis=3)
    return torch.from_numpy(frames)


patches._ORIGINALS["decode"] = fake_decode
patches._ORIGINALS["add_image_condition"] = (
    lambda self, image, frame_index, *a, **k: CALLS.append((np.asarray(image.a).copy(), frame_index, k)))
pipeline = types.SimpleNamespace(vae=types.SimpleNamespace(_model_dtype=torch.float32), device=None)
carried = torch.from_numpy(np.zeros((1, 24, 7, H, W), np.float32))


def video_frame(offset=0.0):
    return torch.from_numpy((LAST + np.float32(offset)).clip(-1, 1))


def join(frame, frame_index=0, anchor=None, pending=True):
    CALLS.clear()
    patches.STATE.anchor_latents = carried if pending else None
    patches._patched_add_image_condition(pipeline, frame, frame_index, [], [], [],
                                         anchor=anchor, timestep=None)
    return CALLS[0][0]


print("option on")
patches.CONFIG.latent_anchor = True
native = video_frame(0.004)            # the video's frame: the same picture, 8-bit rounded
used = join(native)
check("the join frame is the decoded carried latents' last frame", np.allclose(used, LAST))
check("the pending latents are used once", patches.STATE.anchor_latents is None)
check("counted", patches.STATE.stats["anchor_from_latents"] == 1)
SCALE[0] = 2
used = join(native)
check("a x2 decoder's frame is resized to the video frame's size",
      used.shape == LAST.shape and np.allclose(used, LAST, atol=1e-5), str(used.shape))
SCALE[0] = 1
print()

print("left alone")
check("frame index other than 0", np.array_equal(join(native, frame_index=40), np.asarray(native.a)))
check("an anchored condition", np.array_equal(join(native, anchor="frame"), np.asarray(native.a)))
check("nothing pending", np.array_equal(join(native, pending=False), np.asarray(native.a)))
patches.CONFIG.latent_anchor = False
check("option off", np.array_equal(join(native), np.asarray(native.a)))
patches.CONFIG.latent_anchor = True
print()

print("fall backs")
far = video_frame(0.5)
check("a different picture keeps the video's frame", np.array_equal(join(far), np.asarray(far.a)))
RAISE[0] = True
check("a failed decode keeps the video's frame", np.array_equal(join(native), np.asarray(native.a)))
RAISE[0] = False
check("both counted as the video's", patches.STATE.stats["anchor_native"] == 2,
      str(patches.STATE.stats["anchor_native"]))
print()

if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
