#!/usr/bin/env python3
"""Join frame from latents: pin the previous window's own latent for the join.

In the takeover layout (Discard Last Frames 4) the join is the previous
window's 1-frame latent.  Runs the real patches.py on the numpy torch shim,
with a stand-in pipeline whose _add_image_condition encodes the frame through
self._encode_video, as Wan2GP's does.

WHAT IS BEING CHECKED
---------------------
1. With the option on and a join latent pending, the condition gets that
   latent instead of the encode, and it is used once.
2. The pipeline's own _encode_video is back in place afterwards, also when the
   condition raises.
3. Frame other than 0, an anchored condition, option off, nothing pending:
   Wan2GP's encode, untouched.
4. A latent of the wrong shape, or too far from the encode, falls back.
5. _take_cached hands over the right latent: the last-but-one of the window.

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
P = sys.modules["h3lc_anchor.patches"]
P._log = lambda message: None

rng = np.random.default_rng(5)
SHAPE = (1, 24, 1, 2, 3)
ENCODED = rng.standard_normal(SHAPE).astype(np.float32)
OWN = (ENCODED + 0.1 * rng.standard_normal(SHAPE)).astype(np.float32)   # same picture
RECEIVED = []
RAISE = [False]


class Pipeline:
    def _encode_video(self, video, keep_all_latents=False):
        return torch.from_numpy(ENCODED.copy())


def fake_add_image_condition(self, image, frame_index, presentation, visual_latents, keyframes,
                             anchor=None, timestep=None):
    latent = self._encode_video(image)
    RECEIVED.append(np.asarray(latent.a).copy())
    if RAISE[0]:
        raise RuntimeError("boom")


P._ORIGINALS["add_image_condition"] = fake_add_image_condition
pipe = Pipeline()
frame = torch.from_numpy(np.zeros((3, 1, 32, 48), np.float32))


def join(pending=OWN, frame_index=0, anchor=None):
    RECEIVED.clear()
    P.STATE.anchor_latents = torch.from_numpy(pending.copy()) if pending is not None else None
    P._patched_add_image_condition(pipe, frame, frame_index, [], [], [], anchor=anchor, timestep=None)
    return RECEIVED[0]


print("option on")
P.CONFIG.latent_anchor = True
check("the join latent replaces the encode", np.allclose(join(), OWN))
check("used once", P.STATE.anchor_latents is None)
check("the pipeline's own _encode_video is back", "_encode_video" not in pipe.__dict__
      and np.allclose(np.asarray(pipe._encode_video(frame).a), ENCODED))
RAISE[0] = True
try:
    join()
except RuntimeError:
    pass
RAISE[0] = False
check("and back after an error", "_encode_video" not in pipe.__dict__)
print()

print("left alone")
check("frame other than 0", np.allclose(join(frame_index=40), ENCODED))
check("anchored condition", np.allclose(join(anchor="frame"), ENCODED))
check("nothing pending", np.allclose(join(pending=None), ENCODED))
P.CONFIG.latent_anchor = False
check("option off", np.allclose(join(), ENCODED))
P.CONFIG.latent_anchor = True
print()

print("fall backs")
check("wrong shape", np.allclose(join(pending=np.zeros((1, 24, 2, 2, 3), np.float32)), ENCODED))
check("a different picture", np.allclose(join(pending=rng.standard_normal(SHAPE).astype(np.float32) * 3),
                                         ENCODED))
print()

print("which latent")
TOTAL = 37
latents = torch.from_numpy(np.broadcast_to(np.arange(TOTAL - 17, TOTAL, dtype=np.float32)
                                           .reshape(1, 1, 17, 1, 1), (1, 24, 17, 2, 3)).copy())
frames = [rng.uniform(-1, 1, (3, 32, 48)).astype(np.float32) for _ in range(12)]
P.STATE.latents = latents
P.STATE.signatures = torch.stack([P._fingerprint(torch.from_numpy(f)) for f in frames])
P.STATE.signature = None
P.STATE.source_generation, P.STATE.generation = 0, 1
P.STATE.source_window_no, P.STATE.window_no = 1, 2
P.STATE.usable = True
P.STATE.total_latents = P.STATE.expected_latents = TOTAL
P.STATE.source_discard = 4
P.STATE.plan = {"usable": True, "window_no": 2, "expected_latents": TOTAL}
P.CONFIG.latents = 7
video = np.zeros((3, 17, 32, 48), np.float32)
video[:, -1] = frames[-6]
block, reason = P._take_cached(torch.from_numpy(video))
check("the join latent is the window's last-but-one", P.STATE.join_latent is not None
      and float(np.asarray(P.STATE.join_latent.a)[0, 0, 0, 0, 0]) == TOTAL - 2, str(reason))
P.STATE.source_discard = 0
video[:, -1] = frames[-2]
P._take_cached(torch.from_numpy(video))
check("none without the discard", P.STATE.join_latent is None)
print()

if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
