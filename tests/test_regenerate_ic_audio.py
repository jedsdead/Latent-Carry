#!/usr/bin/env python3
"""Regenerate 2K IC phase 2: stale audio keyframes are dropped.

Runs the real patches.py on the numpy torch shim.

WHAT IS BEING CHECKED
---------------------
Wan2GP's Regenerate 2K IC phase 2 sets payload["cond_audio_rows"] = None but
keeps phase 1's audio keyframes, so the packer reserves rows nobody feeds
("index_copy_(): Number of indices (1376) should be equal to source.size(dim)
(1216)" on a Continue Video).

1. The layout guard clears audio keyframes that have no condition rows, and the
   cached layout with them, before the native _layout sees the payload.
2. It leaves a payload alone when the rows are there, or when there are no
   audio keyframes.
3. The extended audio context does not shift a layout that has no audio
   history block (the shift has nothing to make room for).

    python tests/test_regenerate_ic_audio.py
"""

import importlib.util
import pathlib
import sys
import types

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import _torch_shim  # noqa: E402

torch = _torch_shim.install()
FAILURES = []
LOG = []


def check(label, condition, detail=""):
    print(f"  [{'pass' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


package = types.ModuleType("h3lc_regen")
package.__path__ = [str(ROOT)]
sys.modules["h3lc_regen"] = package
for name in ("colour", "store", "patches"):
    spec = importlib.util.spec_from_file_location(f"h3lc_regen.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
P = sys.modules["h3lc_regen.patches"]
P._log = lambda message: LOG.append(message)

SEEN = []


def native_layout(self, text_tags, latent_t, latent_h, latent_w, audio_t, payload):
    SEEN.append(payload.get("audio_keyframes"))
    return "layout"


P._ORIGINALS["layout"] = native_layout

print("1. stale audio keyframes are dropped")
payload = {"cond_audio_rows": None, "layout": "old", "layout_signature": (1,), "rope": "r",
           "audio_keyframes": [{"anchor": "history", "latent_frame_count": 78},
                               {"anchor": "first", "latent_frame_count": 2}]}
result = P._patched_layout(object(), None, 107, 50, 120, 608, payload)
check("native _layout sees no audio keyframes", SEEN[-1] is None, repr(SEEN[-1]))
check("cached layout cleared", not any(k in payload for k in ("layout", "layout_signature", "rope")))
check("result passed through", result == "layout")
check("logged the 80 dropped latents", any("dropped 80" in m for m in LOG), repr(LOG[-1:]))

print("2. a consistent payload is untouched")
rows = object()
keyframes = [{"anchor": "history", "latent_frame_count": 28}]
payload = {"cond_audio_rows": rows, "audio_keyframes": keyframes, "layout": "kept"}
P._patched_layout(object(), None, 107, 50, 120, 608, payload)
check("audio keyframes kept when rows exist", SEEN[-1] is keyframes)
check("cached layout kept", payload.get("layout") == "kept")
payload = {"cond_audio_rows": None, "audio_keyframes": None, "layout": "kept"}
P._patched_layout(object(), None, 107, 50, 120, 608, payload)
check("no audio keyframes, nothing to do", payload.get("layout") == "kept")

print("3. no audio history, no extension shift")
import numpy as np  # noqa: E402

TEXT, CARRIED, ROWS_PER_FRAME = 4, 7, 2
P._PACKING = types.SimpleNamespace(_unpack_keyframe_anchor=lambda a: tuple(a) + (None,) * (3 - len(a)))
P.STATE.active_latents = CARRIED
P.STATE.takeover = True                  # no block translation, isolates the audio shift
P.STATE.audio_extra = 50
length = TEXT + CARRIED * ROWS_PER_FRAME + 10
sequence = types.SimpleNamespace(position_ids=torch.from_numpy(
    np.arange(length * 3, dtype=np.float64).reshape(length, 3)))
before = sequence.position_ids[:, 0].numpy().copy()
P._apply_origin_shift(sequence, TEXT, [("history", CARRIED, None)], (), ROWS_PER_FRAME, 1.0)
after = sequence.position_ids[:, 0].numpy()
check("positions unchanged without an audio history", np.array_equal(before, after),
      f"max moved {np.abs(after - before).max()}")
P.STATE.audio_extra = 0
P.STATE.active_latents = None
P.STATE.takeover = False

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s)")
    sys.exit(1)
print("all checks passed")
