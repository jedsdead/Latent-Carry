#!/usr/bin/env python3
"""Wan2GP's audio tail is cut off before the audio latents are cached.

Wan2GP v18.19+ decodes H3 audio a few latents past the last video frame
(pipeline.H3_AUDIO_TAIL_LATENTS = 5, ~125 ms). Those latents are after the join,
so caching them would hand the next window post-join audio as its history.

WHAT IS BEING CHECKED
---------------------
1. With a tail, the cached latents end at the join, and the audio signature
   covers the half second ending at the join.
2. Without a tail (older Wan2GP), or with no plan, nothing changes.
3. A difference too big to be a tail is not trimmed.
4. Saved entries without the 2.0.1 marker restore their video but not their
   audio; entries with it restore both.

    python tests/test_audio_tail.py
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


package = types.ModuleType("h3lc_tail")
package.__path__ = [str(ROOT)]
sys.modules["h3lc_tail"] = package
for name in ("colour", "store", "patches"):
    path = ROOT / f"{name}.py"
    if not path.exists():
        continue
    spec = importlib.util.spec_from_file_location(f"h3lc_tail.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
patches = sys.modules["h3lc_tail.patches"]
P = patches
patches._log = lambda message: None

HOP = 800
rng = np.random.default_rng(0)


def decode(latents):
    count = latents.shape[-1]
    return torch.from_numpy(rng.standard_normal((1, 2, count * HOP)).astype(np.float32))


def run(count, target):
    patches.STATE.plan = {"usable": True, "audio_target_latents": target} if target is not None else None
    latents = torch.from_numpy(np.arange(count, dtype=np.float32).reshape(1, 1, count).repeat(2, 1))
    waveform = decode(latents)
    patches._ORIGINALS["audio_decode"] = lambda self, lat, *a, **k: waveform
    out = patches._patched_audio_decode(None, latents)
    cached = np.asarray(patches.STATE.audio_latents.a)
    return out, waveform, cached


def same(a, b):
    return np.allclose(np.asarray(a.a if hasattr(a, "a") else a), np.asarray(b.a if hasattr(b, "a") else b))


print("with Wan2GP's 5-latent tail (360 frames at 24 fps = 600 latents)")
out, waveform, cached = run(605, 600)
check("the decoded waveform is returned untouched", out is waveform)
check("cached latents end at the join (latent 599)", cached[0, 0, -1] == 599, str(cached[0, 0, -1]))
check("the signature covers the half second ending at the join",
      same(patches.STATE.audio_signature, patches._audio_signature(waveform[..., :600 * HOP])))
check("not the half second ending at the tail",
      not same(patches.STATE.audio_signature, patches._audio_signature(waveform)))
print()

print("older Wan2GP, no tail")
_, waveform, cached = run(600, 600)
check("cached latents end at the last latent", cached[0, 0, -1] == 599)
check("signature unchanged", same(patches.STATE.audio_signature, patches._audio_signature(waveform)))
_, waveform, cached = run(605, None)
check("no plan: nothing trimmed", cached[0, 0, -1] == 604)
_, waveform, cached = run(640, 600)
check("a difference too big to be a tail is not trimmed", cached[0, 0, -1] == 639)
print()

print("with Discard Last Frames 4 (120 frames at 24 fps, join at frame 116)")
token = P.JOB.set({"discard_last_frames": 4})
patches.STATE.plan = {"usable": True, "audio_target_latents": 200, "target_frames": 120, "fps": 24.0}
latents = torch.from_numpy(np.arange(205, dtype=np.float32).reshape(1, 1, 205).repeat(2, 1))
waveform = decode(latents)
patches._ORIGINALS["audio_decode"] = lambda self, lat, *a, **k: waveform
patches._patched_audio_decode(None, latents)
cached = np.asarray(patches.STATE.audio_latents.a)
check("cached audio ends at the latent nearest the join (193 1/3 -> 193)", cached[0, 0, -1] == 192,
      str(cached[0, 0, -1]))
join_sample = int(round(116 / 24 * 40 * HOP))
check("the signature ends at the exact join sample",
      same(patches.STATE.audio_signature, patches._audio_signature(waveform[..., :join_sample])))
P.JOB.reset(token)
print()

print("saved entries")
patches.STORE = types.SimpleNamespace(touch=lambda sha: None)
manifest = {"total_latents": 37, "expected_latents": 37, "video_name": "x"}
arrays = {"video": np.zeros((1, 24, 7, 2, 2), np.float32),
          "signatures": np.zeros((3, 16, 16), np.float32),
          "audio": np.ones((1, 2, 64), np.float32),
          "audio_signature": np.ones(32, np.float32)}
patches.STATE.audio_latents = None
patches._seed_from_entry((manifest, arrays), "a" * 64)
check("an entry saved before 2.0.1 restores its video", patches.STATE.latents is not None)
check("but not its audio", patches.STATE.audio_latents is None)
patches._seed_from_entry((dict(manifest, audio_ends_at_join=True), arrays), "b" * 64)
check("a 2.0.1 entry restores its audio", patches.STATE.audio_latents is not None)
patches.STATE.audio_latents = None
patches._seed_from_entry((dict(manifest, audio_ends_at_join=True, discard_last_frames=4), arrays), "c" * 64)
check("a discard entry from before 2.4.0 does not restore its audio", patches.STATE.audio_latents is None)
patches._seed_from_entry((dict(manifest, audio_ends_at_join=True, discard_last_frames=4,
                               audio_cut_at_discard=True), arrays), "d" * 64)
check("a 2.4.0 discard entry restores its audio", patches.STATE.audio_latents is not None)
print()

if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
