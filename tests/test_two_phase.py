#!/usr/bin/env python3
"""Two Phases: phase 1 carries the previous window's own phase 1 latents.

Runs the real patches.py on the numpy torch shim.

WHAT IS BEING CHECKED
---------------------
1. Phase 1 (half or two-thirds size) carries from the cached phase 1 latents,
   in the takeover layout when 4 frames were discarded, and pins its own join
   latent; phase 2 still carries the full-size latents.
2. Phase 1 latents from another window are not used.
3. The extended audio context is declined with Two Phases with Tiling, where
   phase 2 carries no video block to shift around it.

    python tests/test_two_phase.py
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
LOG = []


def check(label, condition, detail=""):
    print(f"  [{'pass' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


package = types.ModuleType("h3lc_two")
package.__path__ = [str(ROOT)]
sys.modules["h3lc_two"] = package
for name in ("colour", "store", "patches"):
    spec = importlib.util.spec_from_file_location(f"h3lc_two.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
P = sys.modules["h3lc_two.patches"]
P._log = lambda message: LOG.append(message)

rng = np.random.default_rng(7)
TOTAL = 37
FRAMES = [rng.uniform(-1, 1, (3, 32, 48)).astype(np.float32) for _ in range(12)]


def tail(h, w, base):
    return torch.from_numpy(np.broadcast_to((base + np.arange(TOTAL - 17, TOTAL, dtype=np.float32))
                                            .reshape(1, 1, 17, 1, 1), (1, 24, 17, h, w)).copy())


def seed(discard, p1_generation=0):
    P.STATE.latents = tail(6, 9, 0)                    # full size: 96 x 144
    P.STATE.p1_latents = tail(4, 6, 1000)              # phase 1: 64 x 96
    P.STATE.p1_generation = p1_generation
    P.STATE.signatures = torch.stack([P._fingerprint(torch.from_numpy(f)) for f in FRAMES])
    P.STATE.signature = None
    P.STATE.source_generation, P.STATE.generation = 0, 1
    P.STATE.source_window_no, P.STATE.window_no = 1, 2
    P.STATE.usable = True
    P.STATE.total_latents = P.STATE.expected_latents = TOTAL
    P.STATE.source_discard = discard
    P.STATE.plan = {"usable": True, "window_no": 2, "expected_latents": TOTAL}
    P.CONFIG.latents = 7


def history(h, w, last):
    video = np.zeros((3, 17, h, w), np.float32)
    k = h // last.shape[1]                     # an exact enlargement keeps the fingerprint
    video[:, -1] = last.repeat(k, axis=1).repeat(k, axis=2)
    return torch.from_numpy(video)


def first(block):
    return float(np.asarray(block.a)[0, 0, -1, 0, 0])


print("phase 1, discard 4")
seed(4)
block, reason = P._take_cached(history(64, 96, FRAMES[-6]))
check("carries 5 phase 1 latents", block is not None and block.shape[2] == 5
      and tuple(block.shape[-2:]) == (4, 6), str(reason))
check("from the phase 1 cache, ending one frame before the join",
      block is not None and first(block) == 1000 + TOTAL - 3)
check("its join latent is phase 1's", P.STATE.join_latent is not None
      and tuple(P.STATE.join_latent.shape[-2:]) == (4, 6)
      and float(np.asarray(P.STATE.join_latent.a)[0, 0, 0, 0, 0]) == 1000 + TOTAL - 2)
print()

check("marked as a phase 1 stage", P.STATE.p1_stage is True)
print()

print("phase 2, discard 4")
block, reason = P._take_cached(history(96, 144, FRAMES[-6]))
check("carries the full-size latents", block is not None and tuple(block.shape[-2:]) == (6, 9)
      and first(block) == TOTAL - 3, str(reason))
check("not marked as phase 1", P.STATE.p1_stage is False)
print()

print("phase 1, no discard")
seed(0)
block, reason = P._take_cached(history(64, 96, FRAMES[-2]))
check("carries 7 phase 1 latents ending at the last", block is not None and block.shape[2] == 7
      and first(block) == 1000 + TOTAL - 1, str(reason))
print()

print("phase 1 latents from another window")
seed(4, p1_generation=-5)
block, reason = P._take_cached(history(64, 96, FRAMES[-6]))
check("are not used", block is None and "resolution mismatch" in str(reason), str(reason))
print()

print("audio context with tiled phase 2")
P.CONFIG.audio_context = 2.0
P.CONFIG.video = True
P.STATE.active_latents = 5
P.STATE.plan = {"usable": True, "tiled_phase_2": True}
check("declined", P._audio_extension(30, 80) == 0)
P.STATE.plan = {"usable": True, "tiled_phase_2": False}
check("applied untiled", P._audio_extension(30, 80) == 50)
print()

if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
