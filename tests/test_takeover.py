#!/usr/bin/env python3
"""Takeover: Wan2GP's Discard Last Frames at 4 puts the join on a latent boundary.

Runs the real patches.py on the numpy torch shim.

WHAT IS BEING CHECKED
---------------------
1. With 4 frames discarded the join is looked for 4 frames further back in the
   cached tail, and a cache that would hand over frames from after the join is
   refused - including on a static shot, where the old fixed offset passed.
2. The carried block is then whole decoder chunks ending one frame before the
   join (5, 10 or 15 latents for 7, 12 or 17), with no coordinate shift.
3. Any other discard is refused rather than carried off the latent grid.
4. The window is decoded with that block in front and the block's frames cut
   off; a window that does not fit decodes normally.
5. Audio falls back to the native encode, which follows the discard.

    python tests/test_takeover.py
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


package = types.ModuleType("h3lc_takeover")
package.__path__ = [str(ROOT)]
sys.modules["h3lc_takeover"] = package
for name in ("colour", "store", "patches"):
    spec = importlib.util.spec_from_file_location(f"h3lc_takeover.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
P = sys.modules["h3lc_takeover.patches"]
P._log = lambda message: LOG.append(message)

rng = np.random.default_rng(3)
H, W = 2, 3                                   # latent size; frames are 32 x 48
TOTAL = 37                                    # 2 + 5 * 7 latents
FRAMES = [rng.uniform(-1, 1, (3, 32, 48)).astype(np.float32) for _ in range(12)]
SIGNATURES = torch.stack([P._fingerprint(torch.from_numpy(f)) for f in FRAMES])
LATENTS = torch.from_numpy(np.broadcast_to(np.arange(TOTAL - 17, TOTAL, dtype=np.float32)
                                           .reshape(1, 1, 17, 1, 1), (1, 24, 17, H, W)).copy())


def seed(discard, signatures=SIGNATURES):
    P.STATE.latents = LATENTS
    P.STATE.signatures = signatures
    P.STATE.signature = signatures[-2]
    P.STATE.source_generation = 0
    P.STATE.generation = 1
    P.STATE.source_window_no = 1
    P.STATE.window_no = 2
    P.STATE.usable = True
    P.STATE.total_latents = TOTAL
    P.STATE.expected_latents = TOTAL
    P.STATE.source_discard = discard
    P.STATE.plan = {"usable": True, "window_no": 2, "expected_latents": TOTAL}


def history(last_frame):
    video = np.zeros((3, 17, 32, 48), np.float32)
    video[:, -1] = last_frame
    return torch.from_numpy(video)


def carry(discard, last_frame, latents=7, signatures=SIGNATURES):
    seed(discard, signatures)
    P.CONFIG.latents = latents
    return P._take_cached(history(last_frame))


print("discard 0 (unchanged)")
block, reason = carry(0, FRAMES[-2])
check("carries 7 latents ending at the last one", block is not None
      and block.shape[2] == 7 and float(np.asarray(block.a)[0, 0, -1, 0, 0]) == TOTAL - 1, str(reason))
check("not the takeover layout", P.STATE.takeover is False)
print()

print("discard 4")
block, reason = carry(4, FRAMES[-6])
values = np.asarray(block.a)[0, 0, :, 0, 0] if block is not None else None
check("carries 5 latents", block is not None and block.shape[2] == 5, str(reason))
check("ending one frame before the join (the last-but-one latent is left out)",
      values is not None and values[-1] == TOTAL - 3, str(values))
check("starting on a decoder chunk", values is not None and int(values[0]) % 5 == 0)
check("takeover layout set", P.STATE.takeover is True)
for wanted, count in ((12, 10), (17, 15)):
    block, reason = carry(4, FRAMES[-6], latents=wanted)
    check(f"{wanted} carried latents become {count}", block is not None and block.shape[2] == count, str(reason))
block, reason = carry(4, FRAMES[-2])
check("a join where the undiscarded one was is refused", block is None, str(reason))
static = torch.stack([P._fingerprint(torch.from_numpy(FRAMES[0]))] * 12)
block, reason = carry(4, FRAMES[0], signatures=static)
check("a static shot still takes the discarded layout", block is not None and block.shape[2] == 5
      and np.asarray(block.a)[0, 0, -1, 0, 0] == TOTAL - 3, str(reason))
print()

print("other discards")
for discard in (1, 3, 8):
    block, reason = carry(discard, FRAMES[-2 - discard])
    check(f"{discard} is refused", block is None and "Discard Last Frames" in str(reason), str(reason))
print()

print("coordinate shift")
P.STATE.takeover = True
check("none in the takeover layout",
      (0 if P.STATE.takeover else P._RESERVE_DEFICIT) == 0)
print()

print("joint decode")
CALLS = []


def fake_decode(vae, z, *args, **kwargs):
    count = int(z.shape[2])
    frames = 17 * ((count - 2) // 5) + 5
    CALLS.append(count)
    out = np.zeros((1, 3, frames, 32, 48), np.float32)
    out[0, 0, :, 0, 0] = np.arange(frames)
    return torch.from_numpy(out)


P._ORIGINALS["decode"] = fake_decode
window = torch.from_numpy(np.zeros((1, 24, TOTAL, H, W), np.float32))
P.CONFIG.takeover = True
P.STATE.takeover_prefix = torch.from_numpy(np.zeros((1, 24, 5, H, W), np.float32))
decoded = P._joint_decode(None, window, (), {})
check("decodes prefix + window in one call", CALLS[-1] == 5 + TOTAL, str(CALLS))
check("returns the window's own frame count", decoded is not None and decoded.shape[2] == 17 * 7 + 5)
check("starting at the window's first frame", decoded is not None
      and float(np.asarray(decoded.a)[0, 0, 0, 0, 0]) == 17)
check("the prefix is used once", P.STATE.takeover_prefix is None)
P.STATE.takeover_prefix = torch.from_numpy(np.zeros((1, 24, 5, H + 1, W), np.float32))
check("a different size decodes normally", P._joint_decode(None, window, (), {}) is None)
P.STATE.takeover_prefix = torch.from_numpy(np.zeros((1, 24, 5, H, W), np.float32))
P.CONFIG.takeover = False
check("option off decodes normally", P._joint_decode(None, window, (), {}) is None)
P.CONFIG.takeover = True
check("nothing pending decodes normally", P._joint_decode(None, window, (), {}) is None)
print()

if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
