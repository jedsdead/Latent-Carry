#!/usr/bin/env python3
"""Hold each continuous shot to the look of its own opening (the colour hold).

Runs the real patches.py and colour.py on the numpy torch shim; no GPU, no model.
The drift is built to the measured one: Wan2GP 6479db3, MiniMax H3 Ref2VA, a
continuous shot over 5 windows, where from window 3 each window came out about
3-4 levels darker, about 2% more contrasty and more saturated than the frames
it continued, compounding.

WHAT IS BEING CHECKED
---------------------
1. A shot that continues across windows is held to its opening look - the end
   of every window lands back on it, and the joins stay seamless.
2. The whole frame is measured: the content is an abstract pattern, no subject.
3. A cut at a join starts a new shot with its own reference; the new shot is
   not dragged toward the old one's look.
4. A cut part way through a window: frames before it are held to the old
   shot, frames after it start a new reference.
5. uint8 frames (what current Wan2GP's H3 decode returns) and signed floats get
   the same correction; the carried latents get one correction per latent.
6. A shot carried into Continue Video keeps its reference through the saved
   entry, and a continuation without one holds the continued video's look.

    python tests/test_shot_hold.py
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


package = types.ModuleType("h3lc_shot")
package.__path__ = [str(ROOT)]
sys.modules["h3lc_shot"] = package
for name in ("colour", "patches"):
    spec = importlib.util.spec_from_file_location(f"h3lc_shot.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
patches = sys.modules["h3lc_shot.patches"]
patches._log = lambda message: LOG.append(message)

SIZE = 64
FRAMES = 73
LATENTS = 22
yy, xx = np.mgrid[0:SIZE, 0:SIZE] / (SIZE - 1)
GRAIN = np.random.default_rng(11).standard_normal((SIZE, SIZE))
# Two shots with different structure: a sea-and-sky style horizontal split, and
# a diagonal one.  No subject in either.
SHOT_A = np.stack([90 + 60 * yy, 110 + 50 * yy, 150 - 30 * yy])
SHOT_B = np.stack([60 + 120 * (xx + yy) / 2, 80 + 40 * xx, 70 + 20 * yy])


def frame(base, level=0.0, contrast=1.0, saturation=1.0, grain=3.0, shift=0.0):
    """One uint8 frame [3, H, W]: the base look, then the drift applied."""
    img = np.roll(base, int(shift), axis=2).astype(np.float64)
    mean = img.mean(axis=(1, 2), keepdims=True)
    img = mean + (img - mean) * contrast
    grey = img.mean(axis=0, keepdims=True)
    img = grey + (img - grey) * saturation
    img = img + level + grain * GRAIN[None]
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def window(start, end, base=SHOT_A, frames=FRAMES, cut=None, base2=None, start2=None, end2=None):
    """uint8 [1, 3, F, H, W] drifting from `start` to `end` look (dicts of frame kwargs)."""
    out = np.empty((1, 3, frames, SIZE, SIZE), np.uint8)
    for i in range(frames):
        if cut is not None and i >= cut:
            x = (i - cut) / max(1, frames - 1 - cut)
            look = {k: start2[k] * (1 - x) + end2[k] * x for k in start2}
            out[0, :, i] = frame(base2, shift=i * 0.2, **look)
        else:
            x = i / (frames - 1)
            look = {k: start[k] * (1 - x) + end[k] * x for k in start}
            out[0, :, i] = frame(base, shift=i * 0.2, **look)    # slow pan: motion, not a cut
    return torch.Tensor(out)


CLEAN = dict(level=0.0, contrast=1.0, saturation=1.0, grain=3.0)


def drifted(n):
    """The look after n windows of the measured drift."""
    return dict(level=-3.5 * n, contrast=1.0 + 0.02 * n, saturation=1.0 + 0.07 * n, grain=3.0 * (1 + 0.09 * n))


def fresh(**config):
    patches.STATE.invalidate("test")
    patches.STATE.stats.update({k: 0 for k in patches.STATE.stats})
    settings = dict(colour=True, colour_axes=("brightness", "cast", "contrast", "saturation"),
                    colour_strength=1.0, colour_scene=0.06, colour_max=0.15, refine=False,
                    latents=7, cut_threshold=patches._CUT_MIN)
    settings.update(config)
    for key, value in settings.items():
        setattr(patches.CONFIG, key, value)
    LOG.clear()


def decode(decoded, generation):
    """What the decode hook does around _measure_colour, for consecutive windows."""
    patches.STATE.generation = generation
    patches.STATE.decoding_latents = LATENTS
    out = patches._measure_colour(decoded)
    patches.STATE.signatures = torch.stack([patches._fingerprint(out[0, :, i])
                                            for i in range(out.shape[2] - 8, out.shape[2])])
    patches.STATE.source_generation = generation
    return out


def look(decoded, a, b):
    s = patches._colour_stats(decoded, a, b)
    return np.concatenate([np.asarray(s["mean"]) * 255, np.asarray(s["std"]) * 255])


LABELS = ["Y", "Cb", "Cr", "Ystd", "Cbstd", "Crstd"]

# --------------------------------------------------------------------------
print("a continuous shot across four windows, drifting as measured")
fresh()
opening = None
outs = []
uncorrected_end = None
for n in range(4):
    # Each window of a held shot starts where the last one was held to and
    # drifts by one window's worth; without the hold it would start from the
    # previous window's drifted end.
    raw = window(CLEAN if n == 0 else CLEAN, drifted(1) if n else drifted(0.2))
    out = decode(raw, n + 1)
    outs.append(out)
    if n == 0:
        opening = look(out, 0, 17)
for n, out in enumerate(outs):
    end = look(out, FRAMES - 5, FRAMES)
    error = end - opening
    check(f"window {n + 1}: the end is back on the shot's opening look "
          f"(brightness within 1/255, contrast within 2%)",
          abs(error[0]) < 1.0 and abs(error[3]) < 0.02 * opening[3],
          " ".join(f"{k} {v:+.2f}" for k, v in zip(LABELS, error)))
for n in range(1, 4):
    step = look(outs[n], 0, 1)[:3] - look(outs[n - 1], FRAMES - 1, FRAMES)[:3]
    check(f"join {n}->{n + 1} is seamless (under 1.5/255)", np.abs(step).max() < 1.5,
          np.round(step, 2).tolist())
check("logged as held to the shot's opening",
      any("colour held" in m for m in LOG))
print()

# --------------------------------------------------------------------------
print("uint8 and signed frames, and the carried latents")
fresh()
decode(window(CLEAN, drifted(0.2)), 1)
patches.STATE.decoding_latents = LATENTS
reference = dict(patches.STATE.lock_reference)
out_uint8 = decode(window(drifted(1), drifted(1.2)), 2)
steps = patches.STATE.colour_step
check("uint8 in, uint8 out", out_uint8.dtype == np.uint8)
check(f"one latent correction per carried latent ({patches._cache_depth(LATENTS)})",
      isinstance(steps, list) and len(steps) == patches._cache_depth(LATENTS))
fresh()
decode(window(CLEAN, drifted(0.2)), 1)
signed_in = torch.Tensor(window(drifted(1), drifted(1.2)).a.astype(np.float32) / 127.5 - 1.0)
out_signed = decode(signed_in, 2)
as_uint8 = np.clip((out_signed.a + 1.0) * 127.5, 0, 255)
check("signed frames stay signed and get the same correction (within 1 level)",
      out_signed.dtype == np.float32
      and np.abs(as_uint8 - out_uint8.a.astype(np.float64)).max() <= 1.01)
print()

# --------------------------------------------------------------------------
print("a cut at a join")
fresh()
decode(window(CLEAN, drifted(0.5)), 1)
reference_a = dict(patches.STATE.lock_reference)
raw_b = window(dict(level=-20, contrast=1.0, saturation=1.0, grain=2.0),
               dict(level=-20, contrast=1.0, saturation=1.0, grain=2.0), base=SHOT_B)
out_b = decode(window(dict(level=-20, contrast=1.0, saturation=1.0, grain=2.0),
                      dict(level=-20, contrast=1.0, saturation=1.0, grain=2.0), base=SHOT_B), 2)
check("is detected", any("a cut at the join" in m for m in LOG), " | ".join(LOG)[-160:])
check("the new shot gets its own reference, from its opening",
      patches.STATE.lock_reference["mean"] != reference_a["mean"])
diff = np.abs(out_b.a.astype(int) - raw_b.a.astype(int)).mean()
check("and is not dragged toward the old shot's look (frames left essentially as generated)",
      diff < 1.0, f"mean change {diff:.2f}/255")
print()

# --------------------------------------------------------------------------
print("a cut part way through a window")
fresh()
decode(window(CLEAN, drifted(0.2)), 1)
CUT = 40
raw_c = window(drifted(1), drifted(1.6), cut=CUT, base2=SHOT_B,
               start2=dict(level=-15, contrast=1.0, saturation=1.0, grain=2.0),
               end2=dict(level=-15, contrast=1.0, saturation=1.0, grain=2.0))
out_c = decode(window(drifted(1), drifted(1.6), cut=CUT, base2=SHOT_B,
                      start2=dict(level=-15, contrast=1.0, saturation=1.0, grain=2.0),
                      end2=dict(level=-15, contrast=1.0, saturation=1.0, grain=2.0)), 2)
check(f"is found at frame {CUT}", any(f"cut at frame {CUT}" in m for m in LOG),
      next((m for m in LOG if "colour held" in m), ""))
before_cut = look(out_c, CUT - 5, CUT) - opening if opening is not None else None
fresh_open = look(window(CLEAN, CLEAN), 0, 17)
check("frames before it are held to the old shot's opening (brightness within 1.5/255)",
      abs((look(out_c, CUT - 5, CUT) - fresh_open)[0]) < 1.5,
      f"{(look(out_c, CUT - 5, CUT) - fresh_open)[0]:+.2f}")
after_change = np.abs(out_c.a[:, :, CUT:].astype(int) - raw_c.a[:, :, CUT:].astype(int)).mean()
check("frames after it start the new shot, essentially as generated",
      after_change < 1.0, f"mean change {after_change:.2f}/255")
print()

# --------------------------------------------------------------------------
print("Continue Video")
fresh()
decode(window(CLEAN, drifted(0.3)), 1)
saved = patches.STATE.lock_reference
check("the shot's reference is in what a saved entry records",
      saved is not None and set(saved) == {"mean", "std"})
fresh()
patches.STATE.colour_seed = {"mean": [0.5, 0.0, 0.0], "std": [0.2, 0.05, 0.05]}
decode(window(CLEAN, CLEAN), 1)
check("a continuation without a saved reference holds the continued video's look",
      patches.STATE.lock_reference == {"mean": [0.5, 0.0, 0.0], "std": [0.2, 0.05, 0.05]}
      and any("video being continued" in m for m in LOG))
print()

if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
