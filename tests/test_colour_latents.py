#!/usr/bin/env python3
"""Tests for latent-space colour correction.

Numpy only - no Wan2GP, no torch, no GPU. The claim under test is that a
colour correction expressed in Y'CbCr, collapsed to a 24x24 latent matrix,
produces the intended RGB change when the latents are mapped back through
Wan2GP's latent-to-RGB map.

    python tests/test_colour_latents.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import colour  # noqa: E402

FAILURES = []
rng = np.random.default_rng(11)


def check(label, condition, detail=""):
    print(f"  [{'pass' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


M, BIAS, SOURCE = colour.load_factors()
P = np.linalg.pinv(M)


def sample_latents(n=4000):
    """Latents whose RGB estimate lands inside [0, 1].

    The normalised latent space is roughly unit normal, which maps to signed
    RGB with a standard deviation near 0.73 - so a straight sample runs
    outside the displayable range and clipping would distort the very
    statistics under test. Scaled down, the RGB sits comfortably in range and
    the measurement is testing measurement rather than clipping.
    """
    return rng.normal(0.0, 0.45, (n, M.shape[0]))


def unit_rgb(latents):
    return (colour.latent_to_signed_rgb(latents, M, BIAS) + 1.0) / 2.0


print(f"latent-to-RGB map: {M.shape}, source: {SOURCE}\n")

# ---------------------------------------------------------------- map ----
print("the map itself")
s = np.linalg.svd(M, compute_uv=False)
check("full rank 3 (every RGB delta is reachable)", np.linalg.matrix_rank(M) == 3,
      f"singular values {np.round(s, 4)}")
check("well conditioned", s[0] / s[-1] < 20, f"condition number {s[0] / s[-1]:.2f}")
for name, delta in (("brightness", np.array([.02, .02, .02])),
                    ("warm cast", np.array([.02, .0, -.02])),
                    ("blue lift", np.array([.0, .0, .03]))):
    check(f"{name} delta survives RGB -> latent -> RGB",
          np.allclose((delta @ P) @ M, delta, atol=1e-12))
print()

# ------------------------------------------------------- Y'CbCr basis ----
print("Y'CbCr basis")
check("round trips to RGB", np.allclose(colour.YCBCR_TO_RGB @ colour.RGB_TO_YCBCR,
                                        np.eye(3), atol=1e-12))
grey = np.array([0.5, 0.5, 0.5]) @ colour.RGB_TO_YCBCR.T
check("neutral grey has zero chroma", np.allclose(grey[1:], 0.0, atol=1e-12),
      f"Y={grey[0]:.3f} Cb={grey[1]:.2e} Cr={grey[2]:.2e}")
Q, o = colour.rgb_affine([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
check("identity correction is the identity matrix",
      np.allclose(Q, np.eye(3), atol=1e-12) and np.allclose(o, 0.0, atol=1e-12))
print()

# --------------------------------------------------- the latent affine ----
print("the correction reaches RGB as intended")
latents = sample_latents()
before = unit_rgb(latents)

CASES = {
    "brightness +3%": ([1.0, 1.0, 1.0], [0.03, 0.0, 0.0]),
    "contrast x1.05": ([1.05, 1.0, 1.0], [0.0, 0.0, 0.0]),
    "saturation x0.94": ([1.0, 0.94, 0.94], [0.0, 0.0, 0.0]),
    "warm cast": ([1.0, 1.0, 1.0], [0.0, -0.02, 0.015]),
    "all four at once": ([1.03, 0.96, 0.96], [0.02, -0.01, 0.012]),
}

for label, (gain, offset) in CASES.items():
    A, b = colour.latent_correction(gain, offset, M, BIAS)
    after = unit_rgb(colour.apply_numpy(latents, A, b))

    ycc_before = before @ colour.RGB_TO_YCBCR.T
    ycc_after = after @ colour.RGB_TO_YCBCR.T
    wanted = ycc_before * np.asarray(gain) + np.asarray(offset)
    error = np.abs(ycc_after - wanted).max()
    check(f"{label:20s} lands within 1e-9 of the target", error < 1e-9,
          f"max error {error:.2e}")

A, b = colour.latent_correction([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], M, BIAS)
check("identity correction leaves latents untouched",
      np.allclose(colour.apply_numpy(latents, A, b), latents, atol=1e-10))
check("identity A is the identity matrix", np.allclose(A, np.eye(M.shape[0]), atol=1e-12))
print()

# ------------------------------------------------------- composition ----
print("corrections compose and invert")
A1, b1 = colour.latent_correction([1.0, 1.0, 1.0], [0.03, 0.0, 0.0], M, BIAS)
A2, b2 = colour.latent_correction([1.0, 1.0, 1.0], [-0.03, 0.0, 0.0], M, BIAS)
there_and_back = colour.apply_numpy(colour.apply_numpy(latents, A1, b1), A2, b2)
check("equal and opposite offsets cancel in RGB",
      np.allclose(unit_rgb(there_and_back), before, atol=1e-9),
      f"max RGB drift {np.abs(unit_rgb(there_and_back) - before).max():.2e}")
print()

# ------------------------------------------------------- measurement ----
print("measurement")
ref = colour.ycbcr_stats(before)
drifted = before * 0.97 + 0.012
cur = colour.ycbcr_stats(drifted)

gain, offset, report = colour.measure(ref, cur, strength=1.0)
check("a real drift is measured, not rejected", gain is not None,
      report.get("rejected") or "")
if gain is not None:
    A, b = colour.latent_correction(gain, offset, M, BIAS)
    # Apply to the latents whose RGB is the drifted signal.
    check("measured drift recovers the planted one",
          abs(gain[0] - 1 / 0.97) < 0.01,
          f"luma gain {gain[0]:.4f}, planted inverse {1 / 0.97:.4f}")

    # Correcting the drifted signal should return it to the reference.
    drifted_latents = latents @ colour.latent_correction(
        [0.97, 0.97, 0.97], [0.012, 0.0, 0.0], M, BIAS)[0].T \
        + colour.latent_correction([0.97, 0.97, 0.97], [0.012, 0.0, 0.0], M, BIAS)[1]
    healed = colour.ycbcr_stats(unit_rgb(colour.apply_numpy(drifted_latents, A, b)))
    gap_before = float(np.abs(cur["mean"] - ref["mean"]).max())
    gap_after = float(np.abs(healed["mean"] - ref["mean"]).max())
    check("applying it closes the gap", gap_after < gap_before / 20,
          f"mean gap {gap_before:.5f} -> {gap_after:.2e}")

identical = colour.measure(ref, ref, strength=1.0)
check("no drift measures as a no-op", identical[2]["is_noop"],
      f"gain {np.round(identical[0], 6)}")

cut = colour.ycbcr_stats(before * 0.5)
check("a scene change is rejected, not capped",
      colour.measure(ref, cut)[0] is None,
      colour.measure(ref, cut)[2]["rejected"])

noise = {"mean": np.array([0.05, 0.05, 0.05]), "std": np.array([0.05, 0.05, 0.05])}
_, _, quiet = colour.measure(ref, cur, noise=noise, strength=1.0)
check("drift smaller than the noise floor is held back", quiet["is_noop"],
      f"gain {np.round(quiet['gain'], 6)} offset {np.round(quiet['offset'], 6)}")

_, _, half = colour.measure(ref, cur, strength=0.5)
_, _, full = colour.measure(ref, cur, strength=1.0)
check("strength scales the correction",
      abs(half["offset"][0] - full["offset"][0] / 2) < 1e-9)

_, _, capped = colour.measure(ref, colour.ycbcr_stats(before + 0.05),
                              max_correction=0.01, scene_threshold=1.0)
check("max_correction clamps", max(abs(v) for v in capped["offset"]) <= 0.01 + 1e-12)

_, _, luma_only = colour.measure(ref, cur, axes=("brightness",))
check("axis toggles isolate one axis",
      luma_only["gain"] == [1.0, 1.0, 1.0] and luma_only["offset"][1:] == [0.0, 0.0])
print()

# ------------------------------------------- a cut inside the window ----
print("a cut inside the window withholds the correction")

uniform_head = colour.ycbcr_stats(before)
uniform_tail = colour.ycbcr_stats(before * 0.995 + 0.004)      # ordinary drift
ok, detail = colour.within_window_change(uniform_head, uniform_tail)
check("ordinary within-window drift is allowed through", ok, detail or "")

for label, factor, lift in (("darker scene", 0.55, 0.0),
                            ("brighter scene", 1.0, 0.20),
                            ("colour shift", 1.0, 0.0)):
    shifted = before * factor + lift
    if label == "colour shift":
        shifted = before + np.array([0.10, -0.06, -0.04])
    cut_tail = colour.ycbcr_stats(shifted)
    ok, detail = colour.within_window_change(uniform_head, cut_tail)
    check(f"{label:16s} refused", not ok, (detail or "")[:58])

# A shot change that does not alter the grade needs no guarding.
same_grade = colour.ycbcr_stats(before[rng.permutation(before.shape[0])])
ok, _ = colour.within_window_change(uniform_head, same_grade)
check("a cut that does not change the grade is not refused", ok)

# The threshold is what separates them, and it is configurable.
tight = colour.within_window_change(uniform_head, uniform_tail,
                                    scene_threshold=0.0001)[0]
check("a tighter threshold refuses more", not tight)
print()

# ------------------------------- pixel path and latent path must agree ----
print("the two correction paths agree")


def correct_pixels_numpy(unit, gain, offset):
    """What _correct_pixels does in patches.py, on unit RGB."""
    Q, o = colour.rgb_affine(gain, offset)
    return unit @ Q.T + o


for label, (gain, offset) in CASES.items():
    A, b = colour.latent_correction(gain, offset, M, BIAS)
    via_latents = unit_rgb(colour.apply_numpy(latents, A, b))
    via_pixels = correct_pixels_numpy(before, gain, offset)
    error = np.abs(via_latents - via_pixels).max()
    check(f"{label:20s} same result either way", error < 1e-9,
          f"max RGB difference {error:.2e}")
print()

# -------------------------------------------- continuing a source video ----
print("continuing an existing video")


def continuation(source, seed_reference, windows=4,
                 drift=(0.99, 0.99, 0.99), lift=0.01):
    """The first window of a Continue Video run, with and without a reference.

    `both` scope: the window is rewritten, so each correction is measured
    against an already-corrected predecessor and no total is carried.
    """
    drift_A, drift_b = colour.latent_correction(drift, [lift, 0.0, 0.0], M, BIAS)
    source_stats = colour.ycbcr_stats(unit_rgb(source))

    reference = source_stats if seed_reference else None
    conditioning, trace = source, []
    for _ in range(windows):
        output = colour.apply_numpy(conditioning, drift_A, drift_b)
        head = colour.ycbcr_stats(unit_rgb(output))
        if reference is not None:
            gain, offset, _ = colour.measure(reference, head, strength=1.0)
            if gain is not None:
                A, b = colour.latent_correction(gain, offset, M, BIAS)
                output = colour.apply_numpy(output, A, b)
        corrected = colour.ycbcr_stats(unit_rgb(output))
        trace.append(corrected["mean"][0])
        reference = corrected if reference is not None else None
        conditioning = output
    return source_stats["mean"][0], np.array(trace)


source_clip = sample_latents(2000)
source_luma, unseeded = continuation(source_clip, seed_reference=False)
_, seeded = continuation(source_clip, seed_reference=True)

print(f"    source video luma: {source_luma:.4f}")
print(f"    no reference:      {' '.join(f'{v:.4f}' for v in unseeded)}")
print(f"    seeded reference:  {' '.join(f'{v:.4f}' for v in seeded)}")

first_gap_unseeded = abs(unseeded[0] - source_luma)
first_gap_seeded = abs(seeded[0] - source_luma)
check("without a reference the first window steps away from the source",
      first_gap_unseeded > 0.004, f"step {first_gap_unseeded:.4f}")
check("seeding from the source closes the first join",
      first_gap_seeded < first_gap_unseeded / 10,
      f"{first_gap_unseeded:.4f} -> {first_gap_seeded:.2e}")
check("and later windows stay there",
      abs(seeded[-1] - source_luma) < 0.002,
      f"window 4 gap {abs(seeded[-1] - source_luma):.2e}")
print()

# ------------------------------------------------- an eight-window run ----
print("a simulated eight-window run")


def run(start_latents, windows=8, correct=True,
        drift=(0.995, 0.995, 0.995), lift=0.008):
    """Each window reproduces its conditioning with a small, constant error.

    Without correction that error is inherited window to window and
    compounds. With it, each window's conditioning is pulled back onto the
    previous window's grade before the model ever sees it.

    Both runs must start from the same latents or the comparison is between
    two different samples rather than between two policies.
    """
    drift_A, drift_b = colour.latent_correction(drift, [lift, 0.0, 0.0], M, BIAS)
    tail = start_latents
    opening = colour.ycbcr_stats(unit_rgb(tail))
    reference, total, trace = opening, None, [opening["mean"][0]]

    for _ in range(2, windows + 1):
        conditioning = tail
        if correct and total is not None:
            conditioning = colour.apply_numpy(
                conditioning, *colour.latent_correction(*total, M, BIAS))
        output = colour.apply_numpy(conditioning, drift_A, drift_b)

        head = colour.ycbcr_stats(unit_rgb(output))
        trace.append(head["mean"][0])
        gain, offset, _ = colour.measure(reference, head, strength=1.0)
        if gain is not None:
            total = colour.compose(total, (gain, offset))
        reference, tail = head, output
    return np.array(trace)


seed_latents = sample_latents(2000)
loose = run(seed_latents, correct=False)
held = run(seed_latents, correct=True)
loose_drift = abs(loose[-1] - loose[0])
held_drift = abs(held[-1] - held[0])

print(f"    uncorrected luma: {' '.join(f'{v:.4f}' for v in loose)}")
print(f"    corrected   luma: {' '.join(f'{v:.4f}' for v in held)}")
check("both runs start from the same grade", abs(loose[0] - held[0]) < 1e-12)
check("drift compounds without correction", loose_drift > 0.02,
      f"window 1 -> 8 luma moved {loose_drift:.4f}")
check("correction holds it", held_drift < loose_drift / 5,
      f"{loose_drift:.4f} -> {held_drift:.4f}")
check("uncorrected drift grows monotonically",
      bool(np.all(np.diff(np.abs(loose - loose[0])) >= -1e-9)))
print()

if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
