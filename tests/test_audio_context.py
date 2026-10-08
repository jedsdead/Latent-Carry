#!/usr/bin/env python3
"""Layout geometry for the coordinate correction and extended audio context.

Numpy only - no Wan2GP, no torch, no GPU. Models the row and time layout that
build_packed_sequence produces and the shifts patches.py applies on top, so the
geometry can be checked without a GPU.

THE SHARED TIME AXIS
--------------------
One time unit is one audio latent. Wan2GP's audio autoencoder has a hop of 800
samples at 32 kHz (models/minimax_h3/components/audio_autoencoder.py), so audio
latents run at 40/s; _FRAME_RESCALE is 5/3, which is 40/24. At 24 fps a video
frame is 5/3 units and an audio latent is exactly 1.

WHAT IS BEING CHECKED
---------------------
1. The coordinate correction moves the carried video block one frame later
   rather than moving the target one frame earlier. The two are identical in
   relative terms; the difference is the side effects. Moving the target also
   moved it relative to the text rows and relative to the audio conditions.
   Exempting the audio history from that displaced the audio by a frame - 41.7
   ms at 24 fps - and pushed its last latent past target_origin; including it
   pushed the block below text_len, where the text rows live. Moving the block
   has neither problem.

2. Extended audio context. Wan2GP sizes the audio condition from the video
   overlap, so overlap 18 at 24 fps gives 0.75s. Carrying a longer tail
   lengthens the history block, but the builder lays it forward from
   float(text_len), so it would overrun target_origin. Everything else moves
   later by the same amount instead, leaving the audio history's start put.

    python tests/test_audio_context.py
"""

import sys

import numpy as np

FPT = (1, 4, 4, 4, 4)
RES = 5.0 / 3.0
AUDIO_CHANNELS = 2
ALR = 40.0

FAILURES = []


def check(label, condition, detail=""):
    print(f"  [{'pass' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def span(n):
    return sum(RES * FPT[i % 5] for i in range(n))


def grid(n, origin):
    spans = np.array([RES * FPT[i % 5] for i in range(n)])
    return origin + np.concatenate(([0.0], np.cumsum(spans[:-1])))


def layout(text_len, carried, audio_history, audio_boundary, target_frames,
           rows_per_frame, extra=0, correct=True):
    """Row times after both shifts.

    `carried` is the carried video latent count; `audio_history` the native
    audio history latents, which the extension lengthens by `extra`.
    """
    history = audio_history + extra
    target_origin = text_len + span(carried)

    t = {}
    t["text"] = np.arange(text_len, dtype=float)
    t["video_history"] = np.repeat(grid(carried, float(text_len)), rows_per_frame)
    t["audio_history"] = np.repeat(float(text_len) + np.arange(history), AUDIO_CHANNELS)
    t["audio_first"] = np.repeat(np.full(audio_boundary, target_origin), AUDIO_CHANNELS)
    t["target"] = grid(target_frames, target_origin)
    origin = target_origin

    # shift 1: the carried block moves one frame later.  Nothing else moves.
    if correct:
        t["video_history"] = t["video_history"] + RES

    # shift 2: everything from text_len moves later by `extra`; the audio
    # history is then exempted, so its start stays put.
    if extra:
        for key in ("video_history", "audio_history", "audio_first", "target"):
            t[key] = t[key] + extra
        t["audio_history"] = t["audio_history"] - extra
        origin = origin + extra

    t["_origin"] = origin
    t["_history_latents"] = history
    return t


print(__doc__.strip().splitlines()[0])
print()

TEXT, N, AH, AB, TF, RPF = 64, 7, 28, 2, 60, 6   # n=7 carried, overlap 18 @ 24 fps
CONTENT = ("video_history", "audio_history", "audio_first", "target")
last_len = RES * FPT[(N - 1) % 5]

print("the shared time axis")
check("a video frame is 5/3 units and equals 40/24",
      abs(RES - 40.0 / 24.0) < 1e-12, f"{RES:.6f}")
check("n=7 covers 22 frames", abs(span(N) / RES - 22) < 1e-9, f"{span(N) / RES:.0f}")
print()

raw = layout(TEXT, N, AH, AB, TF, RPF, correct=False)
base = layout(TEXT, N, AH, AB, TF, RPF)

print("the coordinate correction")
check("uncorrected, the block ends exactly at target_origin",
      abs((raw["video_history"].max() + last_len) - raw["_origin"]) < 1e-9,
      "its frames are target -22..-1")
check("corrected, the block ends one frame past target_origin",
      abs((base["video_history"].max() + last_len) - (base["_origin"] + RES)) < 1e-9,
      "its frames are target -21..0, reaching the join frame")
check("the block-to-target offset changes by exactly one frame",
      abs(((base["_origin"] - base["video_history"].min())
           - (raw["_origin"] - raw["video_history"].min())) + RES) < 1e-9)
check("audio-to-target is left exactly as Wan2GP built it",
      abs((base["_origin"] - base["audio_history"].max())
          - (raw["_origin"] - raw["audio_history"].max())) < 1e-9,
      "no 41.7 ms displacement")
check("the audio history does not overrun the target",
      base["audio_history"].max() < base["_origin"],
      f"last audio row {base['audio_history'].max() - TEXT:+.3f}, "
      f"origin {base['_origin'] - TEXT:+.3f}")
check("nothing lands below text_len",
      min(base[k].min() for k in CONTENT) >= TEXT - 1e-9)
print()

print("extended audio context: 2 seconds")
EXTRA = int(round(2.0 * ALR)) - (AH + AB)
ext = layout(TEXT, N, AH, AB, TF, RPF, extra=EXTRA)


def reach(t):
    return (t["_origin"] - t["audio_history"].min()) / ALR


check("the extension is 50 latents", EXTRA == 50, f"{EXTRA}")
check("the history block is lengthened, not merely moved",
      ext["_history_latents"] == AH + EXTRA, f"{ext['_history_latents']} latents")
check("audio context grows by the extension",
      abs((reach(ext) - reach(base)) - EXTRA / ALR) < 1e-9,
      f"{reach(base):.3f}s -> {reach(ext):.3f}s")
check("audio history still starts at text_len",
      abs(ext["audio_history"].min() - TEXT) < 1e-9)
check("nothing lands below text_len",
      min(ext[k].min() for k in CONTENT) >= TEXT - 1e-9)
check("no content collides with the text rows",
      min(ext[k].min() for k in CONTENT) >= ext["text"].max(),
      f"text ends {ext['text'].max() - TEXT:+.0f}, content starts "
      f"{min(ext[k].min() for k in CONTENT) - TEXT:+.0f}")
print()

print("every other relative distance is preserved")
for label, f in (("video history to target", lambda t: t["_origin"] - t["video_history"].min()),
                 ("audio history end to target", lambda t: t["_origin"] - t["audio_history"].max()),
                 ("audio boundary to target", lambda t: t["audio_first"][0] - t["_origin"]),
                 ("block end to target",
                  lambda t: t["video_history"].max() + last_len - t["_origin"])):
    check(f"{label} unchanged", abs(f(ext) - f(base)) < 1e-9,
          f"{f(base):+.3f} -> {f(ext):+.3f}")
check("target internal spacing unchanged",
      np.allclose(np.diff(ext["target"]), np.diff(base["target"])))
print()

print("the audio now reaches back beyond the video")
lead = ext["video_history"].min() - ext["audio_history"].min()
check("by the extension plus the one-frame correction",
      abs(lead - (EXTRA + RES)) < 1e-9,
      f"{lead / ALR:.3f}s of audio-only context")
print()

print("a range of settings")
for seconds in (0.5, 1.0, 2.0, 4.0):
    extra = max(0, int(round(seconds * ALR)) - (AH + AB))
    out = layout(TEXT, N, AH, AB, TF, RPF, extra=extra)
    ok = (min(out[k].min() for k in CONTENT) >= TEXT - 1e-9
          and abs((out["_origin"] - out["video_history"].min())
                  - (base["_origin"] - base["video_history"].min())) < 1e-9
          and abs((out["_origin"] - out["audio_history"].max())
                  - (base["_origin"] - base["audio_history"].max())) < 1e-9)
    check(f"{seconds:>4}s requested -> {reach(out):.3f}s of audio history, video intact", ok)
print()

print("zero extension is a no-op")
check("identical to the unextended layout",
      all(np.allclose(layout(TEXT, N, AH, AB, TF, RPF, extra=0)[k], base[k]) for k in CONTENT))
print()

if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
