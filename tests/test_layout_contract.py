#!/usr/bin/env python3
"""Layout contract test for Sliding Window Latents.

Proves that a carried latent block lands on the pixel frames it actually
depicts. Runs without Wan2GP, without torch and without a GPU - the grid
arithmetic is the whole subject, and it is pure numpy.

    python test_layout_contract.py

The two grid helpers below are transcribed from
models/minimax_h3/components/packing.py at 362c346 (torch -> numpy; the
arithmetic is unchanged). If a Wan2GP update alters _FRAME_PER_TOKEN,
_FRAME_RESCALE, _video_t_grid or _reference_t_span, re-transcribe them and
re-run: a failure here means the coordinate correction in patches.py no
longer describes the layout it is correcting.

WHAT IS BEING CHECKED
---------------------
Wan2GP places a "history" keyframe block so that it occupies the frames
immediately preceding target frame 0:

    target_origin = text_len + _reference_t_span(history_latents)

Stock is correct because its history block deliberately excludes the join
frame - pipeline.py takes continuation[:, -count:-1], and the join frame
itself is pinned separately at target frame 0 by a "first" anchor.

A carried block cannot exclude it. Latents are sliced whole, and the final
latent of a window always ends on that window's last frame. So a carried
block covers one frame more than the layout reserves for it, and the whole
block is placed one pixel frame earlier than its content really sits.

The correction must therefore move the block one frame LATER relative to
target_origin, and patches.py does exactly that: it adds one frame to the
carried block's own row times and leaves everything else alone.

Moving the target one frame earlier instead would be identical in relative
terms, and is what this did up to 1.0.2 - but it also moved the target
relative to the text rows and relative to the audio conditions, which
displaced the carried audio by a frame. See tests/test_audio_context.py.

The sign matters either way: applying it backwards doubles the error,
taking a 1-frame skew to 2.
"""

import sys

import numpy as np

# --- transcribed from packing.py -----------------------------------------

_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_FRAME_RESCALE = 5.0 / 3.0


def video_t_grid(length, origin, time_scale=1.0):
    spans = np.array(
        [_FRAME_RESCALE * time_scale * _FRAME_PER_TOKEN[i % len(_FRAME_PER_TOKEN)]
         for i in range(length)], dtype=np.float64)
    return origin + np.concatenate(([0.0], np.cumsum(spans[:-1])))


def reference_t_span(length, time_scale=1.0):
    return sum(_FRAME_RESCALE * time_scale * _FRAME_PER_TOKEN[i % len(_FRAME_PER_TOKEN)]
               for i in range(length))


# --- model of the two sides ----------------------------------------------

def layout_frames(n_latents, text_len=0.0):
    """Where build_packed_sequence puts a history block, in target-frame units."""
    target_origin = text_len + reference_t_span(n_latents)
    return np.round((video_t_grid(n_latents, text_len) - target_origin) / _FRAME_RESCALE, 6)


def content_frames(n_latents, includes_join_frame):
    """Where the block's content actually sits, in target-frame units.

    The join frame - the previous window's last frame - is pinned at target
    frame 0 by the "first" anchor, so a previous-window frame k steps before
    it belongs at target frame -k.
    """
    covered = int(round(reference_t_span(n_latents) / _FRAME_RESCALE))
    offsets = np.concatenate(
        ([0], np.cumsum([_FRAME_PER_TOKEN[i % len(_FRAME_PER_TOKEN)]
                         for i in range(n_latents)])[:-1]))
    last = 0 if includes_join_frame else -1
    return (last - (covered - 1)) + offsets


def corrected(n_latents, block_shift_frames):
    """Block position after moving the block itself by block_shift_frames."""
    return layout_frames(n_latents) + block_shift_frames


# --- checks ---------------------------------------------------------------

PHASE_ALIGNED = (7, 12, 17)
FAILURES = []


def check(label, condition, detail=""):
    status = "pass" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


print(__doc__.split("WHAT IS BEING CHECKED")[0].strip().splitlines()[0])
print()

print("stock history block (excludes the join frame)")
for n, frames in ((5, 17), (10, 34)):
    lay, con = layout_frames(n), content_frames(n, includes_join_frame=False)
    check(f"{n} latents / {frames} frames need no correction",
          np.allclose(lay, con), f"layout {lay[:3]}... content {con[:3]}...")
print()

print("carried history block (includes the join frame)")
for n in PHASE_ALIGNED:
    lay, con = layout_frames(n), content_frames(n, includes_join_frame=True)
    error = lay - con
    check(f"n={n:2d} is misplaced by a constant -1 frame",
          np.allclose(error, -1.0),
          f"error {np.unique(error)}")
print()

print("phase alignment: a window holds 2 + 5k latents, so n must be 2 (mod 5)")
for n in PHASE_ALIGNED:
    check(f"n={n:2d} is phase-aligned", n % 5 == 2)
for n in (5, 6, 8, 10):
    check(f"n={n:2d} is correctly rejected", n % 5 != 2)
print()

print("the correction: the carried block moves one frame LATER")
for n in PHASE_ALIGNED:
    con = content_frames(n, includes_join_frame=True)
    check(f"n={n:2d} moving the block +1 frame resolves the skew",
          np.allclose(corrected(n, +1), con))
    check(f"n={n:2d} moving it -1 frame does NOT (the sign matters)",
          not np.allclose(corrected(n, -1), con),
          f"leaves {np.unique(corrected(n, -1) - con)} frame residual")
print()

print("carried block reaches target frame 0, where the 'first' anchor also sits")
for n in PHASE_ALIGNED:
    con = content_frames(n, includes_join_frame=True)
    span = _FRAME_PER_TOKEN[(n - 1) % len(_FRAME_PER_TOKEN)]
    check(f"n={n:2d} final latent covers target frames {con[-1]:.0f}..{con[-1] + span - 1:.0f}",
          con[-1] + span - 1 == 0)
print()

if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)

print("all checks passed")
