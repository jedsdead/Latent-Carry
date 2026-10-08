#!/usr/bin/env python3
"""Extended audio context on Ref2VA: does anything land on anything else?

test_audio_context.py does this for FL2VA.  Ref2VA differs in a way that matters
here: build_ref2va_packed_sequence walks a time cursor across the reference
tokens first, from float(text_len) up to time_cursor, and only then places the
history block at history_time = time_cursor and the target at
time_cursor + _reference_t_span(history_frames).  So the span below the history
is occupied rather than vacant, and "move everything except the audio history
later by extra" walks the references into the gap it just opened.

This models the builder's time axis directly from packing.py and checks
occupancy, rather than checking that the code does what it says.  Three layouts
are compared:

  native      extra = 0, nothing moved
  naive       extra > 0, everything from text_len moved (the 1.2.1 behaviour,
              which is why Ref2VA was refused outright)
  corrected   extra > 0, the reference span held still

The naive layout must show a collision and the corrected one must not, since a
test that passes for both would not be testing anything.

    python tests/test_ref2va_audio_context.py
"""

import sys

FAILURES = []


def check(label, condition, detail=""):
    print(f"  [{'pass' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


# --------------------------------------------------------------------------
# packing.py's time axis
# --------------------------------------------------------------------------

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
RESERVE_DEFICIT = 1                      # _RESERVE_DEFICIT

TEXT_LEN = 512
CARRIED = 7                              # H3LC_LATENTS default
NATIVE_AUDIO = 30                        # audio latents in the native 0.75s window
TARGET_FRAMES = 32


def video_t_grid(length, origin):        # packing.py:85
    times, accumulated = [origin], 0.0
    for index in range(length - 1):
        accumulated += FRAME_RESCALE * FRAME_PER_TOKEN[index % 5]
        times.append(origin + accumulated)
    return times


def reference_t_span(length):            # packing.py:_reference_t_span
    return sum(FRAME_RESCALE * FRAME_PER_TOKEN[index % 5] for index in range(length))


def layout(extra, hold_references, references=(1.0, 1.0, 12.0)):
    """Times of every block, as build_ref2va_packed_sequence would set them.

    `references` is each reference's advance of time_cursor: 1.0 for an image
    (packing.py:245), num_audio_latents for an audio clip (251), and
    max(num_audio_latents, _reference_t_span(frames)) for a video (263).
    """
    cursor = float(TEXT_LEN)
    reference_spans = []
    for advance in references:
        reference_spans.append((cursor, cursor + advance))
        cursor += advance

    time_cursor = cursor
    history = video_t_grid(CARRIED, time_cursor)
    origin = time_cursor + reference_t_span(CARRIED)
    target = video_t_grid(TARGET_FRAMES, origin)
    # Audio conditions are laid on a uniform integer axis from time_cursor
    # (_fill_audio_condition_positions), one unit per latent - no periodicity.
    audio_history = [time_cursor + index for index in range(NATIVE_AUDIO + extra)]

    # The plugin's block translation: the carried block alone moves later.
    delta = RESERVE_DEFICIT * FRAME_RESCALE
    history = [time + delta for time in history]

    if extra:
        # Everything except the audio history moves later by extra...
        history = [time + extra for time in history]
        origin += extra
        target = [time + extra for time in target]
        # ...and the references either come along or do not.
        if not hold_references:
            reference_spans = [(start + extra, stop + extra)
                               for start, stop in reference_spans]

    return {"references": reference_spans, "history": history, "origin": origin,
            "target": target, "audio_history": audio_history,
            "time_cursor": time_cursor}


def audio_overlapping_references(shape):
    """Audio-history latents whose time falls inside a reference's span."""
    hits = []
    for time in shape["audio_history"]:
        for start, stop in shape["references"]:
            if start <= time < stop:
                hits.append(time)
                break
    return hits


# --------------------------------------------------------------------------

print("\nno extension (extra = 0)")

native = layout(extra=0, hold_references=False)
check("the audio history does not reach the references",
      not audio_overlapping_references(native))
check("the audio history stops before target_origin",
      native["audio_history"][-1] < native["origin"],
      f"last latent {native['audio_history'][-1]:.2f} < origin "
      f"{native['origin']:.2f}")
# What the block translation is worth is test_layout_contract.py's business.
# All that matters here is that the extension does not disturb it, so the
# distance from the block to the target is recorded and compared, not asserted
# to any particular value.
NATIVE_REACH = native["origin"] - native["history"][-1]
print(f"    (block-to-origin distance without the extension: "
      f"{NATIVE_REACH:.3f} time units)")

EXTRA = 24                               # ~0.6s more context at 40 audio latents/s

print(f"\nextension of {EXTRA} latents, references moved with everything (1.2.1)")

naive = layout(extra=EXTRA, hold_references=False)
collisions = audio_overlapping_references(naive)
check("the extended tail DOES collide with the references",
      len(collisions) > 0,
      f"{len(collisions)} latents inside a reference span - the bug, and the "
      f"reason Ref2VA was refused outright")

print(f"\nextension of {EXTRA} latents, reference span held still")

fixed = layout(extra=EXTRA, hold_references=True)

check("the extended tail no longer touches the references",
      not audio_overlapping_references(fixed))
check("the tail occupies the gap the video history vacated",
      fixed["audio_history"][0] == fixed["time_cursor"]
      and fixed["history"][0] > fixed["time_cursor"],
      f"tail starts at {fixed['audio_history'][0]:.2f}, block at "
      f"{fixed['history'][0]:.2f}")
check("the tail still stops before target_origin",
      fixed["audio_history"][-1] < fixed["origin"],
      f"last latent {fixed['audio_history'][-1]:.2f} < origin "
      f"{fixed['origin']:.2f}")
check("nothing is pushed below text_len",
      min([time for span in fixed["references"] for time in span]
          + fixed["history"] + fixed["audio_history"]) >= TEXT_LEN)
check("the references keep their own order and spacing",
      fixed["references"] == native["references"],
      "held still means unchanged, not merely non-overlapping")
check("the block translation survives the extension",
      abs((fixed["origin"] - fixed["history"][-1]) - NATIVE_REACH) < 1e-9,
      "the block-to-target distance is unchanged, so the extension and the "
      "translation are independent")
check("the audio history keeps its start while the video history moves off it",
      fixed["audio_history"][0] == fixed["time_cursor"]
      and abs(fixed["history"][0]
              - (fixed["time_cursor"] + EXTRA + FRAME_RESCALE * RESERVE_DEFICIT)) < 1e-9,
      f"the tail reaches {EXTRA} latents further back, as on FL2VA")

print("\nvarying the reference load")

for references in ((), (1.0,), (1.0,) * 9, (12.0, 40.0), (1.0,) * 9 + (40.0, 40.0)):
    shape = layout(extra=EXTRA, hold_references=True, references=references)
    ok = (not audio_overlapping_references(shape)
          and shape["audio_history"][-1] < shape["origin"]
          and abs((shape["origin"] - shape["history"][-1]) - NATIVE_REACH) < 1e-9)
    check(f"{len(references)} reference(s) advancing {sum(references):.0f} units",
          ok)

print("\nvarying the extension")

for extra in (1, 8, 24, 48, 96):
    shape = layout(extra=extra, hold_references=True)
    ok = (not audio_overlapping_references(shape)
          and abs((shape["origin"] - shape["history"][-1]) - NATIVE_REACH) < 1e-9)
    check(f"extra = {extra:3d} latents",
          ok,
          "" if shape["audio_history"][-1] < shape["origin"]
          else f"NOTE tail passes origin ({shape['audio_history'][-1]:.1f} vs "
               f"{shape['origin']:.1f}) - H3LC_AUDIO_CONTEXT is clamped well "
               f"below this")

print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
