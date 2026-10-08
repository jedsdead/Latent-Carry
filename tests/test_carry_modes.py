#!/usr/bin/env python3
"""Audio-only carry, and the end-image tail-trim guard.

Unlike test_layout_contract.py and test_audio_context.py, which model the
geometry, this one loads patches.py and exercises _plan_window and _take_cached
for real.  Torch and the Wan2GP modules they import are stubbed - the two paths
under test never reach a tensor operation, so a stub is enough and the suite
keeps its "numpy only, no torch, no GPU" property.

The frame_scheduler helpers are reproduced verbatim from
shared/utils/frame_scheduler.py rather than approximated, because the trim guard
subtracts one number wgp.py computed from another the pipeline computed and an
almost-right normaliser would hide a sign error.

WHAT IS BEING CHECKED
---------------------
1. H3LC_VIDEO=0 refuses the video carry and does so as a configuration rather
   than a failure, so the fall-back counter keeps meaning "wanted to carry and
   could not".  Audio carry is untouched by it: the two substitutions happen in
   different pipeline methods and only the video one needs coordinate work.

2. An end image pinned short of the last generated frame means wgp.py trimmed
   the window, so the cached tail would end past the video the next window hands
   back.  That window must not be cached.  An end image on the last frame - no
   trim - must still cache, because nothing about the geometry changes.

3. Anything conditioned on target frame 0 blocks the carry, because Wan2GP
   already pins the join there and the corrected carried block also reaches it.
   A third assertion makes the model hold on that instant.  This is keyed to the
   collision, not to a plugin: Sliding Window Anchor puts its anchor there
   through frames_to_inject, and Wan2GP's own `L` injection lands a frame at the
   same relative position on the following window.  Blocking the carry is not
   the same as refusing to cache - the window's own output is still sound - so
   the two decisions are checked separately.

    python tests/test_carry_modes.py
"""

import importlib.util
import math
import pathlib
import sys
import types

FAILURES = []


def check(label, condition, detail=""):
    print(f"  [{'pass' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


# --------------------------------------------------------------------------
# Wan2GP stand-ins
# --------------------------------------------------------------------------

# Verbatim from shared/utils/frame_scheduler.py.
def normalize_frame_count(frame_count, minimum, step, offset=1):
    frame_count = max(minimum, frame_count)
    step = max(1, step)
    offset = max(0, offset)
    return math.ceil(max(0, frame_count - offset) / step) * step + offset if step > 1 else frame_count


def floor_frame_count(frame_count, minimum, step, offset=1):
    frame_count = max(minimum, frame_count)
    step = max(1, step)
    offset = max(0, offset)
    if step <= 1:
        return frame_count
    lower = ((frame_count - offset) // step) * step + offset
    return lower if lower >= minimum else normalize_frame_count(minimum, minimum, step, offset)


def normalize_overlap(frame_count, step, offset=1):
    if frame_count < 0:
        return None, "/overlap must be 0 or a positive frame count."
    if frame_count == 0:
        return 0, None
    step = max(1, step)
    offset = max(0, offset)
    overlap = ((frame_count - offset + step // 2) // step) * step + offset
    return max(step if offset == 0 else offset, overlap), None


def video_latent_frames(n):                       # pipeline.video_latent_frames
    return 2 + ((n - 5) // 17) * 5


class _Clip:
    """The slice of _as_video(input_video) that _plan_window actually touches."""

    def __init__(self, frames):
        self.shape = (1, frames)

    def __getitem__(self, key):
        stop = key[1].stop
        start = key[1].start
        length = self.shape[1]
        begin = length + start if start is not None and start < 0 else (start or 0)
        end = length + stop if stop is not None and stop < 0 else (length if stop is None else stop)
        return _Clip(max(0, end - begin))


def _install_stubs():
    torch = types.ModuleType("torch")
    torch.uint8 = "uint8"
    torch.float32 = "float32"
    torch.nn = types.SimpleNamespace(functional=types.SimpleNamespace())
    sys.modules.setdefault("torch", torch)

    for name in ("models", "models.minimax_h3", "shared", "shared.utils"):
        sys.modules.setdefault(name, types.ModuleType(name))

    pipeline = types.ModuleType("models.minimax_h3.pipeline")
    pipeline._as_video = lambda video: video
    pipeline.video_latent_frames = video_latent_frames
    sys.modules["models.minimax_h3.pipeline"] = pipeline

    scheduler = types.ModuleType("shared.utils.frame_scheduler")
    scheduler.normalize_frame_count = normalize_frame_count
    scheduler.floor_frame_count = floor_frame_count
    scheduler.normalize_overlap = normalize_overlap
    sys.modules["shared.utils.frame_scheduler"] = scheduler


def _load_patches():
    _install_stubs()
    path = pathlib.Path(__file__).resolve().parent.parent / "patches.py"
    spec = importlib.util.spec_from_file_location("swl_patches", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["swl_patches"] = module
    spec.loader.exec_module(module)
    return module


patches = _load_patches()


def window(**overrides):
    """A continued window that _plan_window should find cacheable."""
    values = {"window_no": 2,
              "prefix_frames_count": 18,
              "frames_to_inject": (),
              "frames_relative_positions_list": (),
              "image_end": None,
              "input_video": _Clip(124),
              "image_start": None,
              "input_waveform": object(),
              "frame_num": 124,
              "audio_prompt_type": ""}
    values.update(overrides)
    return values


# --------------------------------------------------------------------------
# 1. audio-only carry
# --------------------------------------------------------------------------

print("\naudio-only carry (H3LC_VIDEO=0)")

patches.CONFIG.enable = True
patches.CONFIG.video = True
latents, reason = patches._take_cached(None)
check("with video on, the gate is not what stops the carry",
      reason is not patches._VIDEO_OFF,
      f"reason {reason!r}")

patches.CONFIG.video = False
latents, reason = patches._take_cached(None)
check("with video off, the carry is refused before any tensor work",
      latents is None and reason is patches._VIDEO_OFF)

check("the refusal is identifiable, not just a string that reads right",
      reason is patches._VIDEO_OFF,
      "identity, so _patched_add_video_history cannot confuse it "
      "with a real fall back")

source = (pathlib.Path(__file__).resolve().parent.parent / "patches.py").read_text(encoding="utf-8")
video_off_branch = source.split("if reason is _VIDEO_OFF:", 1)
check("a deliberate audio-only run does not inflate the fall-back counter",
      len(video_off_branch) == 2
      and "fell_back" not in video_off_branch[1].split("else:", 1)[0],
      "fell_back is incremented on the else branch only")

check("audio substitution does not consult CONFIG.video",
      "CONFIG.video" not in source.split("def _patched_encode_audio", 1)[1]
                                  .split("def ", 2)[0],
      "the audio carry is independent of the video carry by construction")

check("extended audio context names H3LC_VIDEO when that is what blocks it",
      "H3LC_VIDEO=0" in source.split("def _audio_extension", 1)[1].split("def ", 2)[0])

patches.CONFIG.video = True


# --------------------------------------------------------------------------
# 2. the end-image tail-trim guard
# --------------------------------------------------------------------------

print("\nend images and the tail trim")

# wgp.py:7724  image_end_frame_position = current_video_length - tail_trim - 1
def end_position(frame_num, trim):
    return frame_num - trim - 1


plan = patches._plan_window(None, (), window())
check("a plain continued window is cacheable",
      plan.get("usable"), f"expected_latents {plan.get('expected_latents')}")

plan = patches._plan_window(None, (), window(
    image_end=object(), image_end_frame_position=end_position(124, 0)))
check("an end image on the last frame still caches",
      plan.get("usable"),
      "no trim, so the cached tail still ends on the join frame")

for trim in (1, 4, 17):
    plan = patches._plan_window(None, (), window(
        image_end=object(), image_end_frame_position=end_position(124, trim)))
    check(f"a {trim}-frame trim refuses the cache",
          not plan.get("usable"), plan.get("reason", ""))

plan = patches._plan_window(None, (), window(
    image_end=object(), image_end_frame_position="not a number"))
check("a position that is not an integer refuses rather than raises",
      not plan.get("usable"), plan.get("reason", ""))

plan = patches._plan_window(None, (), window(image_end_frame_position=None))
check("no end image leaves the window cacheable",
      plan.get("usable"),
      "the guard reads a trim it can see, it does not assume one")

plan = patches._plan_window(None, (), window(image_end=object(),
                                             image_end_frame_position=None))
check("an end image with no explicit position caches",
      plan.get("usable"),
      "pipeline.py:799 puts it on the real last frame, so there is no trim")

# HISTORY_COUNT is the pipeline's, i.e. overlap - 1.
HISTORY_COUNT = 17
plan = patches._plan_window(None, (), window(
    image_end=object(), image_end_frame_position=HISTORY_COUNT - 4))
check("an end image resolving below the target refuses",
      not plan.get("usable"), plan.get("reason", ""))

# The trim is what matters, not the end image.  A window pinned to an end image
# on its final frame is geometrically identical to one with no end image: the
# condition is anchored to target_origin (packing.py:185) and the carried block
# is moved relative to target_origin, so the two never interact.
plain = patches._plan_window(None, (), window())
pinned = patches._plan_window(None, (), window(
    image_end_frame_position=end_position(124, 0)))
check("an untrimmed end image changes nothing about the plan",
      plain.get("expected_latents") == pinned.get("expected_latents")
      and plain.get("continuation_audio") == pinned.get("continuation_audio"),
      f"{plain.get('expected_latents')} latents either way")


# --------------------------------------------------------------------------
# 3. conditions on target frame 0
# --------------------------------------------------------------------------

print("\nconditions on target frame 0")

plan = patches._plan_window(None, (), window())
check("a window with nothing on frame 0 carries",
      plan.get("carry_blocked") is None)

# Sliding Window Anchor appends its anchor at raw index history_count, which
# pipeline.py:802-810 resolves to target frame 0.
anchor = patches._plan_window(None, (), window(
    frames_to_inject=(object(),),
    frames_relative_positions_list=(HISTORY_COUNT,)))
check("an injected frame on frame 0 blocks the carry",
      anchor.get("carry_blocked") is not None,
      anchor.get("carry_blocked", ""))
check("...but the window is still cacheable",
      anchor.get("usable"),
      "its own output is sound; only its conditioning is crowded")

for offset, label in ((1, "one frame into the target"),
                      (40, "mid window")):
    plan = patches._plan_window(None, (), window(
        frames_to_inject=(object(),),
        frames_relative_positions_list=(HISTORY_COUNT + offset,)))
    check(f"an injected frame {label} does not block the carry",
          plan.get("carry_blocked") is None,
          f"resolves to target frame {offset}")

plan = patches._plan_window(None, (), window(
    frames_to_inject=(None,),
    frames_relative_positions_list=(HISTORY_COUNT,)))
check("an unfilled injection slot is not a frame",
      plan.get("carry_blocked") is None)

plan = patches._plan_window(None, (), window(
    image_end=object(), image_end_frame_position=HISTORY_COUNT))
check("an end image pinned to the join frame blocks the carry",
      plan.get("carry_blocked") is not None,
      plan.get("carry_blocked", ""))
check("...and is read as a collision, not as a tail trim",
      plan.get("usable"),
      "a large bogus trim number would have refused the cache instead")

plan = patches._plan_window(None, (), window(
    frames_to_inject=(object(), object()),
    frames_relative_positions_list=(HISTORY_COUNT, HISTORY_COUNT + 60)))
check("one colliding frame among several is enough",
      plan.get("carry_blocked") is not None)

# Wan2GP's own `L` injection must NOT trip the guard, and the reason is
# arithmetic rather than a plugin being absent.  Modelled from wgp.py:7952
# (the slice), 8007 (the relative position) and 7725 (window_start_frame),
# with extract_guide_from_window_start False as it is on FL2VA and Ref2VA.
REUSE_FRAMES = 18                      # overlap; history_count is this minus 1


def native_relative_position(abs_pos, guide_start):
    """What wgp.py hands the pipeline for a frame at abs_pos in this window."""
    window_start = guide_start - REUSE_FRAMES        # wgp.py:7725
    if not guide_start <= abs_pos:                   # wgp.py:7952, slice start
        return None                                  # not in this window at all
    return abs_pos - window_start                    # wgp.py:8007, simplified


GUIDE_START = 500                                    # any window past the first
lowest = native_relative_position(GUIDE_START, GUIDE_START)
check("the lowest relative position a native injection can reach is the overlap",
      lowest == REUSE_FRAMES, f"{lowest}")
check("so the lowest native frame_index is 1, never 0",
      lowest - HISTORY_COUNT == 1)

# An L frame marks a window's last emitted frame, and the next window's slice
# begins at the frame after it (wgp.py:7758 cur_end_pos, 7709 guide_start).
check("an L frame from the previous window is not in this window's slice",
      native_relative_position(GUIDE_START - 1, GUIDE_START) is None,
      "it sits one index below the slice start, so it is not handed forward")

plan = patches._plan_window(None, (), window(
    frames_to_inject=(object(),),
    frames_relative_positions_list=(lowest,)))
check("a native injection at that position carries normally",
      plan.get("carry_blocked") is None and plan.get("usable"))

# --------------------------------------------------------------------------
# 4. Ref2VA and extended audio context
# --------------------------------------------------------------------------

print("\nextended audio context on Ref2VA")


class FakePipeline:
    def __init__(self, reference_mode):
        self.reference_mode = reference_mode


patches.CONFIG.audio_context = 2.0
patches.CONFIG.video = True
patches.CONFIG.fix_coords = True
patches.STATE.active_latents = object()

check("FL2VA still extends",
      patches._audio_extension(30, 192, FakePipeline(False)) > 0)
# Forced, because the stubbed packing module has no MiniMaxH3PackedSequence for
# _reference_rows_locatable to inspect - and it declining on a stub is itself
# the correct answer, checked two assertions below.
patches._REF_ROWS_LOCATABLE = True
check("Ref2VA extends when the reference rows can be located",
      patches._audio_extension(30, 192, FakePipeline(True)) > 0,
      "_apply_origin_shift holds that span still; see "
      "tests/test_ref2va_audio_context.py")

# The capability check is what decides, not the mode.
patches._REF_ROWS_LOCATABLE = False
check("Ref2VA declines when the builder does not report the row counts",
      patches._audio_extension(30, 192, FakePipeline(True)) == 0,
      "nothing safe to do without the reference span")
check("FL2VA is unaffected by that, since it has no reference rows",
      patches._audio_extension(30, 192, FakePipeline(False)) > 0)
patches._REF_ROWS_LOCATABLE = None
check("a pipeline that does not report the mode is treated as FL2VA",
      patches._audio_extension(30, 192, None) > 0,
      "reference_mode is absent on FL2VA, so the default must not refuse")

# The block translation must NOT be gated the same way: it is invariant to the
# reference cursor.  packing.py places the Ref2VA history at
# history_time = time_cursor and the target at time_cursor + _reference_t_span,
# so both move together and the gap the correction closes never changes.
FPT, RESCALE = (1, 4, 4, 4, 4), 5.0 / 3.0


def video_t_grid(length, origin):
    times, accumulated = [origin], 0.0
    for index in range(length - 1):
        accumulated += RESCALE * FPT[index % 5]
        times.append(origin + accumulated)
    return times


def reference_t_span(length):
    return sum(RESCALE * FPT[index % 5] for index in range(length))


gaps = set()
for carried in (7, 12, 17):
    for cursor in (0.0, 512.0, 512.0 + 37.0 / 3):     # text_len, then refs moved it
        block_end = video_t_grid(carried, cursor)[-1]
        origin = cursor + reference_t_span(carried)
        gaps.add(round((origin - block_end) / RESCALE, 6))
check("the gap the correction closes is invariant to the reference cursor",
      len(gaps) == 1, f"{gaps.pop()} frames for every carried count and cursor")

patches.CONFIG.audio_context = 0.0
patches.STATE.active_latents = None


# _take_cached must actually honour it, not merely record it.
patches.STATE.plan = anchor
latents, reason = patches._take_cached(None)
check("_take_cached refuses while the collision stands",
      latents is None and reason == anchor["carry_blocked"])
patches.STATE.plan = None


print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
