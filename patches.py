"""Carry MiniMax H3 latents across sliding windows, and across jobs, instead of
re-encoding pixels.

H3 Latent Carry is Sliding Window Latents 1.2.3 plus a persistence layer: the
last window's latents are saved inside the plugin folder with every video, and
a later Continue Video job picks them up automatically (see "Across jobs").

Wan2GP already pins a run of previous-window frames as history conditioning: at overlap
18 it takes 17 pixel frames, encodes them in a single VAE call with
keep_all_latents=True, and submits them as an "history" keyframe block
(models/minimax_h3/pipeline.py, _add_video_history).  That works, but the pixels
it encodes are themselves VAE output from the previous window, so every window
adds one decode -> encode round trip to the carried context.

This module replaces that re-encode with the previous window's *own* latents,
captured on their way into the decoder.

Seven bindings are patched, all reversible via uninstall():

  1. MiniMaxH3VideoVAE.decode      - capture the tail of the latent tensor plus a
                                     join-frame fingerprint.
  2. MiniMaxH3AudioVAE.decode      - same for audio, plus an envelope signature.
  3. MiniMaxH3Pipeline.generate    - window bookkeeping + usability checks.
  4. MiniMaxH3Pipeline._add_video_history
                                   - substitute cached latents for the encode.
  5. MiniMaxH3Pipeline._encode_audio
                                   - substitute cached audio latents for the
                                     continuation encode (first call only).
  6. transformer.build_packed_sequence
  7. transformer.build_ref2va_packed_sequence
                                   - correct the time coordinates (see below).

Phase alignment, and why the coordinate correction is small
-----------------------------------------------------------
H3 compresses time periodically, not as a single clip-initial special case.
pipeline.video_latent_frames is 2 + ((N-5)//17)*5 - seventeen pixel frames per
five latents - and _FRAME_PER_TOKEN = (1, 4, 4, 4, 4) in components/packing.py is
that same period.  Latent i covers _FRAME_PER_TOKEN[i % 5] pixel frames, so every
fifth latent covers one frame and the rest cover four, all the way through a
window.

packing._video_t_grid walks the pattern from position 0.  That means a carried
block is laid out correctly *if and only if* its first latent sits at an absolute
index divisible by 5.  Since a window always holds 2 + 5k latents, slicing the
last n gives a phase-aligned block exactly when n = 2 (mod 5).  n = 7 is the
smallest useful choice and carries 22 pixel frames, a little more context than
the 17 the native re-encode provides.

With the phase right, the internal spacing needs no correction at all.  What
remains is a pure translation: the native block stops one frame short of the join
(history_frames excludes it), while a carried block's last latent contains the
join frame, so reference_t_span places the whole block one pixel frame too early.
Patch (4) shifts everything anchored to target_origin up by that single frame.

    n=7, carried    truth   -21  -20  -16  -12   -8   -4   -3
                    layout  -22  -21  -17  -13   -9   -5   -4
                            spacing identical, translation -1

Set H3LC_FIX_COORDS=0 to skip patches (6) and (7) and reproduce the uncorrected
layout for A/B purposes.

Job boundaries
--------------
One generate() call is one window, but consecutive generate() calls are not
necessarily consecutive windows of the same video.  Wan2GP's queue runs jobs
back to back in the same process, and a job that repeats a prompt restarts its
window loop for every sample, so "the previous generate call" is routinely the
last window of an unrelated video.

wgp.py passes window_no on every call (it lands in generate's **kwargs), and it
restarts at 1 for each new job and each repeat within a job.  That is the exact
boundary signal: patch (3) drops the cache whenever it sees window_no <= 1, and
the carry paths additionally require the cached window to be numbered exactly one
lower than the window asking for it.  A first window is therefore never served
from the in-memory cache, whatever its pixels look like - including a
continue-video job whose source happens to be the previous job's output.

Across jobs
-----------
The one thing that may seed window 1 is a saved entry for the exact source
file.  Every video this plugin records is hashed after Wan2GP has finished
writing it (metadata included), and the last window's cache is stored under
that SHA-256 in the plugin's latents/ folder.  When a Continue Video job starts,
window 1 hashes the Video to Continue file; an entry of that name is loaded into
the cache as though it were window 0 of the same job.  Everything after that is
the ordinary carry path, so the same checks decide whether it is used: latent
count, phase, resolution, and the join-frame fingerprint search, which confirms
that the frames Wan2GP decoded from the source file are the ones the latents
were decoded into.  A source truncated by the continue settings, a different
fps or resolution, or a different base model is refused before any of that.

An entry is only written when the video file still shows what the latents
describe: at save time the exported frames' tail is fingerprinted against the
decoded window, and fps and resolution are compared.  Temporal or spatial
upsampling, discarded last frames and similar post-processing therefore skip the
save with a log line rather than store latents for a different picture.

If window_no is missing the caller is not wgp.py's window loop and the boundary
cannot be established, so the plugin stays inert rather than guess.  Set
H3LC_REQUIRE_WINDOW_NO=0 to fall back to fingerprint-only matching for a
front end that drives the pipeline itself.

Configuration
-------------
The environment variables below set the initial values only.  CONFIG is read at
each decision point rather than captured at install time, so the Gradio panel
changes behaviour live and anything saved from it wins over the environment.

  H3LC_ENABLE        1/0    master switch                          (default 1)
  H3LC_FIX_COORDS    1/0    apply the coordinate correction        (default 1)
  H3LC_REQUIRE_WINDOW_NO
                     1/0    refuse to carry when the caller does
                            not report window_no                   (default 1)
  H3LC_LATENTS       int    carried latent count; 0 uses 7.  Must
                            be 2 (mod 5): 7, 12, 17...             (default 0)
  H3LC_VIDEO         1/0    carry video latents.  Set 0 with
                            H3LC_AUDIO=1 for audio-only carry,
                            which needs no coordinate work at all  (default 1)
  H3LC_AUDIO         1/0    also carry audio latents               (default 1)
  H3LC_VERBOSE       1/0    log every engage / fall back           (default 1)
  H3LC_MATCH_TOL     float  join-frame fingerprint tolerance       (default 0.03)
  H3LC_AUDIO_TOL     float  audio envelope tolerance               (default 0.15)
  H3LC_COLOUR        1/0    hold each continuous shot's colour     (default 0)
  H3LC_COLOUR_AXES   list   brightness,contrast,saturation,cast    (default all)
  H3LC_COLOUR_STRENGTH float  how much of the drift to remove      (default 1.0)
  H3LC_CUT_THRESHOLD float  frame change read as a cut             (default 0.08)
  H3LC_SAVE         1/0    save the last window's latents with
                            every recorded video                   (default 1)
  H3LC_SAVE_SCOPE   final  one entry per job, written at the end;
                    every  or one per recorded window           (default final)
  H3LC_TRACE        1/0    trace each Wan2GP hook call            (default 0)
  H3LC_RESTORE      1/0    use saved latents when continuing a
                            video                                  (default 1)
  H3LC_STORE_GB     float  size cap for the latents/ folder; least
                            recently used entries go first         (default 2)
  H3LC_STORE_DIR    path   store location                (default <plugin>/latents)
"""

import inspect
import os
import sys
from collections import deque
import threading
import time
from contextvars import ContextVar

import numpy as np
import torch


_FRAME_RESCALE = 5.0 / 3.0          # packing._FRAME_RESCALE
_LATENT_PERIOD = 5                  # latents per 17 pixel frames
_PHASE_RESIDUE = 2                  # a window holds 2 + 5k latents, so n must be 2 (mod 5)
_RESERVE_DEFICIT = 1                # frames of translation error at the join
_DEFAULT_LATENTS = 7                # smallest phase-aligned block worth carrying (22 frames)
_AUDIO_CHANNELS = 2                 # packing.MINIMAX_H3_AUDIO_CHANNELS
_VAE_SPATIAL_FACTOR = 16            # confirmed by pipeline's ceil(target_height / 16)
_FINGERPRINT_GRID = (16, 16)        # resolution-independent frame signature
_ALIGN_SEARCH_FRAMES = 12           # decoded tail frames fingerprinted for the search
_ALIGN_EXPECTED_OFFSET = -2         # where the join frame must sit; see _check_alignment
_ALIGN_DECISIVE = 0.6               # a rival offset must beat expected by this factor
_AUDIO_SAMPLE_RATE = 32000          # pipeline.AUDIO_SAMPLE_RATE
_AUDIO_SIGNATURE_BINS = 32
_AUDIO_SIGNATURE_SECONDS = 0.5
_MAX_CACHED_AUDIO_LATENTS = 192     # 18 frames at 40 latent fps is ~30; headroom for
                                    # extended audio context up to ~4.8s
# Wan2GP's audio autoencoder has a hop of 800 samples at 32 kHz, i.e. 40
# latents/s (models/minimax_h3/components/audio_autoencoder.py).  Resolved from
# the pipeline where available rather than assumed.
_AUDIO_LATENT_FPS_FALLBACK = 40.0
# Wan2GP v18.19+ generates H3 audio a few latents past the last video frame
# (pipeline.H3_AUDIO_TAIL_LATENTS = 5, ~125 ms).  That tail is after the join, so
# it is cut off before caching.  Anything larger than this is not a tail.
_MAX_AUDIO_TAIL_LATENTS = 16
_CLIP_LENGTH = 17                   # H3 window quantum
_COLOUR_AXES = ("brightness", "contrast", "saturation", "cast")
# How far the previous window's own latent for the join frame may sit from
# Wan2GP's encode of the finished frame (mean absolute difference, in the
# normalised latent space) before it is distrusted as a different picture.
_ANCHOR_TOLERANCE = 0.5
# Discard Last Frames that moves the join onto a latent boundary: a window's
# last 5 frames come from its last two latents (1 + 4 frames), the decoder's
# unblended look-ahead.  Dropping the 4 puts the join on the 1-frame latent,
# where a new decoder chunk starts.
_TAKEOVER_FRAMES = 4
_FRAMES_PER_CHUNK = 17              # frames decoded from one 5-latent chunk
_LOCK_REFERENCE_FRAMES = 17         # a shot's opening frames that define the look held
_CUT_MIN = 0.08                     # frame-to-frame structural change that is always a cut
_CUT_RATIO = 6.0                    # ... or this many times the window's typical change
_LOCK_SMOOTH = 8                    # per-frame corrections averaged over +-8 frames
_LOCK_SCENE_SHARE = 0.25            # more frames than this off-reference = a new scene
_COLOUR_POOL = 64                   # measurement is pooled to this width
_COLOUR_CONTINUATION_SPAN = 48      # frames of a continued video that define its look
_VIDEO_OFF = "video carry is switched off (audio-only)"
_MAX_CACHED_LATENTS = 17            # floor for the cached tail; covers n = 7, 12 and 17.
                                    # _cache_depth() raises this to match a larger
                                    # H3LC_LATENTS so the cap can never silently
                                    # undercut the configured slice.


def _flag(name, default):
    return str(os.environ.get(name, default)).strip().lower() not in ("0", "false", "no", "")


class _Config:
    def __init__(self):
        self.enable = _flag("H3LC_ENABLE", "1")
        self.fix_coords = _flag("H3LC_FIX_COORDS", "1")
        self.require_window_no = _flag("H3LC_REQUIRE_WINDOW_NO", "1")
        self.verbose = _flag("H3LC_VERBOSE", "1")
        try:
            self.latents = max(0, int(os.environ.get("H3LC_LATENTS", "0")))
        except ValueError:
            self.latents = 0
        try:
            self.match_tolerance = float(os.environ.get("H3LC_MATCH_TOL", "0.03"))
        except ValueError:
            self.match_tolerance = 0.03
        # The two carries are independent.  Video substitution happens in
        # _add_video_history and needs the coordinate correction; audio
        # substitution happens in _encode_audio and needs none, because
        # _fill_audio_condition_positions lays audio out on a uniform integer
        # axis.  So H3LC_VIDEO=0 with H3LC_AUDIO=1 is a real mode, not a
        # degenerate one - it is the plugin's least invasive configuration.
        self.video = _flag("H3LC_VIDEO", "1")
        self.audio = _flag("H3LC_AUDIO", "1")
        # Across-job persistence.  Both on by default: the point of the plugin
        # is that continuing a video needs no extra steps.
        self.save = _flag("H3LC_SAVE", "1")
        # Traces each Wan2GP hook call to the console.  On by default while the
        # across-jobs feature is new; noisy but it makes a broken chain obvious.
        self.trace = _flag("H3LC_TRACE", "0")
        self.restore = _flag("H3LC_RESTORE", "1")
        # "final": one entry per job, written when the job ends.  "every": one per
        # recorded window, which lets a mid-job file be continued but hashes and
        # writes a growing file every window.
        self.save_scope = os.environ.get("H3LC_SAVE_SCOPE", "final").strip().lower()
        if self.save_scope not in ("final", "every"):
            self.save_scope = "final"
        try:
            self.store_limit_gb = float(os.environ.get("H3LC_STORE_GB", "2"))
        except ValueError:
            self.store_limit_gb = 2.0
        if not 0.0 <= self.store_limit_gb <= 1000.0:
            self.store_limit_gb = 2.0
        # Extended audio context, in seconds.  0 keeps Wan2GP's behaviour, where
        # the audio window is derived from the video overlap and is therefore
        # 0.75s at overlap 18 - ample for motion, short for sound.
        try:
            self.audio_context = float(os.environ.get("H3LC_AUDIO_CONTEXT", "0"))
        except ValueError:
            self.audio_context = 0.0
        if not 0.0 <= self.audio_context <= 4.0:
            self.audio_context = 0.0
        try:
            self.audio_tolerance = float(os.environ.get("H3LC_AUDIO_TOL", "0.15"))
        except ValueError:
            self.audio_tolerance = 0.15
        # Search a run of frames for the join rather than testing one offset.
        self.align_search = _flag("H3LC_ALIGN_SEARCH", "1")
        # In the takeover layout the join is a 1-frame latent of the previous
        # window: pin that latent at frame 0 instead of re-encoding the finished
        # frame.  Off by default while it is being evaluated.
        self.latent_anchor = _flag("H3LC_LATENT_ANCHOR", "0")
        # With Wan2GP's Discard Last Frames at 4, decode each new window together
        # with the carried end of the previous one, so the join is blended the
        # way the middle of a clip is.
        self.takeover = _flag("H3LC_TAKEOVER", "1")
        # Hold each continuous shot to the look of its own opening.  Off by
        # default: a plugin update should not quietly start altering frames.
        self.colour = _flag("H3LC_COLOUR", "0")
        axes = os.environ.get("H3LC_COLOUR_AXES", ",".join(_COLOUR_AXES))
        self.colour_axes = tuple(a.strip() for a in axes.split(",")
                                 if a.strip() in _COLOUR_AXES)
        try:
            self.cut_threshold = float(os.environ.get("H3LC_CUT_THRESHOLD", str(_CUT_MIN)))
        except ValueError:
            self.cut_threshold = _CUT_MIN
        for name, key, default in (("colour_strength", "H3LC_COLOUR_STRENGTH", 1.0),
                                   ("colour_scene", "H3LC_COLOUR_SCENE", 0.06),
                                   ("colour_max", "H3LC_COLOUR_MAX", 0.15)):
            try:
                setattr(self, name, float(os.environ.get(key, str(default))))
            except ValueError:
                setattr(self, name, default)


CONFIG = _Config()


def _carried_count():
    return int(CONFIG.latents or _DEFAULT_LATENTS)


def _cache_depth(total_latents=None):
    """How many latents to keep from each window's tail.

    Read at capture time rather than fixed at import, because the panel changes
    CONFIG.latents live.
    """
    return max(_MAX_CACHED_LATENTS, _carried_count())


def _log(message):
    LOG_HISTORY.append(f"{time.strftime('%H:%M:%S')} {message}")
    state = globals().get("STATE")
    if state is not None:
        state.last_message = message
    if CONFIG.verbose:
        print(f"[h3-latent-carry] {message}")


def _to_unit(tensor):
    """Map a frame to float32 in [0, 1] without assuming its input range.

    Decoded output is [-1, 1]; continuation video reaching _add_video_history may
    be uint8 or float.  Guessing wrong here would make every fingerprint
    comparison fail, so the range is sniffed rather than assumed.
    """
    if tensor.dtype == torch.uint8:
        return tensor.float().div_(255.0)
    values = tensor.float()
    low, high = float(values.min()), float(values.max())
    if low < -0.05:
        return values.add(1.0).div_(2.0)
    if high > 1.5:
        return values.div(255.0)
    return values


def _fingerprint(frame_chw):
    """A small resolution-independent signature of one frame."""
    gray = _to_unit(frame_chw.detach()).mean(dim=0, keepdim=True)
    pooled = torch.nn.functional.adaptive_avg_pool2d(gray.unsqueeze(0), _FINGERPRINT_GRID)
    return pooled[0, 0].to(torch.float32).cpu()


def _audio_signature(waveform):
    """A peak-normalised RMS envelope of the last half second, mono."""
    audio = waveform.detach().float()
    while audio.ndim > 2:
        audio = audio[0]
    if audio.ndim == 2:
        audio = audio.mean(dim=0)
    window = min(int(audio.shape[-1]), int(_AUDIO_SIGNATURE_SECONDS * _AUDIO_SAMPLE_RATE))
    audio = audio[-window:]
    if audio.shape[-1] < _AUDIO_SIGNATURE_BINS:
        return None
    trimmed = (audio.shape[-1] // _AUDIO_SIGNATURE_BINS) * _AUDIO_SIGNATURE_BINS
    envelope = audio[:trimmed].reshape(_AUDIO_SIGNATURE_BINS, -1).pow(2).mean(dim=-1).sqrt()
    peak = float(envelope.max())
    if peak < 1e-5:
        return torch.zeros_like(envelope).cpu()          # silence matches silence
    return (envelope / peak).cpu()


def _audio_matches(cached, incoming):
    if cached is None or incoming is None:
        return False, "no audio signature"
    if float(cached.abs().max()) < 1e-6 and float(incoming.abs().max()) < 1e-6:
        return True, None
    difference = float((cached - incoming).abs().mean())
    if difference > CONFIG.audio_tolerance:
        return False, f"audio mismatch (envelope difference {difference:.4f})"
    return True, None


class _State:
    """Cross-window state.  One generate() call == one window."""

    def __init__(self):
        self.lock = threading.Lock()
        self.generation = 0
        # wgp.py's window number for the call in progress; None when the caller
        # does not report one.  Restarts at 1 for every job and every repeat, so
        # it is what separates "next window" from "next queue entry".
        self.window_no = None
        # cached tail of the previous window
        self.latents = None
        self.signature = None
        self.signatures = None
        self.audio_extra = 0
        self.source_generation = -1
        self.source_window_no = None
        self.usable = False
        self.expected_latents = None
        self.total_latents = None
        # Set the first time the stock _add_video_history is seen returning a
        # value.  See _patched_add_video_history.
        self.history_returns_value = False
        # set per stage by the _add_video_history patch, read by build_packed_sequence
        self.active_latents = None
        self.plan = None
        # audio side
        self.audio_latents = None
        self.audio_signature = None
        self.audio_extra = 0             # extra carried audio latents this window
        self.audio_generation = -1
        self.audio_source_window_no = None
        self.audio_encode_calls = 0
        # colour: statistics survive a window, its frames do not
        self.colour_step = None          # one (A, b) per carried latent, for this window's capture
        self.decoding_latents = 0        # latent count of the decode being measured
        self.lock_reference = None       # the current shot's look: Y'CbCr mean and spread
        self.colour_seed = None          # the look of the video a continuation hands in
        self.last_message = "idle"
        self.stats = {"engaged": 0, "fell_back": 0, "audio_engaged": 0,
                      "colour_applied": 0, "video_off": 0,
                      "restored": 0, "saved": 0, "save_skipped": 0,
                      "hook_generate_media": 0, "hook_save_video": 0,
                      "hook_record_file": 0, "windows": 0, "passthrough": 0,
                      "anchor_from_latents": 0, "anchor_native": 0,
                      "takeover": 0, "joint_decode": 0}
        # Frames the previous window's video lost to Discard Last Frames, which
        # decides where in the cached tail the join sits.  Set at capture from
        # the job, or from a saved entry's manifest.
        self.source_discard = 0
        # This window carries the history that ends one frame before a join on
        # a latent boundary (the takeover layout), and the carried latents to
        # decode ahead of it.
        self.takeover = False
        self.takeover_prefix = None
        # Carried latents waiting to become this stage's join frame; set when
        # the history block is carried, consumed by the next frame-0 condition.
        self.anchor_latents = None
        # The previous window's 1-frame latent for the join (takeover layout).
        self.join_latent = None
        # Name of the source video whose saved latents seeded this job, if any.
        self.restored = None
        # Measured cost of one native history encode, for the panel's estimate.
        self.encode_seconds = 0.0
        self.encode_samples = 0
        # Generation the save hook last saw, to notice hooks that never run.
        self.save_seen = 0
        self.save_warned = 0

    def invalidate(self, reason):
        if self.latents is not None:
            _log(f"cache dropped: {reason}")
        self.latents = None
        self.signature = None
        self.source_window_no = None
        self.audio_latents = None
        self.audio_signature = None
        self.audio_source_window_no = None
        self.usable = False
        self.total_latents = None
        self.lock_reference = None
        self.colour_seed = None
        self.colour_step = None
        self.restored = None


STATE = _State()

VERSION = "2.3.1"

# Per-job context set by lifecycle.py around Wan2GP's generate_media: the model,
# the Video to Continue path and the truncation setting.  None outside a job.
JOB = ContextVar("h3_latent_carry_job", default=None)

# A saved entry the user picked in the panel, used by the next continuation
# whatever file it is given, then cleared.  For footage the lookup cannot
# recognise: a video trimmed or re-encoded outside Wan2GP.
PIN = None

# Set while another refiner's pass runs inside generate() - H3 Face Refiner
# (Plus) calls MiniMaxH3Pipeline.refine_video, which goes through the patched
# generate.  Every hook passes straight through while it is set, so a
# refinement pass is never mistaken for a window: before 1.3.0 one dropped the
# cache and overwrote it with the refiner's latents.
_PASSTHROUGH = ContextVar("h3_latent_carry_passthrough", default=None)

# store.LatentStore, set by plugin.py.  None disables saving and restoring.
STORE = None

# Why the H3 patches are or are not in place, for the panel and diagnostics.
INSTALL_STATE = {"installed": False, "reason": "not attempted yet"}

# The last log lines, so the panel can show them without a console.
LOG_HISTORY = deque(maxlen=40)

# Called at the start of every generate() by lifecycle.py, which uses it to
# re-wrap wgp.py functions another plugin replaced.  None when not installed.
ON_GENERATE = None

# models.minimax_h3.constants.H3_CONTROL_LATENT_CONTINUATION, read in _preflight.
# The pipeline offsets its overlap arithmetic by 5 instead of 1 when this is on
# for a control model; it ships off, and older builds have no such constant.
_CONTROL_LATENT_CONTINUATION = False

# Module name of another copy of this carry logic (e.g. Sliding Window Latents)
# found installed, in which case this plugin stands aside.  See _rival_module.
_RIVAL = None

_PACKING = None                     # packing module, resolved by _preflight()

_ORIGINALS = {}
_ORIGINAL_OWNERS = {}


# --------------------------------------------------------------------------
# 1. latent capture
# --------------------------------------------------------------------------

def _colour_stats(decoded, start, stop):
    """Y'CbCr mean and spread of a slice of decoded frames [B, C, F, H, W].

    Pooled to a small grid first: the statistics wanted are the frame's
    overall grade, so full resolution costs memory without changing the
    answer, and pooling makes the measurement independent of output size.
    """
    from . import colour
    frames = decoded[0, :, start:stop]
    if frames.shape[1] == 0:
        return None
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        frames.permute(1, 0, 2, 3).float(), (_COLOUR_POOL, _COLOUR_POOL))
    unit = _to_unit(pooled).permute(0, 2, 3, 1).cpu().numpy()
    return colour.ycbcr_stats(unit)


def _colour_seed(history_video):
    """Remember the look of the video a Continue Video job hands in.

    Used as the shot's reference when the saved entry carries none (entries
    from before 1.7.0, or a video whose latents could not be matched).
    """
    from . import colour
    if not CONFIG.colour or STATE.lock_reference is not None or STATE.colour_seed is not None:
        return
    frames = history_video
    if frames is None or frames.shape[1] == 0:
        return
    span = min(_COLOUR_CONTINUATION_SPAN, frames.shape[1])
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        frames[:, -span:].permute(1, 0, 2, 3).float(), (_COLOUR_POOL, _COLOUR_POOL))
    unit = _to_unit(pooled).permute(0, 2, 3, 1).cpu().numpy()
    stats = colour.ycbcr_stats(unit)
    STATE.colour_seed = {"mean": np.asarray(stats["mean"]).tolist(),
                         "std": np.asarray(stats["std"]).tolist()}


def _correct_pixels_ramp(decoded, per_frame):
    """Apply a (gain, offset) per frame to the decoded window [B, C, F, H, W].

    Handles both forms decode() returns: signed [-1, 1] floats, and uint8
    frames, which current Wan2GP asks H3's decode for on every window.  The
    correction is defined on unit RGB.  Written back frame by frame into the
    same tensor where it allows that; a tensor that refuses the write falls
    back to a corrected copy, decided by a probe before any real work.
    """
    from . import colour
    is_uint8 = decoded.dtype == torch.uint8
    try:
        decoded[:, :, :1] = decoded[:, :, :1].clone()
        out = decoded
    except Exception:
        out = decoded.clone()
    cache = {}
    dither = None
    if is_uint8:
        # A correction that changes by fractions of a level per frame would,
        # rounded the same way on every pixel, tip the whole frame to the next
        # level at once.  A fixed spatial dither spreads the rounding across
        # pixels without adding noise that moves.
        height, width = int(out.shape[-2]), int(out.shape[-1])
        pattern = np.random.default_rng(0x1A3).random((height, width), dtype=np.float32) - 0.5
        dither = torch.as_tensor(pattern, dtype=torch.float32, device=out.device).view(1, 1, height, width)
    for index, (gain, offset) in enumerate(per_frame[:int(out.shape[2])]):
        key = (tuple(np.round(np.asarray(gain, dtype=np.float64), 6)),
               tuple(np.round(np.asarray(offset, dtype=np.float64), 6)))
        if key not in cache:
            Q, o = colour.rgb_affine(gain, offset)
            cache[key] = (torch.as_tensor(Q, dtype=torch.float32, device=out.device),
                          torch.as_tensor(o, dtype=torch.float32, device=out.device))
        matrix, shift = cache[key]
        frame = out[:, :, index].float()
        unit = frame.div(255.0) if is_uint8 else frame.add(1.0).mul_(0.5)
        moved = torch.einsum("rk,bkhw->brhw", matrix, unit) + shift.view(1, -1, 1, 1)
        if is_uint8:
            moved = moved.mul_(255.0).add_(dither).round_().clamp_(0.0, 255.0)
        else:
            moved = moved.mul_(2.0).sub_(1.0).clamp_(-1.0, 1.0)
        out[:, :, index] = moved.to(out.dtype)
    return out


def _token_start(index):
    """First pixel frame covered by latent `index` (1 then 4 frames per latent, in 17s)."""
    return (index // _LATENT_PERIOD) * _CLIP_LENGTH + (0, 1, 5, 9, 13)[index % _LATENT_PERIOD]


def _latent_steps(per_frame, total_latents, depth):
    """(A, b) for each of the last `depth` latents, from per-frame corrections.

    Each latent takes the correction at the frames it covers, so the carried
    latents decode to what the corrected frames show.
    """
    from . import colour
    frames = len(per_frame)
    steps = []
    for index in range(max(0, total_latents - depth), total_latents):
        first = _token_start(index)
        last = min(frames - 1, first + (0 if index % 5 == 0 else 3))
        span = per_frame[min(first, frames - 1):last + 1] or per_frame[-1:]
        gain = np.mean([np.asarray(g, dtype=np.float64) for g, _ in span], axis=0)
        offset = np.mean([np.asarray(o, dtype=np.float64) for _, o in span], axis=0)
        steps.append(colour.latent_correction(gain, offset))
    return steps


def _apply_latent_steps(latents, steps):
    """Apply one (A, b) per latent frame to the last len(steps) of [B, C, T, H, W]."""
    from . import colour
    count = len(steps)
    if count == 0:
        return latents
    head = latents[:, :, :latents.shape[2] - count]
    moved = [colour.apply_torch(latents[:, :, latents.shape[2] - count + i:latents.shape[2] - count + i + 1], *step)
             for i, step in enumerate(steps)]
    return torch.cat([head] + moved, dim=2) if head.shape[2] else torch.cat(moved, dim=2)


def _frame_colour_stats(decoded):
    """Y'CbCr stats for every frame of [B, C, F, H, W], pooled like _colour_stats."""
    from . import colour
    is_uint8 = decoded.dtype == torch.uint8
    out = []
    for start in range(0, int(decoded.shape[2]), 8):     # a few frames at a time
        chunk = decoded[0, :, start:start + 8].permute(1, 0, 2, 3).float()
        pooled = torch.nn.functional.adaptive_avg_pool2d(chunk, (_COLOUR_POOL, _COLOUR_POOL))
        unit = (pooled.div(255.0) if is_uint8 else pooled.add(1.0).mul_(0.5))
        unit = unit.permute(0, 2, 3, 1).cpu().numpy()
        out.extend(colour.ycbcr_stats(unit[i]) for i in range(unit.shape[0]))
    return out


def _smooth(values, radius):
    """Centred moving average along axis 0, truncated at the ends."""
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    count = values.shape[0]
    for index in range(count):
        lo, hi = max(0, index - radius), min(count, index + radius + 1)
        out[index] = values[lo:hi].mean(axis=0)
    return out


def _reference_from(stats):
    """The look a shot is held to: average Y'CbCr mean and spread of its opening."""
    return {"mean": np.mean([s["mean"] for s in stats], axis=0).tolist(),
            "std": np.mean([s["std"] for s in stats], axis=0).tolist()}


def _find_cuts(fingerprints, previous_last=None):
    """(cut at the join, [frame indices where a new shot starts inside the window]).

    A cut is a frame-to-frame structural change far above the window's own
    motion: at least CONFIG.cut_threshold on the pooled greyscale signature,
    and at least _CUT_RATIO times the median change.
    """
    diffs = [float((fingerprints[i] - fingerprints[i - 1]).abs().mean())
             for i in range(1, len(fingerprints))]
    typical = float(np.median(diffs)) if diffs else 0.0
    threshold = max(CONFIG.cut_threshold, _CUT_RATIO * typical)
    inside = [i + 1 for i, d in enumerate(diffs) if d > threshold]
    merged = []                         # cuts a few frames apart are one event
    for index in inside:
        if not merged or index - merged[-1] > 4:
            merged.append(index)
    at_join = False
    if previous_last is not None and fingerprints:
        at_join = float((fingerprints[0] - previous_last).abs().mean()) > threshold
    return at_join, merged


def _hold_segment(stats, reference, start, stop):
    """Per-frame (gain, offset) holding frames [start, stop) to `reference`.

    Returns (per_frame list or None, number of frames rejected as a different scene).
    """
    from . import colour
    gains, offsets, rejected = [], [], 0
    for item in stats[start:stop]:
        gain, offset, _ = colour.measure(reference, item, noise=None,
                                         strength=CONFIG.colour_strength,
                                         scene_threshold=CONFIG.colour_scene,
                                         max_correction=CONFIG.colour_max,
                                         axes=CONFIG.colour_axes)
        if gain is None:
            rejected += 1
            gains.append(None)
            offsets.append(None)
        else:
            gains.append(np.asarray(gain))
            offsets.append(np.asarray(offset))
    valid = [i for i, g in enumerate(gains) if g is not None]
    if not valid:
        return None, rejected
    for index in range(len(gains)):
        if gains[index] is None:
            # A frame rejected on its own (a flash, something passing close):
            # take its neighbours' correction.
            nearest = min(valid, key=lambda v: abs(v - index))
            gains[index], offsets[index] = gains[nearest], offsets[nearest]
    gains = _smooth(np.stack(gains), _LOCK_SMOOTH)          # within the segment only
    offsets = _smooth(np.stack(offsets), _LOCK_SMOOTH)
    return [(gains[i], offsets[i]) for i in range(len(gains))], rejected


def _measure_colour(decoded):
    """Hold each continuous shot in this window to the look of its own opening.

    Where a shot continues across windows, each window comes out a little
    darker, more contrasty and more saturated than the frames it continues,
    and the carry hands that on, so it compounds.  The reference is the
    opening of the current continuous shot, wherever that is: window 1, a cut
    at a join, or a cut part way through a window.  Every frame is measured
    against its shot's reference over the whole frame and corrected back to
    it, smoothed over time within the shot; the latents carried into the next
    window get the same correction, so a continuing shot starts each window
    from its own opening look.
    """
    frames = int(decoded.shape[2])
    stats = _frame_colour_stats(decoded)
    prints = [_fingerprint(decoded[0, :, i]) for i in range(frames)]
    previous = STATE.signatures[-1] if (STATE.signatures is not None
                                        and STATE.source_generation == STATE.generation - 1) else None
    at_join, cuts = _find_cuts(prints, previous)
    opening = lambda a, b: stats[a:min(b, a + _LOCK_REFERENCE_FRAMES)]

    if STATE.lock_reference is None and STATE.colour_seed is not None and not at_join:
        STATE.lock_reference = STATE.colour_seed
        _log("colour: holding the look of the video being continued")
    if STATE.lock_reference is None or at_join:
        reason = "a cut at the join" if STATE.lock_reference is not None else "the first window"
        STATE.lock_reference = _reference_from(opening(0, cuts[0] if cuts else frames))
        _log(f"colour: new shot ({reason}), its opening is the reference")

    bounds = [0] + cuts + [frames]
    per_frame = [None] * frames
    notes = []
    identity = (np.ones(3), np.zeros(3))
    for number, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
        if number > 0:
            STATE.lock_reference = _reference_from(opening(start, stop))
            notes.append(f"cut at frame {start}, new shot from there")
        reference = {"mean": np.asarray(STATE.lock_reference["mean"]),
                     "std": np.asarray(STATE.lock_reference["std"])}
        segment, rejected = _hold_segment(stats, reference, start, stop)
        if segment is None or rejected > _LOCK_SCENE_SHARE * (stop - start):
            # The look changed too much for one shot without a visible cut (a
            # lighting change, a whip pan).  Hold from here instead.
            STATE.lock_reference = _reference_from(opening(start, stop))
            notes.append(f"look changed from frame {start}, held from there")
            reference = {"mean": np.asarray(STATE.lock_reference["mean"]),
                         "std": np.asarray(STATE.lock_reference["std"])}
            segment, _ = _hold_segment(stats, reference, start, stop)
        for offset_index in range(stop - start):
            per_frame[start + offset_index] = segment[offset_index] if segment else identity

    corrected = _correct_pixels_ramp(decoded, per_frame)
    total = int(STATE.decoding_latents or 0) or _latent_count(frames)
    STATE.colour_step = _latent_steps(per_frame, total, _cache_depth(total))
    STATE.stats["colour_applied"] += 1
    first, last = per_frame[0], per_frame[-1]
    _log(f"colour held: start offset {np.round(np.asarray(first[1]) * 255, 1).tolist()} "
         f"gain {np.round(np.asarray(first[0]), 3).tolist()}, end offset "
         f"{np.round(np.asarray(last[1]) * 255, 1).tolist()} gain "
         f"{np.round(np.asarray(last[0]), 3).tolist()}"
         f"{'; ' + '; '.join(notes) if notes else ''}")
    return corrected


def _latent_count(frames):
    """Latents that decode to `frames` frames (17m + 5 frames -> 2 + 5m)."""
    return 2 + ((max(5, int(frames)) - 5) // _CLIP_LENGTH) * _LATENT_PERIOD


def _job_discard():
    """Discard Last Frames for the running job, 0 when it is not known."""
    job = JOB.get()
    try:
        return max(0, int((job or {}).get("discard_last_frames") or 0))
    except (TypeError, ValueError):
        return 0


def _check_alignment(signatures, signature, incoming, expected_offset=_ALIGN_EXPECTED_OFFSET):
    """Locate the join frame in the cached decoded tail.

    The next window hands back `continuation[:, -overlap:-1]`, whose last frame
    is the decoded window's frame -2 - the join frame itself is excluded and
    pinned separately.  So the incoming frame must match offset -2 of the
    cached tail, and matching anywhere else means the cached latents do not end
    where the emitted video ends.

    Testing that one offset alone cannot detect this.  Neighbouring frames of a
    slow shot differ by far less than the tolerance, so a tail off by several
    frames passes, the carried block is then placed as though it ended at the
    join, and the model is handed footage from after the join as though it were
    before it - which it duly renders again.  Searching the run turns that
    silent misalignment into a number.

    Returns (ok, detail).  `detail` names the measured skew in frames when the
    join sits somewhere other than where it must.
    """
    if signatures is None or signatures.shape[0] < abs(expected_offset):
        if signature is None or expected_offset != _ALIGN_EXPECTED_OFFSET:
            return False, "no join fingerprint captured"
        difference = float((incoming - signature).abs().mean())
        if difference > CONFIG.match_tolerance:
            return False, (f"content mismatch (join difference {difference:.4f} > "
                           f"{CONFIG.match_tolerance:.4f})")
        return True, None

    errors = (signatures - incoming[None]).abs().mean(dim=(1, 2))
    depth = int(signatures.shape[0])
    expected_row = depth + expected_offset
    expected_error = float(errors[expected_row])
    best_row = int(errors.argmin())
    best_error = float(errors[best_row])
    skew = best_row - expected_row

    if not CONFIG.align_search:
        pass
    elif skew != 0 and best_error < expected_error * _ALIGN_DECISIVE:
        # Decisively better elsewhere: the tail really is displaced.  The join
        # frame sitting EARLIER in the cached tail than offset -2 (a negative
        # skew) means the cache holds frames from after the join, which is the
        # case that gets rendered a second time.  Note only +1 is reachable in
        # the other direction, since the cached fingerprints stop at offset -1.
        direction = ("cached latents run past the end of the emitted video"
                     if skew < 0 else "cached latents stop short of it")
        return False, (f"tail misaligned by {skew:+d} frame(s) - {direction} "
                       f"(join matches offset {best_row - depth:+d} at "
                       f"{best_error:.4f}, expected {expected_offset:+d} "
                       f"at {expected_error:.4f})")

    if expected_error > CONFIG.match_tolerance:
        return False, (f"content mismatch (join difference {expected_error:.4f} > "
                       f"{CONFIG.match_tolerance:.4f}); cache belongs to a "
                       f"different video")
    return True, None


def _joint_decode(vae, latents, args, kwargs):
    """Decode the window with the carried history in front, then cut it off.

    Only in the takeover layout, where the carried block is whole decoder
    chunks ending one frame before the join and the window's first latent is
    the join's 1-frame latent: prefix + window is then a valid latent run, and
    decoding it in one go cross-fades the window's first frames with the
    history the way the middle of a clip is, instead of starting them cold.
    Returns None to decode normally.
    """
    prefix = STATE.takeover_prefix
    STATE.takeover_prefix = None
    if prefix is None or not CONFIG.takeover:
        return None
    try:
        expected = (STATE.plan or {}).get("expected_latents")
        if (latents.ndim != 5 or tuple(prefix.shape[-2:]) != tuple(latents.shape[-2:])
                or prefix.shape[2] % _LATENT_PERIOD or (expected and latents.shape[2] != expected)):
            _log("joint decode skipped: the window is not laid out as expected")
            return None
        joined = torch.cat([prefix.to(device=latents.device, dtype=latents.dtype), latents], dim=2)
        decoded = _ORIGINALS["decode"](vae, joined, *args, **kwargs)
        skip = (prefix.shape[2] // _LATENT_PERIOD) * _FRAMES_PER_CHUNK
        frames = 2 + (latents.shape[2] - 2) // _LATENT_PERIOD * _FRAMES_PER_CHUNK + 3
        if decoded.shape[2] - skip != frames:
            _log(f"joint decode skipped: {decoded.shape[2] - skip} frames where {frames} were expected")
            return None
        STATE.stats["joint_decode"] += 1
        _log(f"decoded the window with {skip} frames of the previous one ahead of it")
        return decoded[:, :, skip:]
    except Exception as error:                                   # never break a render
        _log(f"joint decode failed, decoding normally: {error!r}")
        return None


def _patched_decode(self, latents, *args, **kwargs):
    # Everything beyond the latents is forwarded untouched.  Wan2GP added a
    # second positional argument (uint8_rounding) to MiniMaxH3VideoVAE.decode
    # after this plugin was written, and a replacement that mirrored the old
    # signature raised TypeError on every decode.  Forwarding *args/**kwargs
    # cannot break that way again.
    if _PASSTHROUGH.get() is not None:
        return _ORIGINALS["decode"](self, latents, *args, **kwargs)
    decoded = _joint_decode(self, latents, args, kwargs)
    if decoded is None:
        decoded = _ORIGINALS["decode"](self, latents, *args, **kwargs)
    STATE.colour_step = None
    STATE.decoding_latents = int(latents.shape[2])
    if CONFIG.colour:
        try:
            # NOT `_measure_colour(decoded) or decoded`: `or` calls bool() on
            # the tensor, which raises "Boolean value of Tensor with more than
            # one element is ambiguous", and the except below then swallowed it
            # and silently disabled colour on every window.
            corrected = _measure_colour(decoded)
            if corrected is not None:
                decoded = corrected
        except Exception as error:          # never break a render
            # Loud, and worded as a defect rather than a fallback.  The previous
            # wording read like a considered decline, which is how a hard error
            # went unnoticed on every window for a whole release.
            _log(f"BUG: colour correction raised and was skipped for this "
                 f"window - {type(error).__name__}: {error}. This is not a "
                 f"normal fallback; please report it.")
            STATE.colour_step = None
    try:
        tail = latents[:, :, -_cache_depth(latents.shape[2]):].detach().to(torch.float32).cpu()
        if STATE.colour_step is not None:
            # The frames were corrected; the cached latents have to move with
            # them or the next window is conditioned on a grade the video no
            # longer has.
            tail = _apply_latent_steps(tail, STATE.colour_step)
        # Frame -2 of the decoded window is the last frame the next window will
        # hand back as history (continuation[:, -overlap:-1] excludes the join
        # frame itself).  Fingerprinting it lets the next window confirm these
        # latents belong to the video it is actually continuing.
        signature = _fingerprint(decoded[0, :, -2]) if decoded.shape[2] >= 2 else None
        # A run of tail fingerprints, newest last, so the next window can find
        # *where* the join frame sits instead of only asking whether it is where
        # it was assumed to be.
        depth = min(_ALIGN_SEARCH_FRAMES, int(decoded.shape[2]))
        signatures = (torch.stack([_fingerprint(decoded[0, :, i])
                                   for i in range(-depth, 0)]) if depth >= 2 else None)
        with STATE.lock:
            STATE.latents = tail
            STATE.signature = signature
            STATE.signatures = signatures
            STATE.source_generation = STATE.generation
            STATE.total_latents = int(latents.shape[2])
            plan = STATE.plan or {}
            STATE.source_window_no = plan.get("window_no")
            STATE.usable = bool(plan.get("usable"))
            STATE.expected_latents = plan.get("expected_latents")
            STATE.source_discard = _job_discard()
    except Exception as error:                                   # never break a render
        _log(f"capture failed, continuing without cache: {error!r}")
        STATE.invalidate("capture error")
    return decoded


def _audio_overrun(count):
    """Audio latents decoded past the window's last video frame.

    Newer Wan2GP adds a short audio tail after the last frame.  The plan knows
    how many latents the frames themselves span; anything beyond that, up to
    _MAX_AUDIO_TAIL_LATENTS, is the tail.  Older Wan2GP has none, so this is 0.
    """
    target = (STATE.plan or {}).get("audio_target_latents")
    if not target:
        return 0
    extra = int(count) - int(target)
    return extra if 0 < extra <= _MAX_AUDIO_TAIL_LATENTS else 0


def _patched_audio_decode(self, latents, *args, **kwargs):
    waveform = _ORIGINALS["audio_decode"](self, latents, *args, **kwargs)
    if _PASSTHROUGH.get() is not None:
        return waveform
    try:
        count = int(latents.shape[-1])
        overrun = _audio_overrun(count)
        end = count - overrun
        tail = latents[..., :end][..., -_MAX_CACHED_AUDIO_LATENTS:].detach().to(torch.float32).cpu()
        at_join = waveform
        if overrun and count:
            # Signature of the half second ending at the join, not at the tail.
            at_join = waveform[..., :int(round(int(waveform.shape[-1]) * end / count))]
        with STATE.lock:
            STATE.audio_latents = tail
            STATE.audio_signature = _audio_signature(at_join)
            STATE.audio_generation = STATE.generation
            STATE.audio_source_window_no = (STATE.plan or {}).get("window_no")
    except Exception as error:
        _log(f"audio capture failed, continuing without it: {error!r}")
        with STATE.lock:
            STATE.audio_latents = None
    return waveform


def _patched_encode_audio(self, waveform, *args, **kwargs):
    """Substitute the previous window's audio latents for the continuation encode.

    Unlike the video side there is no coordinate work to do: _fill_audio_positions
    lays audio conditions out as origin + arange(length), a uniform integer axis
    with none of the (1, 4, 4, 4, 4) periodicity, so a tail slice needs no phase
    alignment and no translation.

    The real encode is always run.  It fixes the latent count that the pipeline's
    history/boundary split is computed from, and it is the fallback.  The audio
    VAE is cheap beside the video one, so paying for it and discarding the result
    buys a lot of safety for little time.
    """
    native = _ORIGINALS["encode_audio"](self, waveform, *args, **kwargs)
    if _PASSTHROUGH.get() is not None:
        return native

    STATE.audio_encode_calls += 1
    if not (CONFIG.enable and CONFIG.audio) or _RIVAL:
        return native
    if STATE.audio_encode_calls != 1:
        # generate() encodes the continuation first; later calls are reference
        # clips or a supplied soundtrack, which must not be replaced.
        return native
    if not (STATE.plan or {}).get("continuation_audio"):
        # ...but "first call" only means the continuation when this window has a
        # continuation to encode.  Without one, call 1 is a reference clip or the
        # target audio condition, and substituting into either would be wrong.
        return native

    with STATE.lock:
        cached = STATE.audio_latents
        signature = STATE.audio_signature
        source = STATE.audio_generation
        source_window_no = STATE.audio_source_window_no
        generation = STATE.generation
        usable = STATE.usable

    if cached is None or source != generation - 1 or not usable:
        return native

    # The envelope match below cannot carry this on its own: silence matches
    # silence by design, so a silent job following a silent job would otherwise
    # inherit the previous video's audio latents at its first window.
    continues, reason = _continues_cached_window(source_window_no)
    if not continues:
        _log(f"re-encoding continuation audio ({reason})")
        return native

    if STATE.source_discard:
        # Discarded frames rarely land on the 40/s audio grid (4 frames at 24 fps
        # are 6 2/3 latents), so the cached audio cannot be cut at the join.
        _log(f"re-encoding continuation audio (Discard Last Frames "
             f"{STATE.source_discard} does not end on an audio latent)")
        return native

    matched, reason = _audio_matches(signature, _audio_signature(waveform))
    if not matched:
        _log(f"re-encoding continuation audio ({reason})")
        return native

    wanted = int(native.shape[-1])
    if wanted > cached.shape[-1]:
        _log(f"re-encoding continuation audio (cached tail holds "
             f"{cached.shape[-1]} latents, need {wanted})")
        return native
    if tuple(cached.shape[1:-1]) != tuple(native.shape[1:-1]):
        _log(f"re-encoding continuation audio (layout {tuple(cached.shape)} vs "
             f"{tuple(native.shape)})")
        return native

    extra = _audio_extension(wanted, cached.shape[-1], self)
    STATE.audio_extra = extra
    total = wanted + extra
    STATE.stats["audio_engaged"] += 1
    if extra:
        _log(f"carried {total} audio latents from the previous window "
             f"({total / _audio_latent_fps():.2f}s context, {extra} beyond the "
             f"native window of {wanted})")
    else:
        _log(f"carried {wanted} audio latents from the previous window")
    return cached[..., -total:].to(dtype=native.dtype).clone()


# --------------------------------------------------------------------------
# 2. window bookkeeping
# --------------------------------------------------------------------------

def _audio_latent_fps():
    try:
        from models.minimax_h3.pipeline import AUDIO_LATENT_FPS
        rate = float(AUDIO_LATENT_FPS)
        return rate if rate > 0 else _AUDIO_LATENT_FPS_FALLBACK
    except Exception:
        return _AUDIO_LATENT_FPS_FALLBACK


def _reference_rows_locatable():
    """Whether the packed-sequence type reports the row totals we need.

    Checked at the type rather than at an instance, because _audio_extension has
    to decide before any sequence exists.
    """
    global _REF_ROWS_LOCATABLE
    if _REF_ROWS_LOCATABLE is None:
        try:
            fields = set(getattr(_PACKING.MiniMaxH3PackedSequence,
                                 "__dataclass_fields__", {}) or {})
            _REF_ROWS_LOCATABLE = {"num_condition_video_rows",
                                   "num_condition_audio_rows"} <= fields
        except Exception:
            _REF_ROWS_LOCATABLE = False
    return _REF_ROWS_LOCATABLE


_REF_ROWS_LOCATABLE = None


def _audio_extension(native_latents, cached_latents, pipeline=None):
    """How many audio latents to carry beyond what the native encode produced.

    Wan2GP sizes the audio condition from the video overlap - `overlap_samples =
    round(continuation_count / fps * AUDIO_SAMPLE_RATE)` - so at overlap 18 and
    24 fps the model gets 0.75s of audio context.  That is generous for motion
    continuity and short for sound: less than a bar at most tempos, and shorter
    than many single spoken words.

    Returning a longer tail gives the history block more to work with, but the
    builder lays a history block *forward* from `float(text_len)`, so the extra
    latents would run past `target_origin` and collide with the target audio.
    _apply_origin_shift compensates by moving everything except the audio
    history later by the same amount, which is only reached when the video carry
    engaged and fix_coords is on - hence the gate here.
    """
    if CONFIG.audio_context <= 0:
        return 0
    if getattr(pipeline, "reference_mode", False) and not _reference_rows_locatable():
        # Ref2VA.  build_ref2va_packed_sequence lays the reference tokens on the
        # time axis *below* the history, advancing a cursor from float(text_len)
        # to time_cursor (packing.py:236-263), and only then places the history
        # block at history_time = time_cursor and the target at
        # time_cursor + _reference_t_span (271, 268).
        #
        # The block translation is unaffected by that: both derive from the same
        # cursor, so the discrepancy it corrects is invariant to where the cursor
        # ended up.  The extension is not - it has to hold the reference rows
        # still while everything else moves, which means finding their span.  If
        # the builder does not report the row totals that span is derived from,
        # there is nothing safe to do but decline.
        _log("extended audio context is not applied on Ref2VA: this build does "
             "not report the condition row counts needed to locate the "
             "reference tokens; using the native audio window")
        return 0
    if not CONFIG.video:
        # Worth naming: in audio-only mode there is no carried video block, so
        # _apply_origin_shift never runs and the compensating shift has nowhere
        # to live.  Without this the generic message below sends people looking
        # at fix_coords, which is not what is holding it.
        _log("extended audio context needs the video carry, which is off "
             "(H3LC_VIDEO=0); using the native audio window")
        return 0
    if STATE.active_latents is None or not CONFIG.fix_coords:
        _log("extended audio context needs the video carry and fix_coords; "
             "using the native audio window for this window")
        return 0
    desired = int(round(CONFIG.audio_context * _audio_latent_fps()))
    if desired <= int(native_latents):
        _log(f"audio context of {CONFIG.audio_context:g}s is already covered by the "
             f"native window ({int(native_latents)} latents, "
             f"{int(native_latents) / _audio_latent_fps():.2f}s); nothing added")
        return 0
    extra = desired - int(native_latents)
    available = max(0, int(cached_latents) - int(native_latents))
    if extra > available:
        _log(f"extended audio context trimmed to the cached tail "
             f"({available} spare latents, wanted {extra})")
        extra = available
    return extra


def _window_number(values):
    """wgp.py's 1-based window counter, or None if the caller did not send one.

    The pipeline itself never reads this - it lands in generate's **kwargs and is
    ignored - but it is the only argument that distinguishes the second window of
    a video from the first window of the next queue entry.
    """
    raw = values.get("window_no")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _continues_cached_window(source_window_no):
    """Is the current window the immediate successor of the cached one?

    Returns (ok, reason).  The generation counter alone only proves the cache
    came from the previous generate() call, which a queued job's first window
    also satisfies.
    """
    current = STATE.window_no
    if current is None:
        if CONFIG.require_window_no:
            return False, ("caller did not report window_no, so a new job cannot be "
                           "told from a new window (H3LC_REQUIRE_WINDOW_NO=0 to allow)")
        return True, None
    if source_window_no is None:
        if CONFIG.require_window_no:
            return False, "cached window has no window_no recorded"
        return True, None
    if current != source_window_no + 1:
        return False, (f"cache is from window {source_window_no} of its job, "
                       f"this is window {current} - different job")
    return True, None


def _end_image_problem(values, history_count, aligned):
    """Whether this window's end image says something the plugin cannot cache.

    Wan2GP puts an end image on the window's last *emitted* frame:
    `image_end_frame_position = current_video_length - tail_trim_frames - 1`
    (wgp.py:7724), passed through to the pipeline, which subtracts
    history_count and conditions that target frame (pipeline.py:801).  So under
    wgp's own formula the distance from the end image to the last target frame
    is the tail trim - and a trimmed window generates more frames than it
    emits, which means its cached tail ends past the video the next window
    hands back as continuation.  The carried block would then be placed as
    though its last latent were the join, and the overrun is rendered twice.

    `_check_alignment` would normally catch that as a negative skew, but a
    window pinned to an end image tends to settle onto a held composition, and
    on a flat tail the search abstains by design and leaves the decision to
    H3LC_MATCH_TOL - which a static shot passes.  Hence a guard here, where the
    position is a number rather than an inference from pixels.

    Classified rather than subtracted blindly.  The position is an ordinary
    pipeline argument and a plugin can set it to anything; Sliding Window
    Anchor's own notes contemplate an end image landing on the *opening* frame.
    Reading that as a trim would produce a large, confident and wrong number.
    """
    if values.get("image_end") is None:
        return None
    raw = values.get("image_end_frame_position")
    if raw is None:
        # pipeline.py:799 puts it on aligned_target_frames - 1 with a "last"
        # anchor.  That is the window's real last frame, so there is no trim
        # to infer and nothing to refuse.
        return None
    try:
        index = int(raw) - int(history_count)
    except (TypeError, ValueError):
        return "image_end_frame_position is not a whole number"

    if index < 0:
        # pipeline.py:801 does not bounds-check this the way the injection loop
        # at 804 does, so the condition would land below target_origin, inside
        # the span the history block occupies.  Not a layout this plugin models.
        return (f"end image resolves to target frame {index}, below the target; "
                f"the pipeline does not bounds-check that position and the "
                f"resulting layout is not one this plugin models")
    if 0 < index < aligned - 1:
        return (f"end image sits at target frame {index} of {aligned - 1}; under "
                f"wgp.py's own formula that is a {aligned - 1 - index}-frame tail "
                f"trim, so the window emits fewer frames than it generates and "
                f"the cached tail would overrun the emitted video")
    # index == 0 is a condition on the join frame, not a trim.  That is a carry
    # problem, not a caching one, and _frame_zero_collision reports it.
    return None


def _frame_zero_collision(values, history_count):
    """Image conditions the pipeline will place on target frame 0, if any.

    Target frame 0 is the join.  Wan2GP already pins it with a "first" anchor
    (`_add_image_condition(continuation[:, -1:], 0, ...)`, pipeline.py:794) and
    the corrected carried block's last latent also reaches it - two blocks
    describing one instant from two directions, which reinforce.  A third makes
    the model hold on it, and the join gains a frame that does not move.

    The injection loop at pipeline.py:802 subtracts history_count from every
    position it is given and keeps what lands inside the target, so a position
    of exactly history_count resolves to frame 0.  That is not a corner case:
    it is where the Sliding Window Anchor plugin deliberately puts its anchor,
    and Wan2GP's own `L` injection places a frame at the end of a window, which
    is handed to the *next* window at that same relative position.

    So this is keyed to the collision rather than to a plugin name.  Whatever
    put a condition on frame 0, the carried block is the assertion to drop -
    leaving the native pin and the injected frame, which is what the injecting
    feature expects to be working against.  Returns a reason, or None.
    """
    culprits = []
    images = list(values.get("frames_to_inject") or ())
    positions = list(values.get("frames_relative_positions_list") or ())
    for image, position in zip(images, positions):
        if image is None:                    # an unfilled slot, not a frame
            continue
        try:
            if int(position) - int(history_count) == 0:
                culprits.append("an injected frame")
        except (TypeError, ValueError):
            continue

    if values.get("image_end") is not None:
        try:
            if int(values.get("image_end_frame_position")) - int(history_count) == 0:
                culprits.append("an end image")
        except (TypeError, ValueError):
            pass

    if not culprits:
        return None
    what = " and ".join(sorted(set(culprits)))
    return (f"{what} is conditioned on target frame 0, where Wan2GP already pins "
            f"the join frame and the carried block would also reach; carrying "
            f"would make a third block assert that instant")


def _float_or_none(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int_or_none(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _plan_window(pipeline, args, kwargs):
    """Mirror the few lines of generate() that decide the target frame count.

    Returns a dict describing what this window will produce.  Everything here is
    re-validated against the captured tensor before it is ever used, so a drift
    against upstream degrades to a fall back rather than a wrong layout.
    """
    from models.minimax_h3.pipeline import _as_video, video_latent_frames  # noqa: F401
    from shared.utils.frame_scheduler import floor_frame_count, normalize_frame_count, normalize_overlap

    # Read straight from kwargs rather than binding against a signature.  Another
    # plugin may already have wrapped generate() with a (self, *args, **kwargs)
    # shim, and unless it set __wrapped__ inspect.signature would report that
    # shim's parameters instead of the pipeline's.  wgp.py passes everything by
    # keyword, so this is both simpler and more robust.
    values = {} if args else kwargs
    window_no = _window_number(values)
    # What the saved-latent paths compare against: the render geometry.
    geometry = {"fps": _float_or_none(values.get("fps")),
                "width": _int_or_none(values.get("width")),
                "height": _int_or_none(values.get("height")),
                "continuation_frames": 0}

    def refuse(reason, **extra):
        # An unusable window can still carry from its predecessor: "usable"
        # describes what this window hands forward, not what it may receive.
        result = {"usable": False, "window_no": window_no, "reason": reason,
                  "carry_blocked": None, **geometry}
        result.update(extra)
        return result

    if args:
        return refuse("generate() called positionally, cannot read arguments")

    if "2" in (values.get("audio_prompt_type") or ""):
        return refuse("audio-from-control-video rewrites frame_num")

    if window_no is None and CONFIG.require_window_no:
        return refuse("generate() did not report window_no, so job boundaries "
                      "cannot be established")

    # Mirror the pipeline's own overlap arithmetic, flag and all.  It uses an
    # offset of 5 instead of 1 when the control-latent continuation experiment is
    # enabled for a control model; that constant ships disabled, but reading it
    # keeps this from silently disagreeing if it is ever turned on.
    raw_prefix = int(values.get("prefix_frames_count") or 0)
    control_layers = bool(getattr(getattr(pipeline, "transformer", None), "control_layers", None))
    overlap_offset = 5 if (_CONTROL_LATENT_CONTINUATION and control_layers
                           and raw_prefix % _CLIP_LENGTH == 5) else 1
    prefix, overlap_error = normalize_overlap(raw_prefix, _CLIP_LENGTH, overlap_offset)
    if overlap_error:
        return refuse("invalid overlap")

    input_video = values.get("input_video")
    continuation = _as_video(input_video) if input_video is not None and prefix > 0 else None
    count = min(prefix, continuation.shape[1]) if continuation is not None and values.get("image_start") is None else 0
    if count and count < prefix:
        count = floor_frame_count(count, 1, _CLIP_LENGTH,
                                  overlap_offset if count >= overlap_offset else 1)
    if count and control_layers and not bool(values.get("refinement_mode")):
        # A control model keeps the overlap on the target timeline as fixed
        # control latents and visible pixels, and sets its continuation count to
        # zero: _add_video_history is never called, so there is no history block
        # to carry into or to seed from a saved entry.
        return refuse("this model conditions the continuation with control latents "
                      "instead of H3's history block, so there is nothing to carry")
    history_count = continuation[:, -count:-1].shape[1] if count > 1 else 0

    geometry["continuation_frames"] = int(count)

    # generate() rounds frame_num onto H3's 5 + 17n grid before counting frames;
    # without the same step an off-grid count reads as a truncated window.
    frame_num = int(values.get("frame_num") or 0)
    if frame_num > 0:
        frame_num = normalize_frame_count(frame_num, 5, _CLIP_LENGTH, 5)
    target_frames = frame_num - history_count
    if target_frames <= 0:
        return refuse("no target frames")
    aligned = normalize_frame_count(target_frames, 5, _CLIP_LENGTH, 5)

    if aligned != target_frames:
        # The emitted video is truncated relative to the latent grid, so the last
        # cached latent no longer ends on the join frame.
        return refuse(f"target {target_frames} != aligned {aligned}")

    end_problem = _end_image_problem(values, history_count, aligned)
    if end_problem:
        return refuse(end_problem)

    return {"usable": True,
            "window_no": window_no,
            **geometry,
            # Conditions the pipeline will place on target frame 0.  Blocks the
            # carry rather than the cache: it is this window's conditioning that
            # is crowded, and its own output is a perfectly good window to cache.
            "carry_blocked": _frame_zero_collision(values, history_count),
            # generate() only encodes continuation audio when it has both
            # continuation frames and a waveform (pipeline._waveform returns None
            # for a missing input_waveform).  Without both, the first
            # _encode_audio call of this window is something else entirely.
            "continuation_audio": bool(count) and values.get("input_waveform") is not None,
            # Audio latents the window's frames span; any more is Wan2GP's tail.
            "audio_target_latents": (int(round(aligned / geometry["fps"] * _audio_latent_fps()))
                                     if geometry.get("fps") else None),
            "expected_latents": video_latent_frames(aligned)}


def _patched_generate(self, *args, **kwargs):
    global _RIVAL
    if _PASSTHROUGH.get() is not None or kwargs.get("refinement_mode"):
        # A refinement pass, not a window: see _PASSTHROUGH.
        token = _PASSTHROUGH.set(True)
        try:
            STATE.stats["passthrough"] += 1
            return _ORIGINALS["generate"](self, *args, **kwargs)
        finally:
            _PASSTHROUGH.reset(token)
    previous, previous_usable = STATE.generation, (STATE.plan or {}).get("usable")
    # Never let a join frame pending from an earlier call reach this one's
    # start image.
    STATE.anchor_latents = None
    STATE.takeover = False
    STATE.takeover_prefix = None
    STATE.join_latent = None
    if ON_GENERATE is not None:
        try:
            ON_GENERATE()
        except Exception as error:                              # never break a render
            _log(f"could not check the Wan2GP job hooks: {error!r}")
    if (CONFIG.save and STORE is not None and previous and previous_usable
            and STATE.save_seen != previous and STATE.save_warned != previous):
        # The previous window produced latents worth saving, but no save hook ran
        # for it.  Loud, because the alternative is silence.
        STATE.save_warned = previous
        _log("the previous window's video was never seen by the save hook, so nothing "
             "was saved. Open the plugin panel and press Diagnostics; this usually "
             "means another plugin replaced Wan2GP's save functions, or this build "
             "saves videos by a route the plugin does not hook.")
    with STATE.lock:
        STATE.generation += 1
    STATE.active_latents = None
    STATE.audio_extra = 0
    STATE.audio_encode_calls = 0
    STATE.plan = _plan_window(self, args, kwargs)
    STATE.window_no = STATE.plan.get("window_no")
    STATE.stats["windows"] += 1
    if CONFIG.trace:
        _log(f"trace: generate() window={STATE.window_no} "
             f"usable={STATE.plan.get('usable')} continuing="
             f"{STATE.plan.get('continuation_frames')} job={'yes' if JOB.get() else 'NO'}")

    rival = _rival_module()
    if rival:
        # Another copy of this carry logic wrapped us after we installed.  Two
        # carries on one history block is never right, so this one stands aside.
        if rival != _RIVAL or (STATE.window_no or 0) <= 1:
            _log(f"standing aside: {rival} is also installed and patches the same "
                 f"methods. Disable one of the two plugins.")
        _RIVAL = rival
        with STATE.lock:
            STATE.invalidate("another carry plugin is active")
        return _ORIGINALS["generate"](self, *args, **kwargs)
    _RIVAL = None

    # A queue entry's first window is a different video from whatever ran last,
    # even though it is the next generate() call.  So is each repeat of a prompt,
    # which restarts wgp.py's window loop at 1.  Drop the cache here rather than
    # rely on the join fingerprint to notice: two jobs continuing the same source
    # clip, or a static shot regenerated with a new seed, can be similar enough
    # to pass a pixel comparison.
    if STATE.window_no is None:
        # With H3LC_REQUIRE_WINDOW_NO=0 the caller has opted into fingerprint-only
        # matching, so leave the cache alone - dropping it here would make that
        # setting mean "never carry" rather than "carry on the old evidence".
        if CONFIG.require_window_no:
            with STATE.lock:
                STATE.invalidate("caller did not report window_no")
    elif STATE.window_no <= 1:
        with STATE.lock:
            STATE.invalidate(f"window {STATE.window_no} starts a new video")
            STATE.restored = None
        if STATE.window_no == 1:
            try:
                _restore_saved({} if args else kwargs)
            except Exception as error:                          # never break a render
                _log(f"restoring saved latents failed, re-encoding instead: {error!r}")
                with STATE.lock:
                    STATE.invalidate("restore error")

    if not STATE.plan.get("usable"):
        _log(f"this window will not be cached: {STATE.plan.get('reason')}")
    return _ORIGINALS["generate"](self, *args, **kwargs)


# --------------------------------------------------------------------------
# 2b. across jobs: restore on window 1, snapshot at export
# --------------------------------------------------------------------------

def _rival_module():
    """Another installed copy of this carry logic, by module name, or None.

    Sliding Window Latents is this plugin's parent and patches the same seven
    bindings.  Recognised structurally rather than by name, so a renamed folder
    is still caught: a module with its own _REPLACEMENTS table, a non-empty
    _ORIGINALS (i.e. installed) and a _patched_generate.
    """
    mine = globals()
    for name, module in list(sys.modules.items()):
        namespace = getattr(module, "__dict__", None)
        if namespace is None or namespace is mine:
            continue
        if (isinstance(namespace.get("_REPLACEMENTS"), dict)
                and namespace.get("_ORIGINALS")
                and callable(namespace.get("_patched_generate"))):
            return name
    return None


def _continue_plugin_continuing():
    """Whether H3 Latent Continue is running a "Continue with latent" job right now.

    That plugin supplies its own history from its own checkpoint.  Seeding ours
    as well would put two history blocks into one window.
    """
    for module in list(sys.modules.values()):
        job_var = getattr(module, "JOB", None)
        if isinstance(job_var, ContextVar) and job_var.name == "h3_latent_job":
            job = job_var.get()
            if isinstance(job, dict) and (job.get("options") or {}).get("continue"):
                return True
    return False


def _restore_problem(manifest, plan, job):
    """Why a saved entry cannot seed this job, or None."""
    saved_fps, fps = manifest.get("fps"), plan.get("fps")
    if fps and abs(float(saved_fps) - float(fps)) > 1e-3:
        return f"saved at {saved_fps:g} fps, this job renders at {fps:g} fps"
    width, height = plan.get("width"), plan.get("height")
    if width and height and (manifest.get("width"), manifest.get("height")) != (width, height):
        return (f"saved at {manifest.get('width')}x{manifest.get('height')}, this job "
                f"renders at {width}x{height}")
    saved_model = manifest.get("base_model_type") or manifest.get("model_type")
    model = job.get("base_model_type") or job.get("model_type")
    if saved_model and model and saved_model != model:
        return f"saved with {saved_model}, this job uses {model}"
    return None


def _incoming_join_fingerprint(values, plan):
    """Fingerprint of the frame the continuation hands back as its last history frame.

    The same frame _take_cached compares against the cached tail, computed here so
    a saved entry can be recognised by what it shows rather than by its file name.
    """
    video, count = values.get("input_video"), int(plan.get("continuation_frames") or 0)
    if video is None or count < 2:
        return None
    try:
        from models.minimax_h3.pipeline import _as_video
        video = _as_video(video)
        return _fingerprint(video[:, -count:-1][:, -1])
    except Exception:
        return None


def _match_saved_by_content(plan, job, incoming, limit=50):
    """Find an entry whose decoded tail is what this source file shows.

    Wan2GP hands a job the file it was given, which is not always the file that
    was written: the interface copies videos into its own cache under a sanitized
    name, and a copy's bytes need not match either.  The hash lookup then misses
    although the latents are the right ones.  This falls back to the join-frame
    search used between windows, over entries that already match this job's fps,
    resolution and model, so a wrong pairing is rejected the same way.
    """
    if incoming is None or STORE is None:
        return None
    checked = 0
    for sha, _size, _mtime in STORE.entries():
        if checked >= limit:
            break
        manifest = STORE.manifest(sha)
        if not manifest or _restore_problem(manifest, plan, job):
            continue
        checked += 1
        try:
            signatures = torch.from_numpy(np.ascontiguousarray(STORE.signatures(sha),
                                                               dtype=np.float32))
        except Exception:
            continue
        offset = _ALIGN_EXPECTED_OFFSET - (_int_or_none(manifest.get("discard_last_frames")) or 0)
        aligned, _detail = _check_alignment(signatures, None, incoming, offset)
        if aligned:
            return sha
    return None


def _restore_saved(values):
    """Seed the cache for window 1 of a Continue Video job from a saved entry.

    Loaded as though it were window 0 of this job, so the ordinary carry path
    and all of its checks decide whether it is used.
    """
    if not (CONFIG.enable and CONFIG.restore) or STORE is None:
        return
    plan = STATE.plan or {}
    if not plan.get("continuation_frames"):
        return                                   # nothing is being continued
    job = JOB.get()
    if job is None:
        _log("saved latents not checked: this generation did not start from "
             "Wan2GP's generate_media, so the source file is unknown")
        return
    if _continue_plugin_continuing():
        _log("saved latents not used: H3 Latent Continue is supplying this "
             "job's history from its own checkpoint")
        return
    if job.get("fallback"):
        _log("saved latents not checked for this job: its source file is unknown "
             "because another plugin had replaced the job hooks when it started. "
             "They are restored now, so the next job restores normally")
        return
    global PIN
    if PIN:
        pinned, PIN = PIN, None                      # one continuation, then released
        try:
            found = STORE.load(pinned)
        except Exception as error:
            _log(f"the pinned saved latents could not be read, re-encoding: {error}")
            return
        if found is None:
            _log("the pinned saved latents are no longer in the store, re-encoding")
            return
        problem = _restore_problem(found[0], plan, job)
        if problem:
            _log(f"the pinned saved latents do not fit this job: {problem}")
            return
        _log(f"using the pinned saved latents from "
             f"{found[0].get('video_name') or pinned[:12]}")
        _seed_from_entry(found, pinned, note="pinned")
        return

    source = job.get("video_source")
    if not isinstance(source, (str, os.PathLike)) or not os.path.isfile(source):
        _log("saved latents not checked: this job has no local source video file")
        return
    name = os.path.basename(str(source))
    if str(job.get("keep_frames_video_source") or "").strip():
        _log(f"saved latents for {name} not used: 'Truncate Video beyond' is set, so "
             f"the continued frames are not the end of the file")
        return
    try:
        sha = STORE.sha256_file(source)
        found = STORE.load(sha)
    except Exception as error:
        _log(f"saved latents for {name} could not be read, re-encoding: {error}")
        return
    if found is None:
        # The file handed to the job may be a copy of the one that was written.
        match = _match_saved_by_content(plan, job, _incoming_join_fingerprint(values, plan))
        if match is None:
            _log(f"no saved latents for {name}; the first window re-encodes")
            return
        try:
            found = STORE.load(match)
        except Exception as error:
            _log(f"saved latents matching {name} could not be read, re-encoding: {error}")
            return
        saved_name = (found[0].get("video_name") or match[:12]) if found else match[:12]
        _log(f"{name} is a copy of {saved_name} (different bytes, same footage); "
             f"matched its saved latents by content")
        try:                                     # so the next continuation hits directly
            STORE.save(sha, found[1], {**found[0], "video_name": name,
                                       "video_path": os.path.abspath(str(source)),
                                       "copy_of": match})
        except Exception:
            pass
    manifest, arrays = found
    problem = _restore_problem(manifest, plan, job)
    if problem:
        _log(f"saved latents for {name} not used: {problem}")
        return

    _seed_from_entry((manifest, arrays), sha, note=name)


def _seed_from_entry(found, sha, note=""):
    """Load an entry into the cache as though it were window 0 of this job."""
    manifest, arrays = found
    video = torch.from_numpy(np.ascontiguousarray(arrays["video"], dtype=np.float32))
    signatures = torch.from_numpy(np.ascontiguousarray(arrays["signatures"], dtype=np.float32))
    signature = arrays.get("signature")
    audio = arrays.get("audio")
    audio_signature = arrays.get("audio_signature")
    with STATE.lock:
        # Window 0 of this job, as far as every later check is concerned.
        STATE.latents = video
        STATE.signatures = signatures
        STATE.signature = (torch.from_numpy(np.ascontiguousarray(signature, dtype=np.float32))
                           if signature is not None else None)
        STATE.source_generation = STATE.generation - 1
        STATE.source_window_no = 0
        STATE.usable = True
        STATE.total_latents = int(manifest["total_latents"])
        STATE.expected_latents = _int_or_none(manifest.get("expected_latents"))
        STATE.source_discard = _int_or_none(manifest.get("discard_last_frames")) or 0
        if audio is not None and not manifest.get("audio_ends_at_join"):
            # Saved before 2.0.1: with Wan2GP v18.19+ its audio may run ~125 ms
            # past the join.  The video is still used; the audio is re-encoded.
            audio = None
            _log("saved audio latents predate 2.0.1, re-encoding the audio")
        if audio is not None:
            STATE.audio_latents = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
            STATE.audio_signature = (
                torch.from_numpy(np.ascontiguousarray(audio_signature, dtype=np.float32))
                if audio_signature is not None else None)
            STATE.audio_generation = STATE.generation - 1
            STATE.audio_source_window_no = 0
        STATE.restored = manifest.get("video_name") or note or sha[:12]
        if isinstance(manifest.get("lock_reference"), dict):
            # Continue Video continues the same shot, so it keeps holding that
            # shot's opening look, not the look the source had drifted to.
            STATE.lock_reference = {"mean": manifest["lock_reference"]["mean"],
                                    "std": manifest["lock_reference"]["std"]}
    STORE.touch(sha)
    STATE.stats["restored"] += 1
    _log(f"restored saved latents for {STATE.restored}: {int(video.shape[2])} video "
         f"latents{', audio' if audio is not None else ''}. Window 1 carries them if "
         f"they pass the usual checks")


def _export_tail(tensor, depth):
    """The last `depth` frames of what Wan2GP is about to write, as [C, k, H, W].

    For video, wgp.py passes save_video a *list* of the per-window tensors it has
    accumulated (frames_already_processed), not one tensor, so the tail may span
    the end of that list.  A single tensor, with or without a batch dimension, is
    accepted too.  Returns (tail, total frames) or (None, reason).
    """
    parts = list(tensor) if isinstance(tensor, (list, tuple)) else [tensor]
    if not parts:
        return None, "no frames were passed to save_video"
    normalized, total = [], 0
    for part in parts:
        if not hasattr(part, "ndim"):
            return None, f"save_video was passed {type(part).__name__}, which this plugin cannot read"
        frames = part[0] if part.ndim == 5 else part
        if frames.ndim != 4:
            return None, f"unexpected video tensor shape {tuple(part.shape)}"
        normalized.append(frames)
        total += int(frames.shape[1])
    collected, count = [], 0
    for frames in reversed(normalized):
        collected.insert(0, frames)
        count += int(frames.shape[1])
        if count >= depth:
            break
    tail = collected[0] if len(collected) == 1 else torch.cat(collected, dim=1)
    return (tail, total) if count >= depth else (None, "the video is shorter than the fingerprinted tail")


def snapshot_for_export(tensor, fps):
    """Package the window just decoded for saving, if the exported video matches it.

    Called with the frames Wan2GP is about to write.  Returns (snapshot, None),
    or (None, reason) when the file would show something other than what the
    latents describe.
    """
    if _RIVAL:
        return None, f"{_RIVAL} is also installed"
    plan = STATE.plan or {}
    if not plan.get("usable"):
        return None, f"the last window was not cacheable ({plan.get('reason')})"
    with STATE.lock:
        latents, signatures, signature = STATE.latents, STATE.signatures, STATE.signature
        source, generation = STATE.source_generation, STATE.generation
        total, expected = STATE.total_latents, STATE.expected_latents
        audio, audio_signature = STATE.audio_latents, STATE.audio_signature
        audio_generation = STATE.audio_generation
        discard = int(STATE.source_discard or 0)
    if latents is None or signatures is None or source != generation:
        return None, "no latents were captured for the last window"
    if expected is not None and total != expected:
        return None, f"latent count {total} != expected {expected}"

    depth = int(signatures.shape[0])
    tail, frames_or_reason = _export_tail(tensor, depth)
    if tail is None:
        return None, frames_or_reason
    frame_count, height, width = int(frames_or_reason), int(tail.shape[2]), int(tail.shape[3])
    # Post-processing (interpolation, upscaling, a trimmed tail) makes the file
    # show something other than the decoded window.  The latents are still worth
    # keeping - they describe the render - so the entry is written anyway, with
    # the render's own geometry.  Continuing that file is then refused by the
    # ordinary checks, while the entry stays findable by content for the
    # unprocessed footage.
    differs = None
    rendered_fps = plan.get("fps")
    if rendered_fps and abs(float(fps) - float(rendered_fps)) > 1e-3:
        differs = (f"it is written at {float(fps):g} fps but was rendered at "
                   f"{float(rendered_fps):g} fps (temporal upsampling?)")
    rendered_height = int(latents.shape[-2]) * _VAE_SPATIAL_FACTOR
    rendered_width = int(latents.shape[-1]) * _VAE_SPATIAL_FACTOR
    if (-(-height // _VAE_SPATIAL_FACTOR), -(-width // _VAE_SPATIAL_FACTOR)) != tuple(latents.shape[-2:]):
        differs = (f"it is {width}x{height} but the latents describe "
                   f"{rendered_width}x{rendered_height} (spatial upsampling?)")
    # Discard Last Frames cut the file `discard` frames before the decoded end.
    shown = depth - discard
    if shown < 2:
        return None, f"Discard Last Frames {discard} leaves too little to fingerprint"
    exported = torch.stack([_fingerprint(tail[:, i]) for i in range(-shown, 0)])
    errors = (exported - signatures[:shown]).abs().mean(dim=(1, 2))
    worst = float(errors.max())
    if worst > CONFIG.match_tolerance:
        differs = (f"its last frames differ from the decoded window "
                   f"(difference {worst:.4f})")

    keep = max(_MAX_CACHED_LATENTS, _carried_count())
    arrays = {"video": latents[:, :, -keep:].detach().cpu().numpy(),
              "signatures": signatures.detach().cpu().numpy()}
    if signature is not None:
        arrays["signature"] = signature.detach().cpu().numpy()
    if audio is not None and audio_generation == generation:
        arrays["audio"] = audio.detach().cpu().numpy()
        if audio_signature is not None:
            arrays["audio_signature"] = audio_signature.detach().cpu().numpy()
    manifest = {"plugin_version": VERSION,
                # The render's geometry, which is what the latents describe; the
                # written file's may differ when it was post-processed.
                "fps": float(rendered_fps or fps),
                "width": rendered_width, "height": rendered_height,
                "exported_differs": differs,
                "frames": frame_count,
                "total_latents": int(total),
                "expected_latents": _int_or_none(expected),
                "window_no": plan.get("window_no"),
                "carried_from": STATE.restored,
                "audio_ends_at_join": True,
                "discard_last_frames": int(discard),
                "lock_reference": STATE.lock_reference}
    return {"arrays": arrays, "manifest": manifest}, None


# --------------------------------------------------------------------------
# 3. substitution
# --------------------------------------------------------------------------

def _take_cached(history_video):
    """Return latents to use as the history block, or None to fall back."""
    if not CONFIG.enable:
        return None, "disabled"
    if _RIVAL:
        return None, f"{_RIVAL} is also installed"
    if not CONFIG.video:
        return None, _VIDEO_OFF

    blocked = (STATE.plan or {}).get("carry_blocked")
    if blocked:
        # About this window's conditioning, not about the cache, so it is read
        # from the current plan rather than from the cached window's.
        return None, blocked

    with STATE.lock:
        latents = STATE.latents
        signature = STATE.signature
        signatures = STATE.signatures
        source = STATE.source_generation
        source_window_no = STATE.source_window_no
        usable = STATE.usable
        expected = STATE.expected_latents
        total = getattr(STATE, "total_latents", None)
        generation = STATE.generation
        discard = int(STATE.source_discard or 0)

    STATE.takeover = False
    STATE.join_latent = None
    if latents is None:
        return None, "no cached latents"
    if source != generation - 1:
        return None, f"cache is from window {source}, need {generation - 1}"
    continues, reason = _continues_cached_window(source_window_no)
    if not continues:
        return None, reason
    if not usable:
        return None, "previous window was not cacheable"
    if expected is not None and total is not None and total != expected:
        # Our reconstruction of the frame maths disagrees with what the pipeline
        # actually produced.  Refuse rather than guess.
        return None, f"latent count {total} != expected {expected}"

    height, width = history_video.shape[-2], history_video.shape[-1]
    expected_h = -(-height // _VAE_SPATIAL_FACTOR)
    expected_w = -(-width // _VAE_SPATIAL_FACTOR)
    if latents.shape[-2] != expected_h or latents.shape[-1] != expected_w:
        # Normal and expected during two-phase pass 1, which runs at a reduced
        # resolution the cached latents cannot serve.
        return None, (f"resolution mismatch (cached latent {latents.shape[-2]}x{latents.shape[-1]}, "
                      f"need {expected_h}x{expected_w} for {height}x{width})")

    wanted = CONFIG.latents or _DEFAULT_LATENTS
    if discard not in (0, _TAKEOVER_FRAMES):
        # The join then falls inside a latent: no slice of the cache ends one
        # frame before it.
        return None, (f"Discard Last Frames is {discard}; latents can be carried "
                      f"with 0 or {_TAKEOVER_FRAMES}")

    # Do these latents actually depict the frames the pipeline is about to
    # re-encode?  The window-continuity check above rules out a different job;
    # this rules out the same job's window having produced something other than
    # what is being handed back, which the frame maths alone cannot see.  It is a
    # backstop now rather than the only defence, so a similar-looking pair of
    # clips no longer decides anything on its own.
    aligned, detail = _check_alignment(signatures, signature,
                                       _fingerprint(history_video[:, -1]),
                                       _ALIGN_EXPECTED_OFFSET - discard)
    if not aligned:
        return None, detail

    if discard:
        # Takeover layout.  The previous window lost its last latent's 4 frames,
        # so the join is its last-but-one latent, the 1-frame one that opens a
        # decoder chunk.  The history is then whole chunks ending one frame
        # before it - exactly the shape and place of Wan2GP's own history - so
        # it needs no coordinate correction.
        carried = wanted - 2
        if carried <= 0 or carried + 2 > latents.shape[2]:
            return None, f"cached tail holds {latents.shape[2]} latents, need {carried + 2}"
        if total is not None and (total - 2 - carried) % _LATENT_PERIOD:
            return None, f"slice would start at latent {total - 2 - carried}, off phase"
        STATE.takeover = True
        STATE.join_latent = latents[:, :, -2:-1].clone()
        return latents[:, :, -(carried + 2):-2].clone(), None

    if wanted % _LATENT_PERIOD != _PHASE_RESIDUE:
        # A window holds 2 + 5k latents, so only n = 2 (mod 5) starts the slice on
        # a phase-0 latent.  Any other n mislabels which latents cover one frame
        # and which cover four.
        return None, f"H3LC_LATENTS={wanted} is not phase-aligned (need 7, 12, 17...)"
    if wanted > latents.shape[2]:
        return None, f"cached tail holds {latents.shape[2]} latents, need {wanted}"
    if total is not None:
        if wanted >= total:
            return None, "previous window is too short to carry from"
        if (total - wanted) % _LATENT_PERIOD != 0:
            return None, f"slice would start at latent {total - wanted}, off phase"

    return latents[:, :, -wanted:].clone(), None


def _patched_add_video_history(self, video, visual_latents, keyframes, *args, **kwargs):
    """Append the history block, from cache where possible.

    Contract: upstream _add_video_history communicates through the two lists it
    is handed and returns nothing.  The carry path below therefore returns None,
    and the fallback path watches the original to confirm that stays true - if a
    Wan2GP update ever gives it a return value, carrying would start dropping it
    silently, so say so rather than let it pass.
    """
    if _PASSTHROUGH.get() is not None:
        return _ORIGINALS["add_video_history"](self, video, visual_latents, keyframes,
                                               *args, **kwargs)
    STATE.active_latents = None
    STATE.anchor_latents = None
    STATE.takeover = False
    STATE.takeover_prefix = None
    # Cleared per window: a stale value would shift the layout for a window
    # whose audio fell back to the native encode.
    STATE.audio_extra = 0
    if CONFIG.colour:
        try:
            _colour_seed(video)
        except Exception as error:
            _log(f"colour: could not read the continuation reference: {error!r}")
    latents, reason = _take_cached(video)
    if latents is None:
        if reason is _VIDEO_OFF:
            # Asked for, not failed.  Kept out of the fall-back tally so that
            # counter keeps meaning "wanted to carry and could not", which is
            # what the panel readout and the A/B protocol both read it as.
            STATE.stats["video_off"] += 1
            _log("re-encoding history (audio-only carry selected)")
        else:
            STATE.stats["fell_back"] += 1
            _log(f"re-encoding history ({reason})")
        started = time.perf_counter()
        result = _ORIGINALS["add_video_history"](self, video, visual_latents, keyframes,
                                                 *args, **kwargs)
        # What a carry costs instead of: measured here so the panel can say what
        # carrying saved, rather than guess at it.
        STATE.encode_seconds += time.perf_counter() - started
        STATE.encode_samples += 1
        if result is not None and not STATE.history_returns_value:
            STATE.history_returns_value = True
            _log(f"WARNING: stock _add_video_history returned {type(result).__name__}, "
                 f"but the carry path returns None. Upstream's contract has changed - "
                 f"the substitution needs to reproduce that value. Set H3LC_ENABLE=0 "
                 f"until it does.")
        return result

    if STATE.history_returns_value:
        # Observed upstream returning something we cannot reproduce.  Fall back
        # rather than carry and drop it.
        STATE.stats["fell_back"] += 1
        _log("re-encoding history (stock _add_video_history returns a value this "
             "version cannot reproduce)")
        return _ORIGINALS["add_video_history"](self, video, visual_latents, keyframes)

    visual_latents.append(latents)
    keyframes.append({"anchor": "history", "latent_frame_count": latents.shape[2]})
    STATE.active_latents = int(latents.shape[2])
    if STATE.takeover:
        STATE.stats["takeover"] += 1
        STATE.takeover_prefix = latents
    if CONFIG.latent_anchor and STATE.takeover and STATE.join_latent is not None:
        # pipeline.generate pins the join frame right after the history block:
        # add_image_condition(prepare_stage_video(continuation[:, -1:]), 0).
        STATE.anchor_latents = STATE.join_latent
    STATE.stats["engaged"] += 1
    _log(f"carried {latents.shape[2]} latents from the previous window "
         + ("(takeover: the join is on a latent boundary, layout as Wan2GP's own)"
            if STATE.takeover else
            f"(coords {'corrected' if CONFIG.fix_coords else 'UNCORRECTED'})"))

    return None                      # matches stock; see the docstring above


def _patched_add_image_condition(self, image, frame_index, *args, **kwargs):
    """Pin the previous window's own join latent instead of re-encoding the frame.

    Only in the takeover layout, where the join is the previous window's
    1-frame latent: the same kind of latent H3 makes for a pinned frame, so it
    can stand in for the encode with no decode, rounding or re-encode between.
    The native encode still runs, for the comparison that guards against a
    wrong pairing; everything else _add_image_condition does is left to it.
    """
    pending = STATE.anchor_latents
    if pending is None or _PASSTHROUGH.get() is not None:
        return _ORIGINALS["add_image_condition"](self, image, frame_index, *args, **kwargs)
    STATE.anchor_latents = None
    anchor = kwargs.get("anchor", args[3] if len(args) > 3 else None)
    if int(frame_index) != 0 or anchor is not None or not CONFIG.latent_anchor:
        return _ORIGINALS["add_image_condition"](self, image, frame_index, *args, **kwargs)

    missing = object()
    shadowed = self.__dict__.get("_encode_video", missing)
    encode = self._encode_video

    def encode_join(video, *a, **k):
        native = encode(video, *a, **k)
        try:
            if tuple(native.shape) != tuple(pending.shape):
                STATE.stats["anchor_native"] += 1
                _log(f"join latent {tuple(pending.shape)} does not fit the frame's "
                     f"{tuple(native.shape)}; using Wan2GP's encode")
                return native
            difference = float((pending.to(native.dtype) - native).abs().mean())
            if difference > _ANCHOR_TOLERANCE:
                STATE.stats["anchor_native"] += 1
                _log(f"join latent differs from the frame's encode by {difference:.3f}; "
                     f"using Wan2GP's encode")
                return native
            STATE.stats["anchor_from_latents"] += 1
            _log(f"pinned the previous window's own latent for the join frame "
                 f"(differs from the re-encode by {difference:.3f})")
            return pending.to(dtype=native.dtype).clone()
        except Exception as error:                               # never break a render
            STATE.stats["anchor_native"] += 1
            _log(f"join latent failed, using Wan2GP's encode: {error!r}")
            return native

    self._encode_video = encode_join
    try:
        return _ORIGINALS["add_image_condition"](self, image, frame_index, *args, **kwargs)
    finally:
        if shadowed is missing:
            self.__dict__.pop("_encode_video", None)
        else:
            self._encode_video = shadowed


# --------------------------------------------------------------------------
# 4. coordinate correction
# --------------------------------------------------------------------------

def _rows_per_frame(latent_height, latent_width, patch_size, target_spatial_context):
    """Ask packing for the real row count rather than assuming height*width/patch.

    _frame_grid applies target_spatial_context, so a tiled pass does not have
    (latent_height // patch_h) * (latent_width // patch_w) rows per frame.
    """
    _frame_grid = _PACKING._frame_grid
    _, patch_h, patch_w = patch_size
    grid, _ = _frame_grid(latent_height, latent_width, patch_h, patch_w, target_spatial_context)
    return int(grid.shape[0])


def _reference_row_stop(sequence, text_len, reference_start):
    """Row after the last Ref2VA reference row.

    The references occupy one contiguous span between the keyframe audio rows
    and the target audio rows (packing.py:233-265 walk a single cursor from
    reference_start), so the two condition-row totals the builder reports are
    enough to find its end: keyframe rows + reference rows is exactly
    num_condition_video_rows + num_condition_audio_rows.

    Returns reference_start when there are no references, and None when the
    totals cannot be read - the caller must not guess.
    """
    try:
        condition_rows = (int(sequence.num_condition_video_rows)
                          + int(sequence.num_condition_audio_rows))
    except (AttributeError, TypeError, ValueError):
        return None
    stop = text_len + condition_rows
    return stop if stop >= reference_start else None


def _apply_origin_shift(sequence, text_len, keyframe_anchors, audio_condition_anchors,
                        rows_per_frame, video_time_scale):
    """Correct the carried block's placement, and make room for a longer audio tail.

    Two independent adjustments, both applied by moving rows rather than by
    moving target_origin (see the comments on each).

    Shared by both sequence builders.  The row layout is
    [text][condition video][condition audio][target audio][target video] in each;
    Ref2VA additionally packs its reference rows inside the two condition spans,
    contiguously, between the keyframe audio rows and the target audio rows.
    Those references are positioned from a cursor that runs below target_origin,
    and both adjustments leave them where they are - the block translation
    because it touches only the carried block's own rows, the audio extension
    because it skips their span explicitly.  The carried block's internal
    spacing is already correct, because the slice is phase-aligned.
    """
    _unpack_keyframe_anchor = _PACKING._unpack_keyframe_anchor

    carried = STATE.active_latents
    anchors = list(keyframe_anchors)
    if not anchors:
        return sequence
    name, count, _ = _unpack_keyframe_anchor(anchors[0])
    if name != "history" or int(count) != carried:
        _log("keyframe layout not as expected, leaving coordinates untouched")
        return sequence

    times = sequence.position_ids[:, 0]

    # POSITIVE, and applied to the carried video history block alone.
    #
    # Only one relative offset needs changing: the carried block's content sits
    # one frame later than the layout reserves for it, so the block must sit one
    # frame later with respect to the target.  That can be done by moving the
    # target earlier or by moving the block later, and the two are identical in
    # relative terms - but not in their side effects.
    #
    # Moving everything anchored to target_origin earlier, as this did before,
    # also moved the target relative to the text rows and relative to the audio
    # conditions.  The audio history was exempted to avoid that, which displaced
    # it by a frame - 41.7 ms at 24 fps - and pushed its last latent past
    # target_origin.  Including it instead pushed the block below text_len,
    # where the text rows live.
    #
    # Moving the block later has neither problem: every other relative distance
    # is untouched by construction, and nothing lands below text_len.
    # The takeover layout is Wan2GP's own: nothing to move.
    delta = (0 if STATE.takeover else _RESERVE_DEFICIT) * _FRAME_RESCALE * float(video_time_scale)
    block_stop = text_len + carried * rows_per_frame
    times[text_len:block_stop] += delta

    keyframe_frames = sum(_unpack_keyframe_anchor(entry)[1] for entry in anchors)
    keyframe_stop = text_len + keyframe_frames * rows_per_frame

    # Extended audio context.  The audio history is laid forward from
    # float(text_len), so a longer block overruns target_origin, and pulling it
    # backwards would land it on the text rows.  So everything else moves later
    # by the same amount and the audio history keeps its start: the audio history
    # then reaches `extra` latents further back than the video history, with
    # every other relative distance preserved and nothing below text_len.
    extra = int(STATE.audio_extra or 0)
    if extra:
        # "Everything else" excludes the Ref2VA reference rows.  They sit below
        # the history in time, so moving them later by `extra` would march them
        # straight into the gap that is being opened for the audio tail - the
        # references end where the audio history starts, so the last `extra` of
        # their span would land exactly on the first `extra` of the tail.
        # Holding them still puts the gap between the references and the video
        # history, which is the same arrangement FL2VA gets between the text
        # rows and the video history.
        keyframe_audio_rows = sum(
            (entry[1] if isinstance(entry, tuple) else 1) * _AUDIO_CHANNELS
            for entry in audio_condition_anchors)
        reference_start = keyframe_stop + keyframe_audio_rows
        reference_stop = _reference_row_stop(sequence, text_len, reference_start)

        if reference_stop is None:
            # Should be unreachable: _audio_extension refuses reference mode
            # when the row counts cannot be read.
            _log("ERROR: cannot locate the reference rows; extended audio "
                 "context may collide with them. Set H3LC_AUDIO_CONTEXT=0 "
                 "and please report this.")
            times[text_len:] += float(extra)
        elif reference_stop > reference_start:
            times[text_len:reference_start] += float(extra)
            times[reference_stop:] += float(extra)
        else:
            times[text_len:] += float(extra)

        cursor = keyframe_stop
        for entry in audio_condition_anchors:
            anchor, length = entry if isinstance(entry, tuple) else (entry, 1)
            stop = cursor + length * _AUDIO_CHANNELS
            if anchor == "history":
                times[cursor:stop] -= float(extra)
            cursor = stop
    return sequence


_SIGNATURES = {}


def _native_arguments(key, args, kwargs):
    """What the native builder sees, by name, whatever its parameter list is.

    The builders gain parameters between Wan2GP releases - the current ones take
    target_condition_audio_latents and target_condition_video_frames, which did
    not exist when this was written - so the replacements forward *args/**kwargs
    verbatim and read what they need from a binding against the native signature
    rather than restating that signature here.
    """
    signature = _SIGNATURES.get(key)
    if signature is None:
        signature = _SIGNATURES[key] = inspect.signature(_ORIGINALS[key])
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments


def _shift_from_native(key, sequence, args, kwargs):
    values = _native_arguments(key, args, kwargs)
    return _apply_origin_shift(
        sequence, int(values["text_token_tags"].shape[0]),
        values.get("keyframe_anchors") or (), values.get("audio_condition_anchors") or (),
        _rows_per_frame(values["latent_height"], values["latent_width"],
                        values["patch_size"], values.get("target_spatial_context")),
        values.get("video_time_scale", 1.0))


def _patched_build_packed_sequence(*args, **kwargs):
    sequence = _ORIGINALS["build_packed_sequence"](*args, **kwargs)
    # A refinement pass has no carried block; the last window's active_latents
    # would otherwise shift its layout.
    if _PASSTHROUGH.get() is not None or STATE.active_latents is None or not CONFIG.fix_coords:
        return sequence
    return _shift_from_native("build_packed_sequence", sequence, args, kwargs)


def _patched_build_ref2va_packed_sequence(*args, **kwargs):
    sequence = _ORIGINALS["build_ref2va_packed_sequence"](*args, **kwargs)
    # A refinement pass has no carried block; the last window's active_latents
    # would otherwise shift its layout.
    if _PASSTHROUGH.get() is not None or STATE.active_latents is None or not CONFIG.fix_coords:
        return sequence
    return _shift_from_native("build_ref2va_packed_sequence", sequence, args, kwargs)


# --------------------------------------------------------------------------
# install / uninstall
# --------------------------------------------------------------------------

_EXPECTED_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_EXPECTED_FRAME_RESCALE = 5.0 / 3.0

# Patch targets.  (key, module-or-class getter, attribute, required).  Optional
# targets are skipped when absent so an older or newer Wan2GP loses one
# capability instead of the whole plugin.
_TARGETS = (
    ("decode", "video_vae", "decode", True),
    ("generate", "pipeline", "generate", True),
    ("add_video_history", "pipeline", "_add_video_history", True),
    # transformer.py imports the packing helpers by name, so the binding that
    # matters lives in the transformer module namespace, not in packing.
    ("build_packed_sequence", "transformer", "build_packed_sequence", True),
    ("build_ref2va_packed_sequence", "transformer", "build_ref2va_packed_sequence", False),
    ("audio_decode", "audio_vae", "decode", False),
    ("encode_audio", "pipeline", "_encode_audio", False),
    ("add_image_condition", "pipeline", "_add_image_condition", False),
)

_REPLACEMENTS = {
    "decode": "_patched_decode",
    "generate": "_patched_generate",
    "add_video_history": "_patched_add_video_history",
    "build_packed_sequence": "_patched_build_packed_sequence",
    "build_ref2va_packed_sequence": "_patched_build_ref2va_packed_sequence",
    "audio_decode": "_patched_audio_decode",
    "encode_audio": "_patched_encode_audio",
    "add_image_condition": "_patched_add_image_condition",
}


def _import_packing():
    """The packing module, wherever this Wan2GP keeps it."""
    last = None
    for path in ("models.minimax_h3.components.packing", "models.minimax_h3.packing"):
        try:
            return __import__(path, fromlist=["packing"]), None
        except Exception as error:
            last = error
    return None, last


def _preflight():
    """Check every assumption before touching anything.

    The plugin rewrites seven bindings and reproduces a coordinate layout from
    Wan2GP's own constants.  If a future or older release moves a binding or
    changes the frame grid, patching regardless would not fail loudly - it would
    produce a plausible video with the history in the wrong place.  So each
    assumption is verified up front, and anything unverifiable makes the plugin
    stay inert and say why.

    Returns (owners, available, problems).
    """
    problems = []
    try:
        from models.minimax_h3 import transformer
        from models.minimax_h3.pipeline import MiniMaxH3Pipeline
        from models.minimax_h3.video_vae import MiniMaxH3VideoVAE
    except Exception as error:
        return None, None, [f"MiniMax H3 is not available ({error!r})"]

    owners = {"transformer": transformer, "pipeline": MiniMaxH3Pipeline,
              "video_vae": MiniMaxH3VideoVAE}
    try:
        from models.minimax_h3.audio_vae import MiniMaxH3AudioVAE
        owners["audio_vae"] = MiniMaxH3AudioVAE
    except Exception:
        pass

    # The frame grid the coordinate maths and the phase rule are built on.
    global _PACKING
    packing, error = _import_packing()
    _PACKING = packing
    if packing is None:
        problems.append(f"cannot import the H3 packing module ({error!r})")
    else:
        for helper in ("_unpack_keyframe_anchor", "_frame_grid", "_video_t_grid",
                       "_reference_t_span"):
            if not callable(getattr(packing, helper, None)):
                problems.append(f"packing.{helper} is missing")
        grid = tuple(getattr(packing, "_FRAME_PER_TOKEN", ()))
        if grid != _EXPECTED_FRAME_PER_TOKEN:
            problems.append(f"_FRAME_PER_TOKEN is {grid}, expected "
                            f"{_EXPECTED_FRAME_PER_TOKEN}; the phase rule and the "
                            f"coordinate correction would both be wrong")
        rescale = float(getattr(packing, "_FRAME_RESCALE", 0.0))
        if abs(rescale - _EXPECTED_FRAME_RESCALE) > 1e-9:
            problems.append(f"_FRAME_RESCALE is {rescale}, expected "
                            f"{_EXPECTED_FRAME_RESCALE}")
        channels = getattr(packing, "MINIMAX_H3_AUDIO_CHANNELS", None)
        if channels != 2:
            problems.append(f"MINIMAX_H3_AUDIO_CHANNELS is {channels}, expected 2")

    # The pipeline's overlap offset depends on this flag; absent on older builds,
    # where the offset is always 1.
    global _CONTROL_LATENT_CONTINUATION
    try:
        from models.minimax_h3.constants import H3_CONTROL_LATENT_CONTINUATION
    except Exception:
        _CONTROL_LATENT_CONTINUATION = False
    else:
        _CONTROL_LATENT_CONTINUATION = bool(H3_CONTROL_LATENT_CONTINUATION)

    # Verified by performing the same imports the runtime path performs, rather
    # than by probing for them somewhere.  v1.0.0 looked for
    # video_latent_frames on MiniMaxH3Pipeline when it is a module-level
    # function in pipeline.py, so the plugin declared every build unpatchable -
    # including the one it was written against.  Mirroring the real import
    # cannot drift from it that way.
    try:
        from models.minimax_h3.pipeline import _as_video, video_latent_frames
    except Exception as error:
        problems.append(f"pipeline helpers _as_video / video_latent_frames are "
                        f"not importable ({error!r}); the latent count cannot be "
                        f"checked against expectation")
    else:
        if not callable(video_latent_frames) or not callable(_as_video):
            problems.append("pipeline._as_video / pipeline.video_latent_frames "
                            "are not callable")

    available = {}
    for key, owner_name, attribute, required in _TARGETS:
        owner = owners.get(owner_name)
        if owner is not None and getattr(owner, attribute, None) is not None:
            available[key] = (owner, attribute)
        elif required:
            problems.append(f"{owner_name}.{attribute} is missing")
    return owners, available, problems


# --------------------------------------------------------------------------
# Interoperability: expose each native binding behind its replacement
# --------------------------------------------------------------------------
#
# The replacements reach Wan2GP's originals through _ORIGINALS["..."], a global
# lookup that is invisible to anything inspecting the patched attribute.  Other
# plugins that need the native body - H3 Latent Continue edits the native
# MiniMaxH3Pipeline.generate in place and follows __wrapped__ to find it -
# would otherwise see only _patched_generate and refuse to install.  Setting
# __wrapped__ is the standard functools.wraps contract: inspect.unwrap() and
# inspect.signature() follow it, and the replacement is still what runs.
# Only __wrapped__ is set; __name__/__qualname__ stay ours so log lines and
# tracebacks keep naming this plugin.

def _link_wrapped(key, original):
    replacement = globals()[_REPLACEMENTS[key]]
    replacement.__wrapped__ = original


def _unlink_wrapped(key):
    replacement = globals().get(_REPLACEMENTS.get(key, ""))
    if replacement is not None:
        replacement.__dict__.pop("__wrapped__", None)


def install():
    rival = _rival_module()
    if rival:
        reason = (f"{rival} is already installed and patches the same methods. "
                  f"Disable Sliding Window Latents (or the other copy of it) in the "
                  f"Plugins tab and restart Wan2GP.")
        INSTALL_STATE.update(installed=False, reason=reason)
        _log("NOT INSTALLED: " + reason)
        return False
    installed = _install()
    INSTALL_STATE.update(installed=bool(installed),
                         reason="" if installed else (STATE.last_message or "see the console"))
    return installed


def _install():
    if _ORIGINALS:
        return True

    owners, available, problems = _preflight()
    if problems:
        _log("staying inert, nothing patched:")
        for problem in problems:
            _log(f"  - {problem}")
        _log("  this build of Wan2GP is not one this plugin can patch safely")
        return False

    if "audio_decode" not in available or "encode_audio" not in available:
        CONFIG.audio = False
        _log("audio bindings absent, carrying video latents only")
    if "build_ref2va_packed_sequence" not in available:
        _log("Ref2VA builder absent, plain sliding windows only")
    if "add_image_condition" not in available:
        CONFIG.latent_anchor = False
        _log("image-condition binding absent, the join frame stays Wan2GP's")

    # Apply as a unit: any failure rolls the whole thing back rather than
    # leaving some bindings patched and others not.
    applied = []
    try:
        for key, (owner, attribute) in available.items():
            _ORIGINALS[key] = getattr(owner, attribute)
            setattr(owner, attribute, globals()[_REPLACEMENTS[key]])
            applied.append((owner, attribute, key))
            _link_wrapped(key, _ORIGINALS[key])
    except Exception as error:
        for owner, attribute, key in reversed(applied):
            _unlink_wrapped(key)
            setattr(owner, attribute, _ORIGINALS.pop(key))
        _ORIGINALS.clear()
        _log(f"install failed and was rolled back, nothing patched: {error!r}")
        return False

    _ORIGINAL_OWNERS.update({key: (owner, attribute)
                             for key, (owner, attribute) in available.items()})
    _log(f"installed v{VERSION} (enable={CONFIG.enable}, fix_coords={CONFIG.fix_coords}, "
         f"video={CONFIG.video}, "
         f"latents={CONFIG.latents or _DEFAULT_LATENTS}, audio={CONFIG.audio}, "
         f"audio_context={CONFIG.audio_context or 'native'}, "
         f"colour={CONFIG.colour})")
    return True


def uninstall():
    if not _ORIGINALS:
        return
    for key, original in list(_ORIGINALS.items()):
        _unlink_wrapped(key)
        owner, attribute = _ORIGINAL_OWNERS.get(key, (None, None))
        if owner is None:
            continue
        try:
            setattr(owner, attribute, original)
        except Exception as error:
            _log(f"could not restore {attribute}: {error!r}")
    _ORIGINALS.clear()
    _ORIGINAL_OWNERS.clear()
    STATE.invalidate("uninstalled")
    _log("removed")
