#!/usr/bin/env python3
"""Saving latents with a video, and restoring them when that video is continued.

Runs the real patches.py, lifecycle.py and store.py.  Torch is replaced by the
numpy shim in _torch_shim.py; Wan2GP by stand-ins that call the patched methods
in the order wgp.py and the H3 pipeline do.  No GPU, no model.

WHAT IS BEING CHECKED
---------------------
1. Job A renders one window.  save_video sees the frames, record_file_metadata
   sees the finished file, and an entry named after the file's SHA-256 appears
   in the store with the window's latent tail, fingerprints and audio.
2. Job B continues that file.  Window 1 finds the entry by hashing the Video to
   Continue file, and the history block is the saved latents - no video
   re-encode - even though the continued frames went through uint8 as they do
   when Wan2GP decodes an mp4.  The audio carry engages too.
3. "Continue last video" is followed: the file resolved inside generate_media
   is the one looked up.
4. Every refusal falls back to the native re-encode with a reason: no entry,
   different fps, 'Truncate Video beyond' set, a different base model, and
   continued frames that do not match the saved fingerprints.
5. Nothing is saved when the written video is not what was decoded: trimmed
   tail, temporal upsampling (fps), spatial upsampling (resolution).
6. The storage limit evicts the least recently used entries.
7. With Sliding Window Latents also installed, this plugin refuses to install,
   and stands aside at run time if it was wrapped afterwards.

    python tests/test_across_jobs.py
"""

import importlib.util
import math
import os
import pathlib
import shutil
import sys
import tempfile
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


# --------------------------------------------------------------------------
# Wan2GP stand-ins (the frame maths are verbatim from frame_scheduler.py)
# --------------------------------------------------------------------------

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


def video_latent_frames(n):
    return 2 + ((n - 5) // 17) * 5


SIZE = 64                      # decoded frames are SIZE x SIZE, latents SIZE/16
CHANNELS = 24
AUDIO_RATE = 32000


def make_frames(frame_count, seed):
    """Distinct frames in [-1, 1], so the alignment search has something to find."""
    rng = np.random.default_rng(seed)
    return torch.Tensor(rng.uniform(-1, 1, (3, frame_count, SIZE, SIZE)).astype(np.float32))


def make_waveform(seconds, seed):
    rng = np.random.default_rng(seed)
    envelope = np.linspace(0.2, 1.0, int(seconds * AUDIO_RATE), dtype=np.float32)
    return torch.Tensor((rng.standard_normal((2, envelope.size)) * envelope).astype(np.float32))


class Calls:
    encoded_history = 0
    encoded_audio = 0


class MiniMaxH3VideoVAE:
    frames = None

    def decode(self, latents):
        return torch.Tensor(self.frames.a[None])


class MiniMaxH3AudioVAE:
    waveform = None

    def decode(self, latents):
        return self.waveform


class MiniMaxH3Pipeline:
    reference_mode = False

    def generate(self, input_prompt, input_video=None, input_waveform=None,
                 prefix_frames_count=0, frame_num=73, fps=24, width=SIZE, height=SIZE,
                 window_no=1, **kwargs):
        """The calls the real generate() makes that the carry path hooks, in order."""
        visual, keyframes = [], []
        if input_video is not None and prefix_frames_count:
            count = min(prefix_frames_count, input_video.shape[1])
            self._add_video_history(input_video[:, -count:-1], visual, keyframes)
            if input_waveform is not None:
                samples = round(count / fps * AUDIO_RATE)
                self._encode_audio(input_waveform[..., -samples:])
            history = count - 1
        else:
            history = 0
        latents = torch.Tensor(np.random.default_rng(frame_num).standard_normal(
            (1, CHANNELS, video_latent_frames(frame_num - history), SIZE // 16, SIZE // 16)
        ).astype(np.float32))
        MiniMaxH3VideoVAE().decode(latents)
        MiniMaxH3AudioVAE().decode(torch.zeros(1, 32, 2, 120))
        return {"visual": visual, "keyframes": keyframes}

    def _add_video_history(self, video, visual_latents, keyframes):
        Calls.encoded_history += 1
        visual_latents.append(torch.zeros(1, CHANNELS, 5, SIZE // 16, SIZE // 16))
        keyframes.append({"anchor": "history", "latent_frame_count": 5})

    def _encode_audio(self, waveform):
        Calls.encoded_audio += 1
        latents = max(1, round(waveform.shape[-1] / 800))
        return torch.zeros(1, 32, 2, latents)


def install_wan2gp_stubs():
    for name in ("models", "models.minimax_h3", "models.minimax_h3.components", "shared",
                 "shared.utils", "shared.prompt_enhancer"):
        sys.modules.setdefault(name, types.ModuleType(name))

    pipeline = types.ModuleType("models.minimax_h3.pipeline")
    pipeline.MiniMaxH3Pipeline = MiniMaxH3Pipeline
    pipeline._as_video = lambda video: video
    pipeline.video_latent_frames = video_latent_frames
    pipeline.AUDIO_LATENT_FPS = 40
    sys.modules[pipeline.__name__] = pipeline

    transformer = types.ModuleType("models.minimax_h3.transformer")
    transformer.build_packed_sequence = lambda *a, **k: None
    transformer.build_ref2va_packed_sequence = lambda *a, **k: None
    sys.modules[transformer.__name__] = transformer

    for module_name, cls in (("models.minimax_h3.video_vae", MiniMaxH3VideoVAE),
                             ("models.minimax_h3.audio_vae", MiniMaxH3AudioVAE)):
        module = types.ModuleType(module_name)
        setattr(module, cls.__name__, cls)
        sys.modules[module_name] = module

    packing = types.ModuleType("models.minimax_h3.components.packing")
    for helper in ("_unpack_keyframe_anchor", "_frame_grid", "_video_t_grid", "_reference_t_span"):
        setattr(packing, helper, lambda *a, **k: None)
    packing._FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    packing._FRAME_RESCALE = 5.0 / 3.0
    packing.MINIMAX_H3_AUDIO_CHANNELS = 2
    sys.modules[packing.__name__] = packing

    scheduler = types.ModuleType("shared.utils.frame_scheduler")
    scheduler.floor_frame_count = floor_frame_count
    scheduler.normalize_frame_count = normalize_frame_count
    scheduler.normalize_overlap = normalize_overlap
    sys.modules[scheduler.__name__] = scheduler

    images = types.ModuleType("shared.prompt_enhancer.images")
    images.resolve_continue_video = lambda video_source, image_prompt_type, file_list, output_dir: (
        file_list[-1] if "L" in image_prompt_type and file_list else video_source)
    sys.modules[images.__name__] = images
    sys.modules["shared.prompt_enhancer"].images = images


FAKE_WGP_SOURCE = """
import pathlib
import numpy as np

folder = None
processed = []
frames_written = None
SELF = None


def set_wgp_global(variable_name, new_value):       # wgp.py:234, verbatim in effect
    globals()[variable_name] = new_value


def save_video(tensor, save_file=None, fps=30, **kwargs):
    # wgp.py passes frames_already_processed: a LIST of the per-window tensors.
    global frames_written
    import torch
    entries = tensor if isinstance(tensor, (list, tuple)) else [tensor]
    if not all(hasattr(p, "ndim") for p in entries):
        return                                   # wgp.py would raise; the hook ran first
    parts = [p[0] if p.ndim == 5 else p for p in entries]
    frames = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    uint8 = np.clip((frames.a + 1) * 127.5, 0, 255).astype(np.uint8)
    pathlib.Path(save_file).write_bytes(uint8.tobytes() + str(fps).encode())
    frames_written = uint8


def record_file_metadata(video_path, configs, is_image, audio_only, gen, **kwargs):
    with open(video_path, "ab") as handle:              # metadata changes the bytes
        handle.write(b"metadata")
    processed.append(video_path)


def generate_media(task=None, image_prompt_type="", video_source=None,
                   keep_frames_video_source="", model_type="minimax_h3_fl2va",
                   state=None, mode=None, plan=None, **kwargs):
    from shared.prompt_enhancer import images
    if "L" in image_prompt_type:
        video_source = images.resolve_continue_video(video_source, image_prompt_type,
                                                      processed, folder)
    return plan(SELF, video_source)
"""


def make_fake_wgp(folder):
    """A module whose functions call each other by global name, like wgp.py."""
    module = types.ModuleType("fake_wgp")
    exec(FAKE_WGP_SOURCE, module.__dict__)
    module.folder = folder
    module.SELF = module
    return module


def load_package(name, folder):
    spec = importlib.util.spec_from_file_location(
        name, folder / "__init__.py", submodule_search_locations=[str(folder)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    patches = importlib.import_module(f"{name}.patches")
    patches._log = lambda message: LOG.append(message)
    patches.CONFIG.verbose = False
    return package, patches


LOG = []


def last(prefix):
    return next((m for m in reversed(LOG) if prefix in m), None)


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------

print(__doc__.strip().splitlines()[0])
print()

install_wan2gp_stubs()
work = pathlib.Path(tempfile.mkdtemp(prefix="h3lc-test-"))
_, patches = load_package("h3lc", ROOT)
lifecycle = importlib.import_module("h3lc.lifecycle")
store_module = importlib.import_module("h3lc.store")
patches.STORE = store_module.LatentStore(str(work / "latents"))
assert patches.install() is True

wgp = make_fake_wgp(str(work))
ORIGINALS = {name: getattr(wgp, name)
             for name in ("generate_media", "save_video", "record_file_metadata")}
plugin = types.SimpleNamespace(
    generate_media=wgp.generate_media, save_video=wgp.save_video,
    record_file_metadata=wgp.record_file_metadata, _set_wgp_global_func=wgp.set_wgp_global,
    get_base_model_type=lambda model: {"minimax_h3_fl2va_pruned": "minimax_h3_fl2va"}.get(model, model))
plugin.set_global = wgp.set_wgp_global
check("lifecycle hooks install without problems", lifecycle.install(plugin) == [])
check("the continue-last-video resolver is wrapped",
      hasattr(sys.modules["shared.prompt_enhancer.images"].resolve_continue_video, "__wrapped__"))

FRAMES_A = make_frames(73, seed=1)
WAVE_A = make_waveform(73 / 24, seed=2)


def render(wgp, video_path, frames, waveform, write_frames=None, write_fps=24, **generate):
    """One single-window job: generate, then write the file the way wgp.py does."""
    MiniMaxH3VideoVAE.frames = frames
    MiniMaxH3AudioVAE.waveform = waveform
    generate.setdefault("fps", 24)                   # wgp.py always passes fps
    result = MiniMaxH3Pipeline().generate(input_prompt="prompt", **generate)
    written = frames if write_frames is None else write_frames
    # As wgp.py does: a list of the per-window tensors accumulated so far.
    wgp.save_video(tensor=written if isinstance(written, list) else [written],
                   save_file=video_path, fps=write_fps)
    wgp.record_file_metadata(video_path, {"model_type": "minimax_h3_fl2va"}, False, False, {})
    return result


def continue_input(path_frames, count=18):
    """What Wan2GP hands window 1 of a continue job: the file's last frames via uint8."""
    return torch.Tensor(path_frames[:, -count:].astype(np.float32) / 127.5 - 1.0)


# --------------------------------------------------------------------------
# 1. job A saves
# --------------------------------------------------------------------------

print("job A: render and save")
video_a = str(work / "a.mp4")
wgp.generate_media(plan=lambda w, src: render(w, video_a, FRAMES_A, WAVE_A, frame_num=73, window_no=1))
sha_a = store_module.sha256_file(video_a)
check("an entry named after the finished file's SHA-256 exists", patches.STORE.has(sha_a))
manifest, arrays = patches.STORE.load(sha_a)
check("the entry holds the 17-latent tail", arrays["video"].shape == (1, CHANNELS, 17, 4, 4),
      str(arrays["video"].shape))
check("the entry holds fingerprints and audio",
      {"signatures", "signature", "audio", "audio_signature"} <= set(arrays))
check("the manifest records geometry and model",
      (manifest["fps"], manifest["width"], manifest["height"], manifest["total_latents"],
       manifest["base_model_type"]) == (24.0, SIZE, SIZE, 22, "minimax_h3_fl2va"), str(manifest))
check("hashing happened after metadata was written",
      manifest["video_sha256"] == store_module.sha256_file(video_a))
frames_a_file = wgp.frames_written
print()

# --------------------------------------------------------------------------
# 2. job B continues A
# --------------------------------------------------------------------------


def continue_job(source, image_prompt_type="V", keep="", model="minimax_h3_fl2va",
                 fps=24, frames_in=None, out_name="b.mp4"):
    Calls.encoded_history = 0
    Calls.encoded_audio = 0
    engaged = patches.STATE.stats["engaged"]
    audio_engaged = patches.STATE.stats["audio_engaged"]
    frames_in = continue_input(frames_a_file) if frames_in is None else frames_in
    waveform = WAVE_A[..., -round(18 / 24 * AUDIO_RATE):]
    out = {}

    def plan(w, src):
        out["result"] = render(w, str(work / out_name), make_frames(90 - 17, seed=3),
                               make_waveform(73 / 24, seed=4),
                               input_video=frames_in, input_waveform=waveform,
                               prefix_frames_count=18, frame_num=90, fps=fps, window_no=1)
        out["source"] = src

    wgp.generate_media(image_prompt_type=image_prompt_type, video_source=source,
                       keep_frames_video_source=keep, model_type=model, plan=plan)
    out["carried"] = patches.STATE.stats["engaged"] - engaged
    out["audio_carried"] = patches.STATE.stats["audio_engaged"] - audio_engaged
    return out


print("job B: continue A's file")
LOG.clear()
out = continue_job(video_a)
visual = out["result"]["visual"]
check("window 1 carried the saved latents", out["carried"] == 1, last("carried") or "")
check("the history block is the saved tail's last 7 latents",
      len(visual) == 1 and np.array_equal(visual[0].a, arrays["video"][:, :, -7:]))
check("the video history was not re-encoded", Calls.encoded_history == 0)
check("the audio carry engaged from the saved audio", out["audio_carried"] == 1,
      last("audio") or "")
check("job B's own output was saved in turn",
      patches.STORE.has(store_module.sha256_file(str(work / "b.mp4"))))
check("job B's entry records what it was continued from",
      patches.STORE.manifest(store_module.sha256_file(str(work / "b.mp4"))).get("carried_from") == "a.mp4")
print()

print("the source is a renamed or sanitized copy of the saved file")
LOG.clear()
copy_path = work / "a (copy) [duration=6s].mp4"
copy_path.write_bytes(pathlib.Path(video_a).read_bytes() + b"different bytes")
out = continue_job(str(copy_path), out_name="copy.mp4")
check("a copy whose bytes differ is matched by its footage", out["carried"] == 1,
      last("copy of") or last("no saved latents") or "")
check("the match is reported as a content match", any("matched its saved latents by content" in m for m in LOG))
check("the copy gets its own entry, so next time the hash hits",
      patches.STORE.has(store_module.sha256_file(str(copy_path))))
LOG.clear()
unrelated = work / "unrelated.mp4"
unrelated.write_bytes(b"unrelated video")
out = continue_job(str(unrelated), frames_in=continue_input(
    np.clip((make_frames(18, seed=77).a + 1) * 127.5, 0, 255).astype(np.uint8)),
    out_name="unrelated-out.mp4")
check("unrelated footage is not matched to any entry",
      out["carried"] == 0 and any("no saved latents" in m for m in LOG),
      " | ".join(LOG)[:140])
print()

print("continue last video")
wgp.processed.append(video_a)                    # A is the most recent output again
out = continue_job(None, image_prompt_type="L")
check("the resolved file is looked up and carried", out["carried"] == 1 and out["source"] == video_a,
      last("restored") or last("saved latents") or "")
print()

# --------------------------------------------------------------------------
# 4. refusals fall back to the native encode
# --------------------------------------------------------------------------

print("refusals fall back to the re-encode")
other = work / "other.mp4"
other.write_bytes(b"not a video this plugin saved")
for label, kwargs, expect in (
        ("no entry, and footage from nowhere in the store",
         {"source": str(other), "frames_in": continue_input(np.clip(
             (make_frames(18, seed=91).a + 1) * 127.5, 0, 255).astype(np.uint8))},
         "no saved latents"),
        ("different fps", {"source": video_a, "fps": 30}, "fps"),
        ("'Truncate Video beyond' is set", {"source": video_a, "keep": "40"}, "Truncate"),
        ("a different base model", {"source": video_a, "model": "minimax_h3_ref2va"}, "saved with"),
        ("continued frames that do not match",
         {"source": video_a, "frames_in": continue_input(frames_a_file[:, :-3])}, "misaligned")):
    LOG.clear()
    out = continue_job(**{k if k != "source" else "source": v for k, v in kwargs.items()})
    reasons = " | ".join(LOG)
    check(f"{label}: re-encoded, with a reason",
          out["carried"] == 0 and Calls.encoded_history == 1 and expect in reasons,
          next((m for m in LOG if expect in m), reasons[:160]))

LOG.clear()
out = continue_job(video_a, model="minimax_h3_fl2va_pruned")
check("a variant of the same base model is accepted", out["carried"] == 1)
print()

# --------------------------------------------------------------------------
# 5. nothing saved when the file is not what was decoded
# --------------------------------------------------------------------------

print("a post-processed video still keeps its latents")
for label, kwargs, expect, geometry in (
        ("trimmed tail", {"write_frames": FRAMES_A[:, :-2]}, "differ", (SIZE, SIZE)),
        ("temporal upsampling", {"write_fps": 48}, "fps", (SIZE, SIZE)),
        ("spatial upsampling", {"write_frames": torch.Tensor(
            FRAMES_A.a.repeat(2, axis=2).repeat(2, axis=3))}, "upsampling",
         (SIZE, SIZE))):
    LOG.clear()
    path = str(work / f"post-{label.split()[0]}.mp4")
    wgp.generate_media(plan=lambda w, src: render(w, path, FRAMES_A, WAVE_A, frame_num=73,
                                                  window_no=1, **kwargs))
    saved = patches.STORE.manifest(store_module.sha256_file(path))
    check(f"{label}: saved anyway, and marked",
          saved is not None and expect in (saved.get("exported_differs") or ""),
          str(saved and saved.get("exported_differs"))[:90])
    check(f"{label}: the manifest records the render's geometry, not the file's",
          saved is not None and (saved["width"], saved["height"]) == geometry
          and saved["fps"] == 24.0, str(saved and (saved["width"], saved["fps"])))
print()

print("pinning an entry for the next continuation")
LOG.clear()
patches.PIN = sha_a
stranger = work / "edited-by-hand.mp4"
stranger.write_bytes(b"a video this plugin has never seen")
out = continue_job(str(stranger), frames_in=continue_input(frames_a_file), out_name="pinned.mp4")
check("the pinned entry is used whatever file the job is given", out["carried"] == 1,
      last("pinned") or " | ".join(LOG)[:140])
check("the pin is released after one job", patches.PIN is None)
LOG.clear()
out = continue_job(str(stranger), frames_in=continue_input(frames_a_file), out_name="pinned2.mp4")
check("afterwards the ordinary lookup applies again",
      any("copy of" in m or "no saved latents" in m for m in LOG), " | ".join(LOG)[:120])
LOG.clear()
patches.PIN = sha_a
out = continue_job(str(stranger), fps=30, out_name="pinned3.mp4")
check("a pinned entry that does not fit the job is still refused",
      out["carried"] == 0 and any("do not fit" in m for m in LOG), " | ".join(LOG)[:140])
print()

# --------------------------------------------------------------------------
# 5b. another plugin wraps the same functions after this one
# --------------------------------------------------------------------------

print("another plugin replaces the hooks after this one installed")
# What H3 Latent Continue does in its post_ui_setup: wrap the references it was
# handed at load time - the originals - and set_global them over ours.
other_calls = {"n": 0}


def stale_wrap(original):
    import functools

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        other_calls["n"] += 1
        return original(*args, **kwargs)
    return wrapper


for name, original in ORIGINALS.items():
    wgp.set_wgp_global(name, stale_wrap(original))
check("the hooks are gone after the other plugin loads",
      not any(lifecycle._is_ours(getattr(wgp, name)) for name in ORIGINALS))

LOG.clear()
video_c = str(work / "c.mp4")
wgp.generate_media(plan=lambda w, src: render(w, video_c, FRAMES_A, WAVE_A, frame_num=73, window_no=1))
check("the next render notices and wraps again, with a console line",
      any("replaced by another plugin" in m for m in LOG), last("replaced") or " | ".join(LOG)[:200])
check("that render's video is still saved",
      patches.STORE.has(store_module.sha256_file(video_c)), last("saved latents") or "")
check("all three hooks are ours again, on top of the other plugin's",
      all(lifecycle._is_ours(getattr(wgp, name)) for name in ORIGINALS))
before = other_calls["n"]
LOG.clear()
out = continue_job(video_c, out_name="d.mp4")
check("the other plugin's wrappers still run", other_calls["n"] - before == 3,
      f"{other_calls['n'] - before} calls")
check("and continuing that video restores its latents", out["carried"] == 1,
      last("restored") or "")
check("nothing is re-wrapped when nothing changed",
      lifecycle.ensure_installed() == [])
print()

print("one entry per job, or one per window")
LOG.clear()
before_count = patches.STORE.stats()[0]
multi_a, multi_b = str(work / "w1.mp4"), str(work / "w2.mp4")


def two_windows(w, src, name_a=None, name_b=None, seed=21):
    # A job that records a file after each of its two windows, as wgp.py does.
    # Distinct footage per job, so the two files do not share a hash.
    first, second = make_frames(56, seed), make_frames(73, seed + 1)
    render(w, name_a, first, WAVE_A, frame_num=56, window_no=1, write_frames=[first])
    render(w, name_b, second, WAVE_A, frame_num=73, window_no=2,
           input_video=first[:, -18:], input_waveform=WAVE_A[..., -24000:],
           prefix_frames_count=18, write_frames=[first, second])


patches.CONFIG.save_scope = "final"
wgp.generate_media(plan=lambda w, src: two_windows(w, src, multi_a, multi_b))
check("'final' writes one entry, for the job's last file",
      patches.STORE.stats()[0] == before_count + 1
      and patches.STORE.has(store_module.sha256_file(multi_b))
      and not patches.STORE.has(store_module.sha256_file(multi_a)),
      f"{patches.STORE.stats()[0] - before_count} new entries")

before_count = patches.STORE.stats()[0]
patches.CONFIG.save_scope = "every"
multi_c, multi_d = str(work / "w3.mp4"), str(work / "w4.mp4")
wgp.generate_media(plan=lambda w, src: two_windows(w, src, multi_c, multi_d, seed=31))
check("'every' writes an entry per recorded window",
      patches.STORE.stats()[0] == before_count + 2
      and patches.STORE.has(store_module.sha256_file(multi_c)),
      f"{patches.STORE.stats()[0] - before_count} new entries")
patches.CONFIG.save_scope = "final"
print()

print("a control model handles its own continuation")
LOG.clear()
Calls.encoded_history = 0


class ControlTransformer:
    control_layers = [object()]


pipeline_obj = MiniMaxH3Pipeline()
pipeline_obj.transformer = ControlTransformer()
MiniMaxH3VideoVAE.frames = FRAMES_A
MiniMaxH3AudioVAE.waveform = WAVE_A
pipeline_obj.generate(input_prompt="p", input_video=continue_input(frames_a_file),
                      prefix_frames_count=18, frame_num=90, fps=24, window_no=1)
check("the window is declined, naming control latents",
      any("control latents" in m for m in LOG), " | ".join(LOG)[:150])
check("and nothing is restored into it", patches.STATE.latents is None
      or patches.STATE.source_generation != patches.STATE.generation - 1)
print()

print("the frame list wgp.py passes")
LOG.clear()
video_multi = str(work / "multi.mp4")
# The last window's decoded frames are the last entry; earlier windows precede it.
wgp.generate_media(plan=lambda w, src: render(
    w, video_multi, FRAMES_A, WAVE_A, frame_num=73, window_no=1,
    write_frames=[make_frames(40, seed=9), FRAMES_A[:, :30], FRAMES_A[:, 30:]]))
check("a tail spanning several entries of the list is read and saved",
      patches.STORE.has(store_module.sha256_file(video_multi)), last("saved latents") or last("not saved") or "")
manifest_multi = patches.STORE.manifest(store_module.sha256_file(video_multi))
check("the frame count covers the whole list", manifest_multi["frames"] == 40 + 73,
      str(manifest_multi.get("frames")))
LOG.clear()
wgp.generate_media(plan=lambda w, src: render(
    w, str(work / "odd.mp4"), FRAMES_A, WAVE_A, frame_num=73, window_no=1,
    write_frames="not frames"))
check("something that is not frames is refused with a reason",
      any("cannot read" in m for m in LOG), " | ".join(LOG)[:120])
print()

print("a save hook that never runs is reported")
saved_hook = patches.ON_GENERATE
patches.ON_GENERATE = None                       # hooks gone and unable to heal
LOG.clear()
MiniMaxH3VideoVAE.frames = FRAMES_A
MiniMaxH3AudioVAE.waveform = WAVE_A
for _ in range(2):                               # the warning names the window before
    MiniMaxH3Pipeline().generate(input_prompt="p", frame_num=73, fps=24, window_no=1)
check("the plugin notices that nothing saw the finished video",
      any("never seen by the save hook" in m for m in LOG), " | ".join(LOG)[:150])
patches.ON_GENERATE = saved_hook
print()

# --------------------------------------------------------------------------
# 6. storage limit
# --------------------------------------------------------------------------

print("storage limit")
count, size = patches.STORE.stats()
patches.STORE.touch(sha_a)
entries = patches.STORE.entries()
keep_newest = entries[0][0]
removed = patches.STORE.enforce_limit(entries[0][1] + 1)
check("eviction removes all but the most recently used entry",
      removed == count - 1 and patches.STORE.has(keep_newest), f"removed {removed} of {count}")
print()

# --------------------------------------------------------------------------
# 7. Sliding Window Latents installed too
# --------------------------------------------------------------------------

print("with Sliding Window Latents also installed")
patches.uninstall()
swl = types.ModuleType("swl_rival.patches")
swl._REPLACEMENTS = {"generate": "_patched_generate"}
swl._ORIGINALS = {"generate": object()}
swl._patched_generate = lambda *a, **k: None
sys.modules["swl_rival.patches"] = swl
LOG.clear()
check("install refuses, and says which plugin is in the way",
      patches.install() is False and "swl_rival" in (last("NOT INSTALLED") or "")
      and patches.INSTALL_STATE["installed"] is False,
      last("NOT INSTALLED") or "")
del sys.modules["swl_rival.patches"]
check("installs once it is gone",
      patches.install() is True and patches.INSTALL_STATE["installed"] is True)
sys.modules["swl_rival.patches"] = swl
LOG.clear()
Calls.encoded_history = 0
MiniMaxH3VideoVAE.frames = make_frames(90 - 17, seed=3)
MiniMaxH3AudioVAE.waveform = make_waveform(3, seed=4)
MiniMaxH3Pipeline().generate(input_prompt="prompt", input_video=continue_input(frames_a_file),
                             prefix_frames_count=18, frame_num=90, window_no=2)
check("at run time it stands aside and the native path runs",
      Calls.encoded_history == 1 and any("standing aside" in m for m in LOG))
del sys.modules["swl_rival.patches"]
patches.uninstall()
print()

shutil.rmtree(work, ignore_errors=True)
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
