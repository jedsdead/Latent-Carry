#!/usr/bin/env python3
"""Other refiners pass straight through without touching the carry.

H3 Face Refiner (Plus) calls MiniMaxH3Pipeline.refine_video, which goes back
through the patched generate().  Runs the real patches.py against a stand-in H3
pipeline whose generate(), refine_video() and VAE follow the real call shapes.
Before 1.3.0 such a pass dropped the cache and overwrote it.

    python tests/test_other_refiners.py
"""

import importlib.util
import math
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


def check(label, condition, detail=""):
    print(f"  [{'pass' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


# --------------------------------------------------------------------------
# frame maths, verbatim from shared/utils/frame_scheduler.py and pipeline.py
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


def token_start(index):
    return (index // 5) * 17 + (0, 1, 5, 9, 13)[index % 5]


SIZE = 32                       # frames are SIZE x SIZE; latents SIZE / 16
CH = 24
RATE = 32000
LOG = []


def to_uint8(signed):
    """pipeline.py with uint8_rounding="truncate"."""
    half = np.clip(signed, -1, 1).astype(np.float16)
    return np.clip((half + np.float16(1)) * np.float16(127.5), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# a stand-in H3 pipeline
# --------------------------------------------------------------------------

class MiniMaxH3VideoVAE:
    _model_dtype = np.dtype(np.float32)

    def decode(self, latents, uint8_rounding=None):
        """Deterministic: each latent's frames come from that latent alone, by
        position within its span, so a phase-aligned clip decodes exactly as the
        same latents do inside a whole window."""
        lat = latents.a[0]
        count = lat.shape[1]
        frames = []
        for i in range(count):
            span = token_start(i + 1) - token_start(i)
            base = np.repeat(np.repeat(lat[:3, i], 16, axis=1), 16, axis=2)
            for s in range(span):
                frames.append(np.tanh(base * 0.5 + 0.1 * s))
        video = np.stack(frames, axis=1)[None].astype(np.float32)
        if uint8_rounding == "truncate":
            return torch.Tensor(to_uint8(video))
        return torch.Tensor(video)


class MiniMaxH3AudioVAE:
    def decode(self, latents):
        return torch.Tensor(np.full((1, 2, 4000), 0.25, np.float32))


class Calls:
    refine = []
    refine_mode = "faithful"         # or "regenerate": a different take, as in the bug report
    last_refined = None
    last_prompt = None
    history = 0
    encode_audio = 0
    raise_in_refine = False
    generate_refine = []
    refuse_context = False
    abort_refine = False


class Transformer:
    control_layers = None
    cache = None


class MiniMaxH3Pipeline:
    reference_mode = True
    audio_only = False

    def __init__(self):
        self.vae = MiniMaxH3VideoVAE()
        self.audio_vae = MiniMaxH3AudioVAE()
        self.transformer = Transformer()

    def generate(self, input_prompt, input_frames=None, input_ref_images=None, input_masks=None,
                 input_video=None, input_waveform=None, input_waveform_sample_rate=None,
                 prefix_frames_count=0, frame_num=73, height=SIZE, width=SIZE, shift=12.0,
                 sampling_steps=8, seed=0, callback=None, VAE_tile_size=None,
                 audio_prompt_type="", fps=24, sample_solver="euler", starting_sigma=None,
                 refinement_mode=False, set_progress_status=None, denoising_strength=1.0,
                 masking_strength=1.0, preserve_input_mask_values=False, guide_phases=1, **kwargs):
        transformer = sys.modules["models.minimax_h3.transformer"]
        if refinement_mode:
            mask = None if input_masks is None else np.asarray(input_masks.a).reshape(-1)
            Calls.generate_refine.append({"frames": frame_num, "mask": mask,
                                          "starting_sigma": starting_sigma,
                                          "preserve": preserve_input_mask_values,
                                          "step_cache": self.transformer.cache})
            if Calls.refuse_context and mask is not None and mask.min() != mask.max():
                raise ValueError("simulated: Grouped Rows masking is not compatible with Sol Attention")
            # What the real one does in this mode, for the hooks' purposes: a
            # fresh clip of frame_num frames, laid out, encoded audio, decoded.
            transformer.build_packed_sequence(torch.Tensor(np.zeros((4, 2))), 1, 2, 2, 8, (1, 2, 2), (), 1.0)
            if input_waveform is not None:
                self._encode_audio(torch.Tensor(np.zeros((2, 100), np.float32)))
            count = video_latent_frames(frame_num)
            refined = np.random.default_rng(1000 + seed).standard_normal(
                (1, CH, count, SIZE // 16, SIZE // 16)).astype(np.float32)
            if Calls.refine_mode == "faithful":
                # A refine that keeps the shot: invert the stand-in decode on the
                # clip it was given, then nudge it, standing in for restored detail.
                clip = np.clip(input_frames.a, -0.999, 0.999)
                for i in range(count):
                    first = clip[:3, token_start(i), ::16, ::16]
                    refined[0, :3, i] = 2.0 * np.arctanh(first) + 0.15
            Calls.last_refined = refined.copy()
            Calls.last_prompt = input_prompt
            decoded = self.vae.decode(torch.Tensor(refined), None)[0]
            self.audio_vae.decode(torch.Tensor(np.zeros((1, 32, 2, 8), np.float32)))
            return {"x": torch.Tensor(np.clip(decoded.a, -1, 1)), "audio": None}

        visual, keyframes = [], []
        history = 0
        if input_video is not None and prefix_frames_count:
            count = min(prefix_frames_count, input_video.shape[1])
            self._add_video_history(input_video[:, -count:-1], visual, keyframes)
            if input_waveform is not None:
                self._encode_audio(torch.Tensor(np.zeros((2, 18000), np.float32)))
            history = count - 1
        transformer.build_packed_sequence(torch.Tensor(np.zeros((4, 2))), 1, 2, 2, 8, (1, 2, 2), (), 1.0)
        target = frame_num - history
        latents = torch.Tensor(np.random.default_rng(seed).standard_normal(
            (1, CH, video_latent_frames(target), SIZE // 16, SIZE // 16)).astype(np.float32))
        decoded = self.vae.decode(latents, "truncate")[0, :, :target]
        self.audio_vae.decode(torch.Tensor(np.zeros((1, 32, 2, 120), np.float32)))
        if history:
            prefix = input_video[:, -count:-1]
            prefix = prefix if prefix.dtype == np.uint8 else torch.Tensor(to_uint8(prefix.a))
            decoded = torch.cat([prefix, decoded], dim=1)
        samples = round(decoded.shape[1] / fps * RATE)
        return {"x": decoded, "visual": visual,
                "audio": np.zeros((samples, 2), np.float32), "audio_sampling_rate": RATE}

    def refine_video(self, video, *, prompt, strengths, denoising_strength=0.45, sampling_steps=4,
                     shift=12.0, seed=0, fps=24.0, sample_solver="euler", VAE_tile_size=None,
                     audio_waveform=None, audio_sample_rate=0, reference_images=None,
                     image_refs_relative_size=100.0, callback=None, set_progress_status=None):
        Calls.refine.append({"frames": int(video.shape[1]), "dtype": video.dtype,
                             "range": (float(video.a.min()), float(video.a.max())),
                             "strengths": len(strengths), "denoise": denoising_strength,
                             "steps": sampling_steps, "refs": reference_images,
                             "audio": None if audio_waveform is None else audio_waveform.shape,
                             "seed": seed, "step_cache": self.transformer.cache})
        if Calls.raise_in_refine:
            raise RuntimeError("simulated out of memory")
        if Calls.abort_refine:
            return None
        result = self.generate(input_prompt=prompt, input_frames=video, input_ref_images=reference_images,
                               input_masks=torch.Tensor(np.ones((1, video.shape[1], 1, 1))),
                               input_waveform=audio_waveform, frame_num=video.shape[1],
                               height=video.shape[-2], width=video.shape[-1], seed=seed, fps=fps,
                               refinement_mode=True)
        return None if result is None else result["x"]

    def _add_video_history(self, video, visual_latents, keyframes):
        Calls.history += 1
        visual_latents.append(torch.Tensor(np.zeros((1, CH, 5, SIZE // 16, SIZE // 16), np.float32)))
        keyframes.append({"anchor": "history", "latent_frame_count": 5})

    def _encode_audio(self, waveform):
        Calls.encode_audio += 1
        return torch.Tensor(np.zeros((1, 32, 2, 30), np.float32))


def install_stubs():
    for name in ("models", "models.minimax_h3", "models.minimax_h3.components", "shared", "shared.utils"):
        sys.modules.setdefault(name, types.ModuleType(name))
    pipeline = types.ModuleType("models.minimax_h3.pipeline")
    pipeline.MiniMaxH3Pipeline = MiniMaxH3Pipeline
    pipeline._as_video = lambda video: video
    pipeline.video_latent_frames = video_latent_frames
    pipeline.AUDIO_LATENT_FPS = 40
    sys.modules[pipeline.__name__] = pipeline
    transformer = types.ModuleType("models.minimax_h3.transformer")
    transformer.build_packed_sequence = lambda *a, **k: torch.Tensor(np.zeros((8, 4), np.float32))
    transformer.build_ref2va_packed_sequence = lambda *a, **k: torch.Tensor(np.zeros((8, 4), np.float32))
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
    constants = types.ModuleType("models.minimax_h3.constants")
    constants.H3_CONTROL_LATENT_CONTINUATION = False
    sys.modules[constants.__name__] = constants
    scheduler = types.ModuleType("shared.utils.frame_scheduler")
    scheduler.floor_frame_count = floor_frame_count
    scheduler.normalize_frame_count = normalize_frame_count
    scheduler.normalize_overlap = normalize_overlap
    sys.modules[scheduler.__name__] = scheduler


print(__doc__.strip().splitlines()[0])
print()
install_stubs()
spec = importlib.util.spec_from_file_location("h3lc_patches_refine", ROOT / "patches.py")
patches = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = patches
spec.loader.exec_module(patches)
patches._log = lambda message: LOG.append(message)
assert patches.install() is True

# Count coordinate corrections rather than perform them: the stand-in builders
# have no real layout.  The real correction is covered by test_layout_contract.
SHIFTS = []
patches._shift_from_native = lambda key, sequence, *a, **k: (SHIFTS.append(key), sequence)[1]


def raw_decode(latents, rounding=None):
    """The stand-in VAE without the plugin's hook, for expected values."""
    return patches._ORIGINALS["decode"](MiniMaxH3VideoVAE(), latents, rounding)


def fresh(**config):
    patches.STATE.invalidate("test")
    patches.STATE.stats.update({k: 0 for k in patches.STATE.stats})
    patches.STATE.generation = 0
    patches.STATE.anchor_stats = None
    patches.STATE.restored = None
    Calls.refine.clear()
    Calls.generate_refine.clear()
    Calls.raise_in_refine = Calls.abort_refine = Calls.refuse_context = False
    Calls.refine_mode = "faithful"
    defaults = dict(latents=7, colour=False, enable=True)
    defaults.update(config)
    for key, value in defaults.items():
        setattr(patches.CONFIG, key, value)
    LOG.clear()


def window(pipe, window_no, frame_num=73, previous=None, seed=0, refs=("ref",), prompt="p"):
    kwargs = dict(input_prompt=prompt, frame_num=frame_num, fps=24, width=SIZE, height=SIZE,
                  seed=seed, shift=12.0, window_no=window_no, input_ref_images=list(refs))
    if previous is not None:
        kwargs.update(input_video=previous[:, -18:], prefix_frames_count=18,
                      input_waveform=np.zeros((18000, 2), np.float32))
    return pipe.generate(**kwargs)


def raw_window(frame_num=73, seed=0):
    """The same window without the plugin's refine, for comparison."""
    saved = patches.CONFIG.refine
    patches.CONFIG.refine = False
    try:
        return MiniMaxH3Pipeline().generate(input_prompt="p", frame_num=frame_num, fps=24,
                                            seed=seed, window_no=1)
    finally:
        patches.CONFIG.refine = saved


# --------------------------------------------------------------------------
fresh()
pipe = MiniMaxH3Pipeline()
window(pipe, 1)
check("a window fills the cache", patches.STATE.latents is not None)
print("another refiner calling refine_video")
before = (patches.STATE.latents.a.copy(), patches.STATE.generation, patches.STATE.plan,
          patches.STATE.audio_latents.a.copy())
LOG.clear()
pipe.refine_video(torch.Tensor(np.zeros((3, 39, SIZE, SIZE), np.float32)), prompt="face",
                  strengths=[1.0] * 39)
after = (patches.STATE.latents.a, patches.STATE.generation, patches.STATE.plan,
         patches.STATE.audio_latents.a)
check("the cache, window count, plan and audio are untouched",
      np.array_equal(before[0], after[0]) and before[1] == after[1] and before[2] is after[2]
      and np.array_equal(before[3], after[3]))
check("nothing was dropped", not any("cache dropped" in m for m in LOG), " | ".join(LOG))
check("and it is counted as passed through", patches.STATE.stats["passthrough"] == 1)
print()

patches.uninstall()
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
