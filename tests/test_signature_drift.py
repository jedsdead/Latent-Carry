#!/usr/bin/env python3
"""Every replacement is called the way current Wan2GP calls the method it replaces.

WHY THIS EXISTS
---------------
Wan2GP added a second positional argument to MiniMaxH3VideoVAE.decode
(uint8_rounding) after this plugin was written.  The replacement mirrored the
old signature, so every decode raised

    TypeError: _patched_decode() takes 2 positional arguments but 3 were given

and no amount of checking that the *binding* still existed would have caught it.
The packing builders had gained two keyword arguments the same way.

WHAT IS BEING CHECKED
---------------------
Each replacement is called exactly as the current pipeline calls it - extra
positional and keyword arguments included - and must forward everything to the
native method untouched, while still doing its own work.  The natives here
record what they received, so a replacement that quietly drops an argument fails
too.

    python tests/test_signature_drift.py
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
SEEN = {}


def check(label, condition, detail=""):
    print(f"  [{'pass' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


# --------------------------------------------------------------------------
# Wan2GP stand-ins with the CURRENT signatures (Wan2GP ec9566a, 6 Oct 2026)
# --------------------------------------------------------------------------

class MiniMaxH3VideoVAE:
    def decode(self, latents, uint8_rounding):           # uint8_rounding: added upstream
        SEEN["decode"] = {"uint8_rounding": uint8_rounding}
        return torch.Tensor(np.zeros((1, 3, 24, 64, 64), np.float32))


class MiniMaxH3AudioVAE:
    def decode(self, latents):
        SEEN["audio_decode"] = {}
        return torch.Tensor(np.zeros((1, 2, 32000), np.float32))


class MiniMaxH3Pipeline:
    def generate(self, input_prompt, **kwargs):
        SEEN["generate"] = dict(kwargs)
        return "generated"

    def _add_video_history(self, video, visual_latents, keyframes):
        SEEN["add_video_history"] = {"frames": int(video.shape[1])}
        visual_latents.append(torch.Tensor(np.zeros((1, 24, 5, 4, 4), np.float32)))
        keyframes.append({"anchor": "history", "latent_frame_count": 5})

    def _encode_audio(self, waveform):
        SEEN["encode_audio"] = {}
        return torch.Tensor(np.zeros((1, 32, 2, 30), np.float32))


def build_packed_sequence(text_token_tags, num_latent_frames, latent_height, latent_width,
                          num_audio_latents, patch_size, keyframe_anchors, video_time_scale,
                          audio_condition_anchors, target_condition_audio_latents,
                          target_condition_video_frames, target_spatial_context):
    SEEN["build_packed_sequence"] = {
        "target_condition_audio_latents": target_condition_audio_latents,
        "target_condition_video_frames": target_condition_video_frames,
        "target_spatial_context": target_spatial_context,
        "keyframe_anchors": keyframe_anchors}
    return torch.Tensor(np.zeros((8, 4), np.float32))


def build_ref2va_packed_sequence(text_token_tags, references, num_latent_frames, latent_height,
                                 latent_width, num_audio_latents, patch_size, video_time_scale,
                                 keyframe_anchors, audio_condition_anchors,
                                 target_condition_audio_latents, target_condition_video_frames,
                                 target_spatial_context):
    SEEN["build_ref2va_packed_sequence"] = {
        "references": references,
        "target_condition_video_frames": target_condition_video_frames}
    return torch.Tensor(np.zeros((8, 4), np.float32))


def install_stubs():
    for name in ("models", "models.minimax_h3", "models.minimax_h3.components",
                 "shared", "shared.utils"):
        sys.modules.setdefault(name, types.ModuleType(name))
    pipeline = types.ModuleType("models.minimax_h3.pipeline")
    pipeline.MiniMaxH3Pipeline = MiniMaxH3Pipeline
    pipeline._as_video = lambda video: video
    pipeline.video_latent_frames = lambda n: 2 + ((n - 5) // 17) * 5
    sys.modules[pipeline.__name__] = pipeline
    transformer = types.ModuleType("models.minimax_h3.transformer")
    transformer.build_packed_sequence = build_packed_sequence
    transformer.build_ref2va_packed_sequence = build_ref2va_packed_sequence
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
    scheduler.floor_frame_count = scheduler.normalize_frame_count = lambda *a, **k: 0
    scheduler.normalize_overlap = lambda *a, **k: (0, None)
    sys.modules[scheduler.__name__] = scheduler
    return transformer


print(__doc__.strip().splitlines()[0])
print()

transformer = install_stubs()
spec = importlib.util.spec_from_file_location("h3lc_patches_drift", ROOT / "patches.py")
patches = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = patches
spec.loader.exec_module(patches)
patches._log = lambda message: None
assert patches.install() is True

print("the VAE decodes, called as pipeline.py calls them")
vae = MiniMaxH3VideoVAE()
latents = torch.Tensor(np.zeros((1, 24, 22, 4, 4), np.float32))
try:
    vae.decode(latents, True)                     # the call that used to raise TypeError
    check("video decode accepts the extra positional argument", True)
    check("and uint8_rounding reaches the native decode", SEEN["decode"]["uint8_rounding"] is True)
except TypeError as error:
    check("video decode accepts the extra positional argument", False, repr(error))
try:
    vae.decode(latents, uint8_rounding=False)
    check("video decode accepts it as a keyword too", SEEN["decode"]["uint8_rounding"] is False)
except TypeError as error:
    check("video decode accepts it as a keyword too", False, repr(error))
check("the latent tail was still captured", patches.STATE.latents is not None)

MiniMaxH3AudioVAE().decode(torch.Tensor(np.zeros((1, 32, 2, 200), np.float32)))
check("audio decode still works", "audio_decode" in SEEN and patches.STATE.audio_latents is not None)
print()

print("the packing builders, with the arguments the current transformer passes")
tags = torch.Tensor(np.zeros((12, 3), np.float32))
try:
    transformer.build_packed_sequence(tags, 22, 4, 4, 30, (1, 2, 2), (), 1.0,
                                      audio_condition_anchors=(),
                                      target_condition_audio_latents=7,
                                      target_condition_video_frames=3,
                                      target_spatial_context=None)
    seen = SEEN["build_packed_sequence"]
    check("build_packed_sequence forwards the new keywords",
          (seen["target_condition_audio_latents"], seen["target_condition_video_frames"]) == (7, 3))
except TypeError as error:
    check("build_packed_sequence forwards the new keywords", False, repr(error))
try:
    transformer.build_ref2va_packed_sequence(tags, ["ref"], 22, 4, 4, 30, (1, 2, 2), 1.0,
                                             keyframe_anchors=(), audio_condition_anchors=(),
                                             target_condition_audio_latents=1,
                                             target_condition_video_frames=2,
                                             target_spatial_context=None)
    check("build_ref2va_packed_sequence forwards references and the new keywords",
          SEEN["build_ref2va_packed_sequence"] == {"references": ["ref"],
                                                   "target_condition_video_frames": 2})
except TypeError as error:
    check("build_ref2va_packed_sequence forwards references and the new keywords", False, repr(error))
print()

print("the pipeline methods")
pipe = MiniMaxH3Pipeline()
visual, keyframes = [], []
pipe._add_video_history(torch.Tensor(np.zeros((3, 17, 64, 64), np.float32)), visual, keyframes)
check("_add_video_history still appends to the two lists", len(visual) == 1 and len(keyframes) == 1)
pipe._encode_audio(torch.Tensor(np.zeros((2, 24000), np.float32)))
check("_encode_audio still runs the native encode", "encode_audio" in SEEN)
pipe.generate(input_prompt="p", frame_num=73, fps=24, width=64, height=64, window_no=1)
check("generate forwards its keywords", SEEN["generate"]["window_no"] == 1)
print()

patches.uninstall()
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
