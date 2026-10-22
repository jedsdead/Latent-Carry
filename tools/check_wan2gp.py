#!/usr/bin/env python3
"""Check an H3 Latent Carry install against a Wan2GP checkout, without running either.

Reads Wan2GP's source with ast, so no torch, no GPU and no model download.  Each
check names the assumption, where it comes from in the plugin, and what the
Wan2GP source actually says.

    python check_wan2gp.py /path/to/Wan2GP
"""
import ast, pathlib, sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
PROBLEMS, NOTES = [], []

def parse(rel):
    path = ROOT / rel
    if not path.exists():
        return None
    return ast.parse(path.read_text(encoding="utf-8", errors="replace"))

def find(tree, name, cls=None):
    """Top-level or class-level function/class by dotted name: 'Class.method' or 'func'."""
    if tree is None:
        return None
    parts = name.split(".")
    nodes = tree.body
    for i, part in enumerate(parts):
        match = None
        for n in nodes:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part:
                match = n
                break
        if match is None:
            return None
        nodes = match.body
    return match

def params(fn):
    if fn is None:
        return set()
    a = fn.args
    return {p.arg for p in (a.posonlyargs + a.args + a.kwonlyargs)} | \
           {a.vararg.arg for a in [a] if a.vararg} | {a.kwarg.arg for a in [a] if a.kwarg}

def check(label, ok, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        PROBLEMS.append(label)

def note(label, detail):
    print(f"  [note] {label}  -- {detail}")
    NOTES.append(label)

print(f"Wan2GP at {ROOT}\n")

# ---------------------------------------------------------------- pipeline side
print("The seven patched bindings")
pipeline = parse("models/minimax_h3/pipeline.py")
transformer = parse("models/minimax_h3/transformer.py")
video_vae = parse("models/minimax_h3/video_vae.py")
audio_vae = parse("models/minimax_h3/audio_vae.py")
packing = parse("models/minimax_h3/components/packing.py") or parse("models/minimax_h3/packing.py")

check("pipeline.MiniMaxH3Pipeline.generate", find(pipeline, "MiniMaxH3Pipeline.generate") is not None)
check("pipeline.MiniMaxH3Pipeline._add_video_history", find(pipeline, "MiniMaxH3Pipeline._add_video_history") is not None)
check("pipeline.MiniMaxH3Pipeline._encode_audio (optional)", find(pipeline, "MiniMaxH3Pipeline._encode_audio") is not None)
check("video_vae.MiniMaxH3VideoVAE.decode", find(video_vae, "MiniMaxH3VideoVAE.decode") is not None)
check("audio_vae.MiniMaxH3AudioVAE.decode (optional)", find(audio_vae, "MiniMaxH3AudioVAE.decode") is not None)
src_transformer = (ROOT / "models/minimax_h3/transformer.py").read_text(errors="replace")
check("transformer imports build_packed_sequence by name", "build_packed_sequence" in src_transformer)
check("transformer imports build_ref2va_packed_sequence (optional)", "build_ref2va_packed_sequence" in src_transformer)
check("pipeline._as_video and pipeline.video_latent_frames are module-level",
      find(pipeline, "_as_video") is not None and find(pipeline, "video_latent_frames") is not None)
print()

print("Signatures of the patched methods (the plugin forwards *args/**kwargs)")
# A replacement that restates a signature breaks the moment Wan2GP adds an
# argument: MiniMaxH3VideoVAE.decode gained uint8_rounding exactly that way.
for rel, dotted in (("models/minimax_h3/video_vae.py", "MiniMaxH3VideoVAE.decode"),
                    ("models/minimax_h3/audio_vae.py", "MiniMaxH3AudioVAE.decode"),
                    ("models/minimax_h3/pipeline.py", "MiniMaxH3Pipeline._add_video_history"),
                    ("models/minimax_h3/pipeline.py", "MiniMaxH3Pipeline._encode_audio")):
    fn = find(parse(rel), dotted)
    if fn is None:
        continue
    note(dotted, "(" + ", ".join(p.arg for p in fn.args.args) + ")")
for name in ("build_packed_sequence", "build_ref2va_packed_sequence"):
    fn = find(packing, name)
    if fn is not None:
        note(name, "(" + ", ".join(p.arg for p in fn.args.args) + ")")
print()

print("Other refiners' passes are recognised and passed through")
check("generate() still has a refinement mode the plugin can recognise",
      "refinement_mode" in params(find(parse("models/minimax_h3/pipeline.py"), "MiniMaxH3Pipeline.generate")))
print()

print("generate() keywords the plugin reads")
gen = find(pipeline, "MiniMaxH3Pipeline.generate")
needed = {"input_prompt", "input_video", "input_waveform", "prefix_frames_count",
          "frame_num", "fps", "width", "height", "window_no"}
have = params(gen)
src_wgp_all = (ROOT / "wgp.py").read_text(errors="replace")
takes_kwargs = gen is not None and gen.args.kwarg is not None
for name in sorted(needed):
    named = name in have
    # generate() takes **kwargs, so a keyword wgp.py passes is still readable.
    passed = takes_kwargs and f"{name} = {name}" in src_wgp_all
    check(f"generate(..., {name}=...)", named or passed,
          "" if named else "arrives via **kwargs from wgp.py")
print()

print("Packing constants the coordinate maths depends on")
src_packing = ((ROOT / "models/minimax_h3/components/packing.py").exists() and
               (ROOT / "models/minimax_h3/components/packing.py").read_text(errors="replace")) or ""
for helper in ("_unpack_keyframe_anchor", "_frame_grid", "_video_t_grid", "_reference_t_span"):
    check(f"packing.{helper}", find(packing, helper) is not None)
for const, expected in (("_FRAME_PER_TOKEN", "(1, 4, 4, 4, 4)"), ("_FRAME_RESCALE", "5"),
                        ("MINIMAX_H3_AUDIO_CHANNELS", "2")):
    line = next((l.strip() for l in src_packing.splitlines() if l.startswith(const)), None)
    check(f"packing.{const}", line is not None and expected.strip("()").split(",")[0] in line,
          line or "not found")
print()

# ---------------------------------------------------------------- wgp.py side
print("Wan2GP functions the save/restore hooks wrap")
wgp = parse("wgp.py")
for name, needed_params in (
        ("generate_media", {"model_type", "image_prompt_type", "video_source", "keep_frames_video_source"}),
        ("record_file_metadata", {"video_path", "configs", "is_image", "audio_only"}),
        ("set_wgp_global", set())):
    fn = find(wgp, name)
    check(f"wgp.{name} exists", fn is not None)
    missing = needed_params - params(fn)
    if fn is not None and missing:
        check(f"wgp.{name} parameters", False, f"missing {sorted(missing)}")
    elif fn is not None and needed_params:
        check(f"wgp.{name} parameters", True)
check("wgp.get_base_model_type exists", find(wgp, "get_base_model_type") is not None)

audio_video = parse("shared/utils/audio_video.py")
save = find(audio_video, "save_video")
check("audio_video.save_video exists", save is not None)
check("save_video(tensor=..., fps=...)", {"tensor", "fps"} <= params(save))
check("media_recording.record_file_metadata exists",
      find(parse("shared/utils/media_recording.py"), "record_file_metadata") is not None)
images = parse("shared/prompt_enhancer/images.py")
check("prompt_enhancer.images.resolve_continue_video exists", find(images, "resolve_continue_video") is not None)
print()

print("How save_video is called (the plugin reads a list of per-window tensors)")
src_wgp = (ROOT / "wgp.py").read_text(errors="replace")
calls = [l.strip() for l in src_wgp.splitlines() if "save_video(" in l and "def " not in l]
listish = [l for l in calls if "frames_already_processed" in l or "output_video_frames" in l]
check("save_video is called with the accumulated frame list", bool(listish),
      listish[0][:90] if listish else f"{len(calls)} call sites, none naming the frame list")
assign = [l.strip() for l in src_wgp.splitlines()
          if l.strip().startswith("output_video_frames") and "=" in l]
listed = [l for l in assign if "frames_already_processed" in l]
check("output_video_frames is still the accumulated frame list", bool(listed),
      listed[0][:90] if listed else f"assignments: {assign[:2]}")
hdr = [l.strip() for l in src_wgp.splitlines() if "save_hdr_video(" in l and "def " not in l]
if hdr:
    note("HDR output is written by save_hdr_video, which is not hooked",
         "HDR videos get no saved latents, by design")
print()

print("Continuation assumptions the carry mirrors")
pipe_src = (ROOT / "models/minimax_h3/pipeline.py").read_text(errors="replace")
check("the pipeline still normalises the overlap with normalize_overlap(..., 17, offset)",
      "normalize_overlap(int(prefix_frames_count or 0), 17" in pipe_src)
check("history is still continuation[:, -count:-1]",
      "continuation[:, -continuation_count:-1]" in pipe_src)
check("the target is still normalize_frame_count(target_frames, 5, 17, 5)",
      "normalize_frame_count(target_frames, 5, 17, 5)" in pipe_src)
check("pipeline.MiniMaxH3Pipeline._add_image_condition (join frame from latents)",
      find(pipeline, "MiniMaxH3Pipeline._add_image_condition") is not None)
check("the join frame is still pinned right after the history block",
      "add_image_condition(prepare_stage_video(continuation[:, -1:]), 0)" in pipe_src)
vae_src = ((ROOT / "models/minimax_h3/components/video_autoencoder.py").read_text(errors="replace")
           if (ROOT / "models/minimax_h3/components/video_autoencoder.py").exists() else "")
check("the video decoder still works in 17-frame clips with 3 dropped latents (takeover)",
      "clip_length: int = 17" in vae_src and "token_drop: int = 3" in vae_src)
wgp_src = (ROOT / "wgp.py").read_text(errors="replace") if (ROOT / "wgp.py").exists() else ""
check("generate_media still takes sliding_window_discard_last_frames (takeover)",
      "    sliding_window_discard_last_frames,\n" in wgp_src)
check("Wan2GP still cuts discarded frames before taking the next overlap (takeover)",
      "sample = sample[: , :-discard_last_frames]" in wgp_src)
check("_add_video_history is still what receives the history frames",
      "self._add_video_history(prepare_stage_video(history_frames)" in pipe_src)
flag = [l.strip() for l in (ROOT / "models/minimax_h3/constants.py").read_text(errors="replace").splitlines()
        if l.startswith("H3_CONTROL_LATENT_CONTINUATION")]
if flag:
    note(f"control-latent continuation flag: {flag[0]}",
         "the plugin reads this; offset 5 applies only when it is True")
if "control_known_frames, continuation_count = continuation" in pipe_src:
    note("control models divert the continuation into control latents",
         "the plugin declines those windows instead of expecting a history block")
print()

print("Plugin API")
plugins = parse("shared/utils/plugins.py")
for name in ("request_global", "set_global", "insert_after", "post_ui_setup"):
    check(f"WAN2GPPlugin.{name}", find(plugins, f"WAN2GPPlugin.{name}") is not None)
src_plugins = (ROOT / "shared/utils/plugins.py").read_text(errors="replace")
check("_set_wgp_global_func is still how set_global reaches wgp.py",
      "_set_wgp_global_func" in src_plugins)
restricted = [l.strip() for l in src_plugins.splitlines() if "restricted" in l.lower()][:3]
for name in ("generate_media", "save_video", "record_file_metadata", "get_base_model_type"):
    blocked = any(name in l for l in restricted)
    if blocked:
        check(f"{name} is not on the restricted list", False, "; ".join(restricted)[:120])
print()

print("The plugin's own replacements absorb whatever those signatures become")
# The decode TypeError in 1.2.1 happened because a replacement restated a
# signature.  Each one must take **kwargs (and, for methods, *args) so an added
# argument lands there instead of raising.
plugin_src = pathlib.Path(__file__).resolve().parent.parent / "patches.py"
if not plugin_src.exists():
    note("patches.py not found next to this tool", "plugin-side check skipped")
else:
    tree = ast.parse(plugin_src.read_text(encoding="utf-8"))
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name.startswith("_patched_"):
            forwards = fn.args.kwarg is not None and fn.args.vararg is not None
            check(f"{fn.name} forwards *args/**kwargs", forwards,
                  "" if forwards else "restates a fixed signature; an added argument would raise")
print()

print(f"{len(PROBLEMS)} problem(s)" + (":" if PROBLEMS else ""))
for p in PROBLEMS:
    print(f"  - {p}")
sys.exit(1 if PROBLEMS else 0)
