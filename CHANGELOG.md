# Changelog

## 2.1.0

- **Join frame from latents** (experimental, off by default). H3 pins the
  join frame at the start of each window. Wan2GP takes it from the finished
  video: rounded to 8-bit between windows, and read back from the compressed
  file for Continue Video. With this on, it is decoded from the carried
  latents instead, in full precision, with whatever decoder is loaded. If the
  result does not match the video's frame, the video's frame is used.

## 2.0.2

- **Off-grid window lengths now carry.** H3 rounds each window's frame count to
  its 5 + 17n grid before generating. The plugin now does the same before
  planning, so a window whose length is off the grid carries instead of quietly
  re-encoding (from Sliding Window Latents 1.2.5).

## 2.0.1

- **Fixed audio carry with Wan2GP v18.19+.** Wan2GP now generates about 125 ms
  of audio past each window's last frame. The plugin was carrying that tail into
  the next window as audio from before the join. It is now cut off first.
- Saved latents from earlier versions still continue the video; their audio is
  re-encoded as normal.

## 2.0.0

Simplified to what worked on real runs: latent carry, the colour hold, and
saving latents for Continue Video. The plugin slows the copy-of-a-copy loss over
roughly 4-5 windows; nothing tried here stopped it beyond that.

- **Removed refine feedback**, with motion context, the statistics anchor, tail
  hold and refine prompt options. On long runs it made the drift worse.
- **Removed history trust.** It sped the drift up and added steps at the joins.
- **Removed the colour hold's texture axis**, the join-match modes and the
  latents-only scope. The hold is now one mode: each continuous shot is held to
  its own opening, and cuts start a new reference.
- **Removed moment matching and the statistics diagnostics.**
- Other refiners (such as H3 Face Refiner) still pass straight through.
- The panel and README are rewritten: one line per setting.
- Old `settings.json` files still load; removed settings are ignored.

## 1.10.0

- **The refined tail is held to the shot's look too.** With the shot hold and
  the tail refine together, a real run showed the refine pass re-introducing
  H3's darkening: the refined tail came out 3-5 levels darker than the held
  window it replaced, and the next window, held again, opened with a 3-6 level
  brightening jump at every join. The refined clip's frames are now corrected to
  the shot's reference before blending, and its latents get the matching
  per-latent correction. The same run showed what the refine is for: face
  texture +1% and local contrast +4% by window 5, against +13% and +11% with the
  hold alone.
- **The statistics anchor is skipped while the shot hold is on.** It pulled the
  refined latents toward the carried latents, whose colour correction went
  through the approximate latent-to-RGB map (0.72 channel-mean shift on a real
  run, each window darker). The hold does its job from the frames.

## 1.9.0

- **Removed the spatial softening of carried latents.** On a real 7-window run
  of the shot hold, the softening controller climbed to its cap (0.20, 0.33,
  0.35 at windows 5-7) without stopping the texture rise, and the model read
  the smoothed latents as damaged conditioning: glossy, mottled, smeared faces
  by windows 6-7. Frames are still softened toward the shot's opening texture.
- The hold's console line now reports the texture at each window's start as
  well as its end.
- History trust (1.8.0) at 0.97 made the drift faster and added 3-5 level steps
  at every join on the same run; left in as an experiment, off by default.

## 1.8.0

- **History trust (experimental, off by default).** Goes at the cause of the
  copy-of-a-copy drift rather than its symptoms: H3 presents the previous
  window's frames as ground truth (noise level 0.001), so each window copies the
  last one's look exactly and adds its own bias. A new panel setting presents
  the history at a lower trust (0.99 to 0.90) by pre-mixing seeded noise, so the
  model keeps motion and layout from it but redraws texture, contrast and grade.
  Applies to carried and re-encoded history alike; references are untouched.
  The model is still told 0.999 (the simple version).
- Tests: `test_history_trust.py`.

## 1.7.0

Aimed at the degradation itself: where a shot continues across windows, each
window comes out darker, more contrasty, more saturated and more textured than
the frames it continues (measured on a real 5-window run: -9 levels, +7%
contrast, +14% saturation, +18% texture by window 5), and the carry hands it on.

- **Hold the continuous shot's look** (new colour match, the default): every
  frame of a continuing shot is held to the look of that shot's opening -
  brightness, cast, contrast, saturation and fine texture - over the whole
  frame, and the carried latents are corrected to match, so the drift cannot
  compound. Cuts, at a join or part way through a window, are detected from
  frame structure and start a new shot with its own reference; corrections are
  never smoothed across a cut. The shot's reference is saved with Continue
  Video entries.
- Texture: frames are blended toward a 3x3 blur where they carry more detail
  than the shot's opening; the carried latents are spatially softened by a
  controller on the next window's opening texture, reset at every new shot.
- **Removed**: "ease to window 1" and "window 1 throughout" (read as hold).
  Window 1 is not special: a continuous shot can start anywhere.
- The refine prompt no longer asks for sharp skin and fine detail, which fed the
  hardening; it asks for the same soft, natural texture, contrast and saturation.
- Tests: `test_shot_hold.py`.

## 1.6.0

Brings the refine closer to H3-LongTakeNoCuts, after reading its workflow file.

- **Skip window 1** (default on): the first window is no longer refined, as the
  workflow leaves shot 1 alone. A Continue Video run's window 1 continues saved
  latents and is still refined.
- **Refine prompt: the window's own** (option, whole-window scope only): refine
  with the window's prompt and image references, as the workflow does. Tail
  scope always keeps the neutral prompt. Works on both the plain and the
  motion-context refine paths; the log names which prompt was used.
- README: what the workflow actually does, and the closest settings here.
  Correction to earlier notes: its refine gets no motion context.

## 1.5.0

- **Join matching.** A real two-window Ref2VA run showed a step at the join:
  window 2's first frame 3.2 / 3.7 / 4.1 levels darker in R / G / B than window
  1's last, uniformly across the frame, then warming by about +3 R. Colour
  correction now defaults to correcting the new window's frames and its carried
  latents together (scope `both`), matched to the previous window's last frames,
  so the join is seamless and the next window starts from what is shown.
- **New match, the default: the join, easing to window 1.** The correction eases
  over the window from matching the join to matching window 1's look, pulling
  drift inside the window back without a step.
- **uint8 frames.** Current Wan2GP's H3 decode returns uint8; the `both` path
  assumed signed floats and would have crushed frames to black. Both forms are
  now handled, uint8 with a fixed spatial dither against whole-frame one-level
  steps.
- The carried latents get one correction per latent, from the frames each
  decodes to.
- After a refine, the next join's reference is re-measured from the frames as
  written.
- Panel: a *Correct* dropdown (join matching, or carried latents only), the new
  match option, and a warning when colour is on with no axes ticked. Saved
  settings from before 1.5.0 in which colour was never switched on take the new
  defaults instead of an old match mode and an empty axes list.
- Tests: `test_join_match.py`, on windows built to the measured join.

## 1.4.0

- **Motion context for the refine (off by default).** New panel setting and
  `H3LC_REFINE_CONTEXT` (0, 17 or 34 frames): original frames from just before
  the refined clip are held fixed while it is refined, so the refine continues
  their motion instead of re-staging the clip from its own noised frames - the
  H3-LongTakeNoCuts motion context. `refine_video` only takes a uniform strength,
  so this calls `generate()`'s masked video-to-video path with `refinement_mode`
  and `refine_video`'s own starting sigma. Whole-window scope takes the context
  from the kept frames at the window start. Falls back to the plain refine, with
  a console line, if this Wan2GP lacks the parameters or refuses the mask.
- The status line counts refines that used motion context; the log line names it.
- `tools/check_wan2gp.py` checks the masked path and the start-sigma formula.
- Tests: motion-context geometry, the masked call, placement, the fallback, and
  detection of a Wan2GP without the parameters.

## 1.3.2

- **The refine pass runs without step skipping.** Wan2GP sets the job's step
  skipping (First Block Cache or Spectrum) on the transformer for the whole job,
  and `refine_video` picked it up, so a 4-step refine could skip some of its
  steps, with a start step computed from the main sampler's step count. The
  cache is now set aside for the refine and restored after it, as Wan2GP's H3
  Face Refiner does. The main windows keep whatever step skipping is chosen.

## 1.3.1

Fixes the 1.3.0 refine, which on the first real run regenerated the end of the
window instead of refining it: a double exposure at the start of the ramp, then a
different take - another face, another framing, the last action performed again.

- **The refine no longer gets the window's prompt.** That prompt carries the
  window's timeline, and a 39-frame refine re-performed it in 1.6 seconds. The
  refine now gets a neutral restoration prompt modelled on Wan2GP's H3 Face
  Refiner (preserve everything in the clip, restore detail, add nothing);
  `H3LC_REFINE_PROMPT` overrides it.
- **No references are passed to the refine.** A Ref2VA reference video cannot
  reach `refine_video` at all, and unlabelled image references invite the model
  to insert them. Identity is held by the clip, as in the face refiner.
- **A refine that changes the picture is rejected.** Refined frames are compared
  with the window's own at the same moments; above the limit (0.03 by default,
  `H3LC_REFINE_MAX_CHANGE`, and a panel slider) the window is kept as generated.
  The measured change is logged for every refine.
- Default denoise lowered from 0.5 to 0.45, Wan2GP's face-refiner default.
- Tests: the stand-in refine now keeps the shot as a real one should, and a
  regenerating refine is checked to be rejected.

## 1.3.0

- **Refine feedback (experimental, off by default).** After each window, the tail
  that will be carried is resampled with `MiniMaxH3Pipeline.refine_video` at
  partial denoise and blended back on a frame ramp, so detail lost to drift is
  restored before it conditions the next window. Based on the H3-LongTakeNoCuts
  ComfyUI workflow by xyzDist. Denoise, steps, scope (carried tail or whole
  window), ramp and hold are configurable in a new panel section.
- **Statistics anchor** for the refined latents: off, this window's statistics,
  or the first window of the chain, against the saturation creep a refine alone
  does not stop. The chain anchor is stored in saved entries and restored with
  them.
- **Fixed: H3 Face Refiner (and Face Refiner Plus) broke the carry.** Their
  refinement pass goes through `MiniMaxH3Pipeline.generate`, which this plugin
  took for a new window: it dropped the cache ("caller did not report
  window_no") and overwrote it with the refiner's latents. Every refinement
  pass now passes straight through all seven hooks, and is counted in the panel.
- Saved entries keep only the latents a carry can use, even when the refine
  holds a whole window in the cache.
- Diagnostics report the refine settings and whether a chain anchor is set.
  `tools/check_wan2gp.py` checks `refine_video`'s parameters and that
  `generate()` still has a refinement mode.
- Added `tests/test_refine.py`. The numpy torch stand-in gained the few
  operations the refine path uses.

## 1.2.3

- `tools/check_wan2gp.py` now also checks the plugin side: every `_patched_*`
  must forward `*args`/`**kwargs`, so the 1.2.2 fix cannot regress unnoticed.
- Re-checked against Wan2GP `ec9566a` (6 Oct 2026): no problems.

## 1.2.2

- **Fixed: TypeError on every decode with current Wan2GP.**
  `MiniMaxH3VideoVAE.decode` gained a second positional argument
  (`uint8_rounding`), and the replacement restated the old signature:
  `_patched_decode() takes 2 positional arguments but 3 were given`. All six
  method replacements now forward `*args`/`**kwargs` to the native method
  untouched, so an added argument cannot break them again.
- The two packing builders did the same thing with their parameter lists; they
  gained `target_condition_audio_latents` and `target_condition_video_frames`
  upstream. Both now forward verbatim and read what they need from a binding
  against the native signature instead of restating it.
- Added `tests/test_signature_drift.py`, which calls every replacement the way
  current Wan2GP calls it, including the new arguments, and checks that each one
  reaches the native method. `tools/check_wan2gp.py` now prints the current
  signature of each patched method.

## 1.2.1

Checked against Wan2GP `ec9566a` (6 Oct 2026), newer than the build the carry was
written for. The seven bindings, the packing constants, the wgp.py functions and
the plugin API are all unchanged. Two differences in the H3 pipeline were found
and handled:

- **The pipeline's overlap offset is no longer always 1.** It uses 5 when
  `H3_CONTROL_LATENT_CONTINUATION` is enabled for a control model. That constant
  ships disabled, so nothing changes today, but the plugin hardcoded 1 and would
  have disagreed with the pipeline silently if it were turned on. The flag is now
  read in the preflight and mirrored in both the overlap and floor arithmetic.
- **Control models no longer use H3's history block.** When the transformer has
  control layers, the pipeline keeps the overlap as control latents and sets its
  continuation count to zero, so `_add_video_history` is never called. The plugin
  used to plan a carry for those windows that could never happen; it now declines
  them with that reason, and does not restore a saved entry into them either.
- Added `tools/check_wan2gp.py`, which checks all of the above against any Wan2GP
  checkout with `ast` alone, so the next update can be checked in one command.

## 1.2.0

- **Post-processed videos keep their latents.** Interpolation, upscaling or a
  trimmed tail used to skip the save entirely, leaving nothing to continue from.
  The entry is now written with the render's own fps and resolution, marked with
  what differed, and shown as *(post-processed)* in the panel. Continuing such a
  file is still refused by the ordinary checks, but the entry can be found by
  footage or pinned.
- **Pin an entry for the next continuation.** Pick a saved entry in the panel and
  the next continuing job uses it whatever file it is given, then releases the
  pin. For footage the lookup cannot recognise: a video trimmed or re-encoded
  outside Wan2GP. The fit checks and the join-frame check still apply.
- Both ideas come from the AcademiaSD ComfyUI H3 workflow, whose Moviola nodes
  continue from the raw sampler latent and keep takes in a numbered project
  folder, with post-processing downstream of the saved latent.

## 1.1.0

- **One entry per job by default.** The entry is written when the job ends,
  instead of hashing and rewriting a growing file after every sliding window.
  `H3LC_SAVE_SCOPE=every`, or "Save latents for: Every sliding window" in the
  panel, restores the old behaviour, which lets a mid-job file be continued.
- **Entries are about half the size.** Latents are stored as float16; they are
  VAE-space conditioning, well within that precision. Fingerprints stay float32
  because they are compared against a tolerance. A 1344x768 entry is now ~3 MB.
- **The panel reports what the carry saved**: re-encodes avoided, and an estimate
  of the time saved, timed from windows that actually fell back to the native
  encode rather than assumed.
- **`H3LC_TRACE` now defaults to off**, after the across-jobs path was diagnosed.
- Tests cover both save scopes and the float16 round trip.

## 1.0.6

- **Fixed: a saved video was not recognised when continued.** Wan2GP hands the job
  a copy of the file made by the interface (sanitized name, and bytes that need
  not match), so the hash lookup missed and the first window re-encoded.
- When the hash misses, entries are now matched by footage: the frame the
  continuation hands back is fingerprinted and compared against entries that
  already match the job's fps, resolution and model, with the join-frame check
  used between windows. A match is saved under the copy's hash too, so the next
  continuation hits the fast path.
- Unrelated footage is still refused; the fallback only accepts a decisive
  join-frame match.

## 1.0.5

- **Fixed: nothing was ever saved.** For video, wgp.py passes `save_video` a list
  of the per-window tensors it has accumulated, not a single tensor. The save
  hook required a tensor and skipped every call without a word, which is why
  1.0.0 to 1.0.4 saved nothing while reporting every hook as healthy. The tail is
  now read from the list, across several entries where the last window is short,
  and anything else is refused with a reason instead of silently.
- Tests now call the save hook the way wgp.py does, including a tail that spans
  two entries of the list.

## 1.0.4

- Each Wan2GP hook now counts its calls, and `H3LC_TRACE` (on by default) traces
  them to the console: the job starting, each window, the save hook and the
  record hook, including why a hook had nothing to do. Diagnostics also reports
  the generation count, the last window's plan and what the save hook last saw.
  This is for diagnosing a chain that stops somewhere; set `H3LC_TRACE=0` to
  quieten it.

## 1.0.3

- `save_video` and `record_file_metadata` are now wrapped where they are defined
  (`shared/utils/audio_video.py`, `shared/utils/media_recording.py`) as well as in
  wgp.py's globals, in case a build reaches them by another route.
- A window whose finished video is never seen by the save hook is now reported in
  the console at the next window, instead of nothing being saved silently.

## 1.0.2

- **Diagnostics button** in the panel: reports whether the H3 patches installed
  and why not, the state of each Wan2GP hook (ours, replaced, or missing),
  whether wgp.py's globals are reachable, the store path and entry count, the
  counters, and the last messages. Output goes to both the panel and the console.
- A refused install is now visible without the console: the panel header reads
  "NOT INSTALLED, see Diagnostics", and the startup line says which plugin is in
  the way. This is what happens when Sliding Window Latents is still enabled.
- Fixed: the 1.0.1 log function dropped the panel's "Last:" status line.

## 1.0.1

- **Fixed: nothing was saved, silently, when H3 Latent Continue was also
  installed.** Wan2GP gives each plugin references to its functions at load time,
  and `set_global` does not refresh the references other plugins hold. H3 Latent
  Continue sets up after this plugin (folder order) and wrapped the original
  `generate_media`, `save_video` and `record_file_metadata`, discarding this
  plugin's wrappers. The carry between windows was unaffected.
- Hooks now wrap `wgp.py`'s live globals, and are checked at the start of every
  render and on the panel's timer. If another plugin has replaced them, they are
  wrapped again on top of it, with a console line; both plugins' wrappers then run.
  The render that notices is still saved.
- Startup now prints whether saving and restoring are active, and where the store
  is, so a missing hook cannot go unnoticed.
- Test: `test_across_jobs.py` reproduces the other plugin replacing the hooks.

## 1.0.0 — H3 Latent Carry

First release under this name. Built on Sliding Window Latents 1.2.3, whose
history is below; the carry between windows is unchanged.

- **Latents are kept across jobs.** Every recorded H3 video is saved with its
  last window's latent tail, decoded-frame fingerprints and audio tail, in the
  plugin's own `latents/` folder under the SHA-256 of the finished file.
- **Continue Video uses them automatically.** Window 1 of a continuation hashes
  the Video to Continue file (and follows Continue last video); a matching entry
  seeds the cache as window 0 of the same job, and the ordinary carry path and
  its checks decide whether it is used. The first window of a continuation no
  longer has to re-encode frames that have been through the video codec.
- **Only matching files are saved.** The frames being written are fingerprinted
  against the decoded window, and fps and resolution compared, so upsampling,
  trimmed tails and similar post-processing skip the save with a reason.
- **Refusals are explicit**: a different fps, resolution or base model, the
  Truncate Video setting, H3 Latent Continue's own continuation, or continued
  frames that fail the alignment check each re-encode and say why.
- **Storage management**: a size limit (2 GB default, least recently used
  first), a panel readout of recent entries, and buttons to remove entries for
  deleted videos or clear everything. Entries are plain safetensors with a
  validated manifest; nothing is unpickled.
- **Refuses to run beside Sliding Window Latents**, which patches the same
  methods: install is refused if it is present, and the plugin stands aside at
  run time if it was installed afterwards.
- New modules `store.py` (entries) and `lifecycle.py` (wraps `generate_media`,
  `save_video`, `record_file_metadata` and `resolve_continue_video`, each with
  `__wrapped__`).
- Settings renamed from `SWL_*` to `H3LC_*`; new `H3LC_SAVE`, `H3LC_RESTORE`,
  `H3LC_STORE_GB` and `H3LC_STORE_DIR`. Console tag `[h3-latent-carry]`.
- Tests: `test_store.py`, and `test_across_jobs.py`, which runs the real carry,
  lifecycle and storage code through two jobs using a numpy stand-in for torch.

## Sliding Window Latents history

Entries below use the names of that plugin (`SWL_*` settings).

### 1.2.3
- Interoperability: each replacement now sets `__wrapped__` to the binding it
  displaced (the `functools.wraps` contract), and `uninstall()` and the install
  rollback remove it again. The originals were only reachable through the
  module-level `_ORIGINALS` dict, so plugins that locate the native H3 body by
  unwrapping the class attribute could not see past `_patched_generate`.
  H3 Latent Continue refused to install its latent hooks with "no reachable
  native method contains the required latent insertion points" whenever this
  plugin was active; it now finds the native `generate` without changes on its
  side. Only `__wrapped__` is set, so replacements keep their own names in logs
  and tracebacks. Behaviour of this plugin is unchanged.
- `inspect.signature()` on the patched attributes now reports Wan2GP's native
  parameters instead of `(self, *args, **kwargs)`.
- Added `tests/test_interop_wrapped.py`, which runs the real `install()`,
  `uninstall()` and a forced rollback against stubbed Wan2GP modules.
- `plugin_info.json` now points at the published repository; the previous `url`
  did not resolve.
- README: new H3 Latent Continue section, version line corrected (it still said
  1.1), and the test list and file tree now include every test.

### 1.2.2
- `SWL_AUDIO_CONTEXT` now works on Ref2VA. 1.2.1 refused it there because the
  extension moved the reference tokens later along with everything else, walking
  them back into the gap it had just opened for the longer audio tail. The
  reference rows are held still instead: they are one contiguous span between
  the keyframe audio rows and the target audio rows, ending at
  `text_len + num_condition_video_rows + num_condition_audio_rows`. That puts
  the gap between the references and the video history, matching what FL2VA has
  between the text rows and the video history.
- The extension still declines, with a log line, on a build that does not report
  those two row counts, since the span cannot be located without them.
- Added `tests/test_ref2va_audio_context.py`, which models the builder's time
  axis and checks occupancy across reference loads from none to eleven and
  extensions from 1 to 96 latents. It asserts the naive version collides, so the
  regression cannot pass silently.
- Corrected `_apply_origin_shift`'s docstring, which claimed the references were
  left alone — true of the block translation, not of the extension, which was
  the bug.

### 1.2.1
- Added `SWL_VIDEO`, with a matching panel control. `SWL_VIDEO=0` with
  `SWL_AUDIO=1` carries audio only, leaving Wan2GP's pixel re-encode in place
  for the video history. The two carries were always independent - video
  substitution happens in `_add_video_history`, audio in `_encode_audio` - but
  there was no way to ask for one without the other. Audio-only needs no
  coordinate correction at all, since audio conditions are laid on a uniform
  integer axis, so it is the plugin's least invasive configuration.
  `SWL_AUDIO_CONTEXT` is unavailable in this mode and says so: the compensating
  layout shift lives in the video carry.
- A deliberate audio-only run no longer counts as a fall back, so that counter
  keeps meaning "wanted to carry and could not".
- Added a guard for trimmed windows. `wgp.py` sets
  `image_end_frame_position = current_video_length - tail_trim_frames - 1`, so
  an end image pinned short of the last generated frame reveals a tail trim -
  the one place a trim is visible from inside `generate()`. Such a window
  generates more frames than it emits, so its cached tail would end past the
  video the next window hands back. `_check_alignment` would usually catch that
  as a negative skew, but a window pinned to an end image tends to settle onto a
  held composition, and on a flat tail the search abstains by design and leaves
  the decision to the tolerance, which a static shot passes.
- Added a guard for anything conditioned on target frame 0. Wan2GP pins the join
  there with a `"first"` anchor and the corrected carried block reaches it too;
  a third block asserting that instant makes the model hold on it and the join
  gains a frame that does not move. `pipeline.py:802` subtracts `history_count`
  from every injected position, so a raw position of exactly `history_count`
  lands on frame 0 - which is where Sliding Window Anchor puts its anchor
  (`_history_count` returns 17 at overlap 18 and `_inject` appends it raw). It
  blocks the carry for that window, not the cache: the window's own output is
  still sound. Keyed to the collision rather than to a plugin name.
- Wan2GP's own `L` frame injection does **not** reach frame 0 and needs no
  guard. On FL2VA and Ref2VA `extract_guide_from_window_start` is False, so a
  frame's relative position resolves to `abs_pos - window_start_frame` and the
  smallest value reachable is `reuse_frames`, giving a minimum `frame_index` of
  1. An `L` frame also sits one index below the next window's slice, so it is
  not handed forward at all. Latents carry normally alongside `L` frames;
  `tests/test_carry_modes.py` pins the arithmetic.
- The end-image position is now classified rather than subtracted blindly. A
  plugin can set `image_end_frame_position` to anything, and reading a position
  on the opening frame as a tail trim would have produced a large, confident and
  wrong number.
- An end image that is *not* short of the last frame changes nothing and is not
  refused: it arrives as a `"frame"` anchor positioned relative to
  `target_origin`, while the correction moves the carried block relative to the
  same origin, so the two cannot interact. Documented under *End images*.
- `SWL_AUDIO_CONTEXT` is now refused in Ref2VA (reference) mode. The extension
  opens a gap of `extra` below the history by moving everything else later; on
  FL2VA that gap is vacant, but `build_ref2va_packed_sequence` puts the
  reference tokens there and they move up with everything else, landing on the
  tail the gap was opened for. Checked against MiniMax H3 RefMods, which
  guarantees references exist, but it applies to native Ref2VA references too.
  The block translation is *not* gated: it is invariant to the reference cursor,
  since `history_time` and `target_origin` both derive from it.
- Added `tests/test_carry_modes.py`, which loads `patches.py` against stubbed
  Wan2GP modules and exercises both paths.

### 1.1
- Fixed: the coordinate correction displaced the carried audio history by one
  frame - 41.7 ms at 24 fps - and pushed its last latent past `target_origin`.
  It moved everything anchored to `target_origin` earlier and exempted the audio
  history, so the audio's relationship to the target changed even though the
  carried audio tail ends at the join exactly where a native encode's does. The
  correction now moves the carried video block one frame later instead, which is
  identical in relative terms and leaves every other distance - audio, text,
  keyframes - untouched.
- Added `SWL_AUDIO_CONTEXT`, extra seconds of audio history beyond Wan2GP's
  overlap-derived 0.75s window, with a matching panel control. Off by default.
  Gated on the video carry and `fix_coords`, since the compensating layout shift
  lives there.
- The install line now reports `audio_context`, and every window logs which
  carry path it took, so a setting that is not applying can be seen rather than
  guessed at.
- Added `tests/test_audio_context.py`, which checks the layout geometry
  numerically at 0.5, 1, 2 and 4 seconds.

### 1.0.2
- Fixed: the plugin stayed inert on every build. `_preflight` looked for
  `video_latent_frames` on `MiniMaxH3Pipeline`, but it is a module-level
  function in `pipeline.py`, so the check could never pass and 1.0.0 and 1.0.1
  declared every Wan2GP unpatchable - including the one they were written
  against. Preflight now verifies pipeline helpers by performing the same
  import the runtime path performs, so the two cannot drift apart.
- `tests/test_install_table.py` now checks that every name imported from
  `models.minimax_h3.*` anywhere in the plugin is also imported by
  `_preflight`, and that no module-level name is probed on a class. It fails
  on the 1.0.0 code.

### 1.0.1
- Colour consistency now checks that a window is uniform before correcting it.
  The correction is measured at the window's opening and applied to its closing
  latents, so a cut part way through put the measurement and its target on
  opposite sides of it and the carried block was corrected toward a grade its
  own scene never had. The window's head is now compared against its own tail
  and the correction is withheld beyond `SWL_COLOUR_SCENE`.

### 1.0.0
- First release.
- Install now preflights every patch target and the packing constants, and
  applies all seven bindings as a unit with rollback on failure. A Wan2GP
  that moves a binding or changes the frame grid makes the plugin stay
  inert with a reason rather than patch something it cannot reason about.
- Optional bindings degrade instead of aborting: no audio VAE means video
  latents only, no Ref2VA builder means plain sliding windows only.
- The packing module is resolved once at preflight and reused, so the
  coordinate correction no longer hard-codes one import path while
  preflight accepts two.
- Video and audio latent carry are both on by default.
- Colour consistency is labelled experimental in the panel.
- Documented that this must not be run alongside Sliding Window Anchor.

### 0.7.2
- Colour correction never ran. `_patched_decode` used
  `_measure_colour(decoded) or decoded`, and `or` calls `bool()` on a tensor,
  which raises; the surrounding `except` then caught it and logged a line that
  read like a deliberate fallback. Fixed, and the handler now words an
  unexpected failure as a defect.
- Added `tests/test_no_tensor_truthiness.py`, an AST lint forbidding tensor
  values in boolean contexts across the plugin.

### 0.7.1
- The join check tested one offset with a tolerance, which a slow shot passes
  even when the cached tail is displaced by several frames; the carried block
  was then placed as though it ended at the join and the overrun was rendered
  again. It now searches the decoded tail for the join and reports the skew.
- Added `tests/test_alignment.py`.

### 0.7.0
- Colour consistency, applied to the carried latents rather than to decoded
  pixels, so it is compatible with carrying. Off by default.
- `both` scope corrects the window as well, keeping the cached latents in
  step with it, so a visible join can be fixed rather than only contained.
- A Continue Video run takes its colour reference from the frames handed in,
  so the join between source and new video is no longer the one never
  checked.

### 0.6.3
- Coordinate correction shifted the target-anchored rows in the wrong direction,
  widening the skew from one frame to two rather than closing it. With
  `SWL_FIX_COORDS=1` the default was worse aligned than `SWL_FIX_COORDS=0`.
- Added `tests/test_layout_contract.py`, which fails on the old sign.
- Added `tools/measure_joins.py`.
- Renamed to Sliding Window Latents; environment prefix `H3LC_` is now `SWL_`,
  and the plugin id is now `SlidingWindowLatents`. Update the enabled-plugins
  entry and the folder name to match.
