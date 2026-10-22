"""Wan2GP plugin entry point for H3 Latent Carry (MiniMax H3)."""

import json
import os

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

from . import lifecycle, patches
from .store import LatentStore


PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
INSERT_AFTER_TARGET = "video_info_accordion"   # sibling immediately before generate_btn
TAG = "[h3-latent-carry]"

PlugIn_Name = "H3 Latent Carry"
PlugIn_Id = "H3LatentCarry"

DESCRIPTION = (
    "Hands each MiniMax H3 sliding window the previous window's latents instead of "
    "re-encoding its frames, so long shots degrade more slowly."
)

NOTE = "Don't enable together with Sliding Window Latents or Sliding Window Anchor."

# Anything the panel exposes.  Env vars still set the initial value; the saved
# settings file wins over them once the panel has been used.
TOGGLES = ("enable", "video", "audio", "fix_coords", "colour", "save", "restore",
           "latent_anchor", "takeover")
STORE_LIMITS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


class H3LatentCarryPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = patches.VERSION
        self.description = DESCRIPTION

        self.state_dir = os.path.join(PLUGIN_DIR, "state")
        os.makedirs(self.state_dir, exist_ok=True)
        self._settings_path = os.path.join(self.state_dir, "settings.json")
        self._load_settings()

        store_dir = os.environ.get("H3LC_STORE_DIR") or os.path.join(PLUGIN_DIR, "latents")
        patches.STORE = LatentStore(store_dir)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def setup_ui(self):
        for name in ("generate_media", "save_video", "record_file_metadata",
                     "get_base_model_type"):
            self.request_global(name)
        patches.install()

    def post_ui_setup(self, components):
        # Model modules load lazily, so a failed install at setup_ui time is
        # normal and worth retrying here.
        patches.install()
        problems = lifecycle.install(self)
        for problem in problems:
            print(f"{TAG} saving/restoring across jobs: {problem}")
        if problems:
            print(f"{TAG} ^ saving and restoring across jobs are NOT active. Open the "
                  f"plugin panel and press Diagnostics for details.")
        else:
            print(f"{TAG} saving/restoring across jobs is active "
                  f"(save={patches.CONFIG.save}, restore={patches.CONFIG.restore}); "
                  f"store: {patches.STORE.root}")

        def build_panel():
            with gr.Accordion(self._panel_title(), open=False) as panel:
                gr.Markdown(DESCRIPTION)

                enable_cb = gr.Checkbox(
                    value=patches.CONFIG.enable, label="Enable latent carry",
                    info="Carry the previous window's latents into the next one.")
                video_cb = gr.Checkbox(
                    value=patches.CONFIG.video, label="Carry video latents",
                    info="Off: video is re-encoded as normal; audio can still be carried.")
                audio_cb = gr.Checkbox(
                    value=patches.CONFIG.audio, label="Carry audio latents",
                    info="Carry the soundtrack's latents too.")
                audio_ctx_dd = gr.Dropdown(
                    choices=[0.0, 0.5, 1.0, 2.0, 4.0],
                    value=float(patches.CONFIG.audio_context),
                    label="Audio context (seconds)",
                    info="How much audio the next window hears. 0 = Wan2GP's own (0.75 s).")
                latents_dd = gr.Dropdown(
                    choices=[7, 12, 17], value=patches.CONFIG.latents or 7,
                    label="Carried latents",
                    info="How much of the previous window is handed on: "
                         "7 = 22 frames, 12 = 39, 17 = 56.")
                takeover_cb = gr.Checkbox(
                    value=patches.CONFIG.takeover, label="Blend joins (needs Discard Last Frames = 4)",
                    info="With Wan2GP's Discard Last Frames at 4, each window is decoded together "
                         "with the end of the previous one, so the join blends like the middle "
                         "of a clip.")
                anchor_cb = gr.Checkbox(
                    value=patches.CONFIG.latent_anchor, label="Join frame from latents",
                    info="Experimental. With Discard Last Frames at 4, pins the previous "
                         "window's own latent for the join frame instead of re-encoding "
                         "the finished frame.")

                with gr.Accordion("Colour hold", open=True):
                    colour_cb = gr.Checkbox(
                        value=patches.CONFIG.colour, label="Hold each shot's colour",
                        info="Keeps brightness, contrast and colour steady while a shot "
                             "continues across windows. A cut starts a new reference.")
                    colour_axes_cb = gr.CheckboxGroup(
                        choices=list(patches._COLOUR_AXES),
                        value=list(patches.CONFIG.colour_axes),
                        label="Hold", info="What to keep steady.")
                    colour_strength_sl = gr.Slider(
                        0.0, 1.0, value=patches.CONFIG.colour_strength, step=0.05,
                        label="Strength",
                        info="1 removes all of the drift; lower removes part of it.")

                with gr.Accordion("Continue Video", open=False):
                    gr.Markdown("Each video is saved with its last window's latents, so "
                                "Continue Video on that exact file picks up where it left off.")
                    save_cb = gr.Checkbox(
                        value=patches.CONFIG.save, label="Save latents with each video",
                        info="About 2-20 MB per video.")
                    scope_dd = gr.Dropdown(
                        choices=[("Final video of each job", "final"),
                                 ("Every sliding window", "every")],
                        value=patches.CONFIG.save_scope, label="Save latents for",
                        info="Every window lets you continue a job that stopped early.")
                    restore_cb = gr.Checkbox(
                        value=patches.CONFIG.restore,
                        label="Use saved latents when continuing a video")
                    limit_dd = gr.Dropdown(
                        choices=STORE_LIMITS, value=self._limit_choice(),
                        label="Storage limit (GB)", info="Oldest entries are removed first.")
                    store_md = gr.Markdown(self._store_text())
                    with gr.Row():
                        pin_dd = gr.Dropdown(
                            choices=self._entry_choices(), value=None, label="Saved latents",
                            info="Force an entry for the next Continue Video, e.g. for a "
                                 "video you trimmed. Used once.")
                        pin_btn = gr.Button("Use for next continuation", size="sm")
                    with gr.Row():
                        diagnose_btn = gr.Button("Diagnostics", size="sm",
                                                 variant="primary")
                        refresh_btn = gr.Button("Refresh", size="sm")
                        prune_btn = gr.Button("Remove entries for deleted videos", size="sm")
                        clear_btn = gr.Button("Delete all saved latents", size="sm",
                                              variant="stop")

                with gr.Accordion("Advanced", open=False):
                    coords_cb = gr.Checkbox(
                        value=patches.CONFIG.fix_coords, label="Coordinate correction",
                        info="Keep on. Off is only for comparison.")

                status_md = gr.Markdown(self._status_text())
                gr.Markdown(NOTE)

                timer = gr.Timer(3)

                controls = [enable_cb, video_cb, audio_cb, audio_ctx_dd, latents_dd,
                            takeover_cb, anchor_cb, colour_cb, colour_axes_cb, colour_strength_sl,
                            save_cb, scope_dd, restore_cb, limit_dd, coords_cb]

                def _apply(enable, video, audio, audio_context, carried, takeover, latent_anchor,
                           colour, colour_axes, colour_strength,
                           save, scope, restore, limit, fix_coords):
                    patches.CONFIG.enable = bool(enable)
                    patches.CONFIG.video = bool(video)
                    patches.CONFIG.audio = bool(audio)
                    patches.CONFIG.audio_context = float(audio_context)
                    patches.CONFIG.latents = int(carried)
                    patches.CONFIG.takeover = bool(takeover)
                    patches.CONFIG.latent_anchor = bool(latent_anchor)
                    patches.CONFIG.colour = bool(colour)
                    patches.CONFIG.colour_axes = tuple(a for a in (colour_axes or ()) if a in patches._COLOUR_AXES)
                    patches.CONFIG.colour_strength = float(colour_strength)
                    patches.CONFIG.save = bool(save)
                    patches.CONFIG.save_scope = str(scope)
                    patches.CONFIG.restore = bool(restore)
                    patches.CONFIG.store_limit_gb = float(limit)
                    patches.CONFIG.fix_coords = bool(fix_coords)
                    self._save_settings()
                    return gr.update(label=self._panel_title()), self._status_text()

                for control in controls:
                    control.change(_apply, inputs=controls,
                                   outputs=[panel, status_md])

                def _poll():
                    try:
                        lifecycle.ensure_installed()
                    except Exception as error:
                        print(f"{TAG} could not check the Wan2GP job hooks: {error!r}")
                    return self._status_text(), gr.update(label=self._panel_title())

                timer.tick(_poll, inputs=[], outputs=[status_md, panel])

                def _refresh():
                    return self._store_text()

                def _prune():
                    removed = patches.STORE.prune_missing() if patches.STORE else 0
                    print(f"{TAG} removed {removed} saved latent file(s) whose video is gone")
                    return self._store_text()

                def _clear():
                    removed = patches.STORE.clear() if patches.STORE else 0
                    print(f"{TAG} deleted {removed} saved latent file(s)")
                    return self._store_text()

                def _diagnose():
                    report = self._diagnostics()
                    print(report.replace("**", ""))
                    return report

                diagnose_btn.click(_diagnose, inputs=[], outputs=[store_md])

                def _refresh_entries():
                    return gr.update(choices=self._entry_choices())

                def _pin(sha):
                    patches.PIN = sha or None
                    if patches.PIN:
                        print(f"{TAG} the next continuation will use the saved latents "
                              f"you picked, whatever file it is given")
                    return self._store_text()

                pin_btn.click(_pin, inputs=[pin_dd], outputs=[store_md])
                refresh_btn.click(_refresh_entries, inputs=[], outputs=[pin_dd])
                refresh_btn.click(_refresh, inputs=[], outputs=[store_md])
                prune_btn.click(_prune, inputs=[], outputs=[store_md])
                clear_btn.click(_clear, inputs=[], outputs=[store_md])

            return panel

        try:
            self.insert_after(
                target_component_id=INSERT_AFTER_TARGET,
                new_component_constructor=build_panel,
            )
        except Exception as error:
            # The panel is only controls and a readout; the patches are already
            # installed and keep working without it.
            print(f"{TAG} could not place the panel next to "
                  f"'{INSERT_AFTER_TARGET}' ({error}). The plugin still works; "
                  f"settings fall back to environment variables.")
        return {}

    # ------------------------------------------------------------------ #
    # panel display
    # ------------------------------------------------------------------ #

    def _panel_title(self):
        if not patches.INSTALL_STATE.get("installed"):
            return "H3 Latent Carry — NOT INSTALLED, see Diagnostics"
        if not patches.CONFIG.enable:
            return "H3 Latent Carry (off)"
        extras = []
        if not patches.CONFIG.video:
            extras.append("audio only" if patches.CONFIG.audio else "nothing carried")
        if not patches.CONFIG.fix_coords:
            extras.append("uncorrected coords")
        if not patches.CONFIG.save:
            extras.append("not saving")
        if patches.CONFIG.colour:
            extras.append("colour held")
        suffix = f" — {', '.join(extras)}" if extras else ""
        return f"H3 Latent Carry{suffix}"

    def _status_text(self):
        stats = patches.STATE.stats
        warning = (" - **nothing ticked under Hold**" if patches.CONFIG.colour
                   and not patches.CONFIG.colour_axes else "")
        return (f"Windows carried: **{stats.get('engaged', 0)}** "
                f"&nbsp;·&nbsp; re-encoded: **{stats.get('fell_back', 0)}** "
                f"&nbsp;·&nbsp; audio carried: **{stats.get('audio_engaged', 0)}**"
                f"{self._anchor_text()}  \n"
                f"Colour held: **{stats.get('colour_applied', 0)}** window(s){warning}  \n"
                f"{self._savings_text()}  \n"
                f"Saved: **{stats.get('saved', 0)}** &nbsp;·&nbsp; not saved: "
                f"**{stats.get('save_skipped', 0)}** &nbsp;·&nbsp; restored for "
                f"Continue Video: **{stats.get('restored', 0)}**  \n"
                f"Last: {patches.STATE.last_message}")

    def _anchor_text(self):
        stats = patches.STATE.stats
        text = ""
        if stats.get("takeover", 0) or stats.get("joint_decode", 0):
            text += (f"  \nJoins on a latent boundary: **{stats.get('takeover', 0)}** "
                     f"&nbsp;·&nbsp; blended: **{stats.get('joint_decode', 0)}**")
        if not patches.CONFIG.latent_anchor:
            return text
        return text + (f"  \nJoin frames from latents: **{stats.get('anchor_from_latents', 0)}** "
                f"&nbsp;·&nbsp; from the video: **{stats.get('anchor_native', 0)}**")

    def _entry_choices(self):
        """(label, sha) for the saved entries, newest first."""
        store = patches.STORE
        if store is None:
            return []
        choices = []
        for sha, size, _mtime in store.entries()[:40]:
            info = store.manifest(sha) or {}
            name = info.get("video_name") or sha[:12]
            note = " (post-processed)" if info.get("exported_differs") else ""
            choices.append((f"{name} — {info.get('width')}x{info.get('height')}"
                            f", {info.get('frames')} frames{note}", sha))
        return choices

    def _diagnostics(self):
        """Everything needed to tell which part is not working, for the console and panel."""
        state = patches.INSTALL_STATE
        lines = [f"**H3 Latent Carry {patches.VERSION} diagnostics**", ""]
        if state.get("installed"):
            lines.append("- H3 patches: **installed**")
        else:
            lines.append(f"- H3 patches: **NOT installed** — {state.get('reason')}")
        try:
            report = lifecycle.status()
        except Exception as error:
            report = None
            lines.append(f"- Job hooks: could not be read ({error!r})")
        if report is not None:
            if not report["namespace"]:
                lines.append("- Job hooks: **no access to wgp.py's globals**; saving and "
                             "restoring cannot work")
            for name, condition in report["hooks"].items():
                mark = "" if condition == "ours" else " **<-- problem**"
                lines.append(f"- `{name}`: {condition}{mark}")
            lines.append(f"- 'Continue last video' resolver: "
                         f"{'wrapped' if report['resolver'] else '**not wrapped**'}")
        lines += [f"- Saving: **{patches.CONFIG.save}** · restoring: "
                  f"**{patches.CONFIG.restore}** · carry: **{patches.CONFIG.enable}**",
                  f"- Store: `{patches.STORE.root if patches.STORE else 'unavailable'}` "
                  f"({patches.STORE.stats()[0] if patches.STORE else 0} entries)",
                  f"- Colour hold: **{patches.CONFIG.colour}** "
                  f"({', '.join(patches.CONFIG.colour_axes) or 'nothing'}, strength "
                  f"{patches.CONFIG.colour_strength:g}; shot reference "
                  f"{'set' if patches.STATE.lock_reference else 'not set'})",
                  f"- Counters: {patches.STATE.stats}",
                  f"- Generations: {patches.STATE.generation}, last window: "
                  f"{patches.STATE.window_no}, save hook last saw generation "
                  f"{patches.STATE.save_seen}",
                  f"- Last plan: {patches.STATE.plan}", "", "Recent messages:"]
        lines += [f"- {message}" for message in list(patches.LOG_HISTORY)[-12:]] or ["- (none)"]
        return "\n".join(lines)

    def _savings_text(self):
        """What carrying has actually saved, from measured native encodes."""
        state = patches.STATE
        avoided = state.stats.get("engaged", 0)
        if not avoided:
            return "Re-encodes avoided: **0**"
        if not state.encode_samples:
            return (f"Re-encodes avoided: **{avoided}** &nbsp;·&nbsp; time saved: not "
                    f"measurable yet (no window has fallen back to compare against)")
        average = state.encode_seconds / state.encode_samples
        return (f"Re-encodes avoided: **{avoided}** &nbsp;·&nbsp; time saved: about "
                f"**{avoided * average:.0f}s** (measured {average:.1f}s per encode over "
                f"{state.encode_samples})")

    def _limit_choice(self):
        value = float(patches.CONFIG.store_limit_gb)
        return value if value in STORE_LIMITS else min(STORE_LIMITS, key=lambda c: abs(c - value))

    def _store_text(self):
        store = patches.STORE
        if store is None:
            return "Storage unavailable."
        count, size = store.stats()
        lines = [f"**{count}** saved, **{size / 1024 ** 2:.0f} MB** of "
                 f"{patches.CONFIG.store_limit_gb:g} GB  \n`{store.root}`"]
        if patches.PIN:
            info = store.manifest(patches.PIN) or {}
            lines[0] += (f"  \nNext continuation will use: "
                         f"**{info.get('video_name') or patches.PIN[:12]}**")
        for sha, _, _ in store.entries()[:5]:
            info = store.manifest(sha) or {}
            name = info.get("video_name") or sha[:12]
            detail = f"{info.get('width')}x{info.get('height')}, {info.get('frames')} frames"
            lines.append(f"- {name} ({detail})")
        return lines[0] + ("\n\n" + "\n".join(lines[1:]) if len(lines) > 1 else "")

    # ------------------------------------------------------------------ #
    # settings
    # ------------------------------------------------------------------ #

    def _load_settings(self):
        try:
            if not os.path.isfile(self._settings_path):
                return
            with open(self._settings_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            for name in TOGGLES:
                if name in data:
                    setattr(patches.CONFIG, name, bool(data[name]))
            if "latents" in data:
                patches.CONFIG.latents = int(data["latents"])
            if "audio_context" in data:
                patches.CONFIG.audio_context = float(data["audio_context"])
            if "colour_axes" in data:
                axes = tuple(a for a in data["colour_axes"] if a in patches._COLOUR_AXES)
                # An empty list holds nothing; settings from before 2.0 could
                # hold "texture" alone, which no longer exists.
                patches.CONFIG.colour_axes = axes or patches._COLOUR_AXES
            if "colour_strength" in data:
                patches.CONFIG.colour_strength = float(data["colour_strength"])
            if "store_limit_gb" in data:
                patches.CONFIG.store_limit_gb = float(data["store_limit_gb"])
            if data.get("save_scope") in ("final", "every"):
                patches.CONFIG.save_scope = data["save_scope"]
        except Exception as error:
            print(f"{TAG} could not load settings ({error}); using defaults.")

    def _save_settings(self):
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            payload = {name: bool(getattr(patches.CONFIG, name)) for name in TOGGLES}
            payload["latents"] = int(patches.CONFIG.latents or 7)
            payload["audio_context"] = float(patches.CONFIG.audio_context)
            payload["colour_axes"] = list(patches.CONFIG.colour_axes)
            payload["colour_strength"] = float(patches.CONFIG.colour_strength)
            payload["store_limit_gb"] = float(patches.CONFIG.store_limit_gb)
            payload["save_scope"] = str(patches.CONFIG.save_scope)
            with open(self._settings_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as error:
            print(f"{TAG} could not save settings: {error}")
