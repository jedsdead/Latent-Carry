"""Wan2GP job lifecycle hooks for saving and restoring latents across jobs.

The carry itself (patches.py) lives inside the H3 pipeline and never sees file
names.  Saving and restoring need two facts that only wgp.py knows, so four of
its functions are wrapped.  Every wrapper delegates to the function it replaced
and sets __wrapped__, so other plugins wrapping the same functions still work.

  generate_media           opens a job context: model, Video to Continue path,
                           and the "Truncate Video beyond" setting.
  resolve_continue_video   in shared/prompt_enhancer/images.py.  "Continue last
                           video" picks its file here, inside generate_media, so
                           the context is updated with the file actually used.
  save_video               sees the exact frames written.  The window just
                           decoded is snapshotted here, and only if those frames
                           still match it (patches.snapshot_for_export).
  record_file_metadata     runs after the file, its audio and its metadata are
                           final.  The file is hashed here and the snapshot is
                           stored under that hash.

Nothing here raises into Wan2GP.  A failure to save or restore is logged and the
render carries on exactly as it would without the plugin.

Other plugins that wrap the same functions
------------------------------------------
Wan2GP hands each plugin a reference to its functions when plugins load, and
set_global only overwrites wgp.py's global; it does not refresh the references
other plugins already hold.  So a plugin that sets up after this one and wraps
the same function (H3 Latent Continue wraps all three wgp.py functions) wraps
the *original* and silently discards this plugin's wrapper.

Two things prevent that.  Wrappers are built around wgp.py's live global, read
from the namespace set_global writes to, rather than the stale reference.  And
ensure_installed() checks each global's __wrapped__ chain for this plugin's
wrapper and wraps again on top of whatever replaced it; it runs at the start of
every generate() call and on the panel's status timer.
"""

import functools
import inspect
import os
import sys

from . import patches

_INSTALLED = {}
_HOOKS = {}                   # wgp.py global name -> wrapper factory
_NAMESPACE = None             # wgp.py's module globals, when reachable
_MARK = "_h3_latent_carry_hook"

# wgp.py holds direct references to these, so replacing its global is what
# normally matters.  The defining modules are wrapped as well, in case a build
# reaches them by another route (wgp.record_file_metadata delegates to the
# media_recording one, and other modules may call save_video directly).
_SOURCE_MODULES = {"save_video": ("shared.utils.audio_video", "save_video"),
                   "record_file_metadata": ("shared.utils.media_recording",
                                            "record_file_metadata")}


def _log(message):
    patches._log(message)


def _bind(signature, args, kwargs):
    bound = signature.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    values = dict(bound.arguments)
    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD and name in values:
            values.update(values.pop(name) or {})
    return values


def _signature(function):
    try:
        return inspect.signature(function)
    except (TypeError, ValueError):
        return None


def _local_path(value):
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) == 1 else None
    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    if isinstance(value, (str, os.PathLike)) and os.path.isfile(value):
        return os.fspath(value)
    return None


# ---------------------------------------------------------------------- generate_media

def new_job(fallback=False):
    return {"model_type": None, "base_model_type": None, "video_source": None,
            "keep_frames_video_source": "", "pending": None, "last_sha": None,
            "last_path": None, "fallback": fallback}


def wrap_generate_media(original, get_base_model_type=None):
    signature = _signature(original)

    @functools.wraps(original)
    def generate_media(*args, **kwargs):
        job = new_job()
        patches.STATE.stats["hook_generate_media"] += 1
        try:
            values = _bind(signature, args, kwargs) if signature is not None else kwargs
            model_type = values.get("model_type")
            job["model_type"] = model_type
            if model_type and get_base_model_type is not None:
                try:
                    job["base_model_type"] = get_base_model_type(model_type)
                except Exception:
                    job["base_model_type"] = None
            image_prompt_type = str(values.get("image_prompt_type") or "")
            if "V" in image_prompt_type:
                job["video_source"] = _local_path(values.get("video_source"))
            job["keep_frames_video_source"] = str(values.get("keep_frames_video_source") or "")
        except Exception as error:
            _log(f"could not read the job's settings, saved latents are off for it: {error!r}")
        if patches.CONFIG.trace:
            _log(f"trace: job started, model={job['model_type']}, "
                 f"source={os.path.basename(str(job['video_source'])) if job['video_source'] else 'none'}")
        token = patches.JOB.set(job)
        try:
            return original(*args, **kwargs)
        finally:
            try:
                _finish_job(job)
            except Exception as error:                      # never break a render
                _log(f"latents not saved: {error!r}")
            patches.JOB.reset(token)

    return generate_media


# ---------------------------------------------------------------------- continue last video

def wrap_resolve_continue_video(original):
    @functools.wraps(original)
    def resolve_continue_video(*args, **kwargs):
        # marked below, so status() can recognise it
        result = original(*args, **kwargs)
        job = patches.JOB.get()
        if job is not None:
            path = _local_path(result)
            if path:
                job["video_source"] = path
        return result

    return resolve_continue_video


# ---------------------------------------------------------------------- save / record

def wrap_save_video(original):
    signature = _signature(original)

    @functools.wraps(original)
    def save_video(*args, **kwargs):
        patches.STATE.stats["hook_save_video"] += 1
        job = patches.JOB.get()
        if patches.CONFIG.trace:
            _log(f"trace: save hook, job={'yes' if job else 'NO'}, "
                 f"save={patches.CONFIG.save}, store={'yes' if patches.STORE else 'NO'}")
        if job is not None and patches.CONFIG.save and patches.STORE is not None:
            try:
                values = _bind(signature, args, kwargs) if signature is not None else kwargs
                tensor, fps = values.get("tensor"), values.get("fps")
                patches.STATE.save_seen = patches.STATE.generation
                if tensor is not None and fps is not None:
                    job["pending"] = patches.snapshot_for_export(tensor, fps)
                    job["pending_generation"] = patches.STATE.generation
            except Exception as error:
                job["pending"] = (None, f"snapshot failed: {error!r}")
                job["pending_generation"] = patches.STATE.generation
        return original(*args, **kwargs)

    return save_video


def wrap_record_file_metadata(original):
    signature = _signature(original)

    @functools.wraps(original)
    def record_file_metadata(*args, **kwargs):
        result = original(*args, **kwargs)
        patches.STATE.stats["hook_record_file"] += 1
        job = patches.JOB.get()
        if job is None or job.get("pending") is None:
            if patches.CONFIG.trace:
                _log(f"trace: record hook, nothing to save "
                     f"(job={'yes' if job else 'NO'}, "
                     f"snapshot={'none' if job is None else job.get('pending')})")
            return result
        snapshot, reason = job["pending"]
        job["pending"] = None
        if job.get("pending_generation") != patches.STATE.generation:
            # Taken for a window that never reached this point; not this file's.
            snapshot, reason = None, "the frames seen at save time belong to an earlier window"
        if patches.CONFIG.trace:
            _log(f"trace: record hook, snapshot={'yes' if snapshot else 'no'}")
        try:
            values = _bind(signature, args, kwargs) if signature is not None else kwargs
            if values.get("is_image") or values.get("audio_only"):
                return result
            path = _local_path(values.get("video_path"))
            if path is None:
                return result
            name = os.path.basename(path)
            if snapshot is None:
                patches.STATE.stats["save_skipped"] += 1
                _log(f"latents not saved for {name}: {reason}")
                return result
            if patches.CONFIG.save_scope == "every":
                _store_snapshot(job, path, snapshot, values.get("configs") or {})
            else:
                # One entry per job: remember the newest window and write it when
                # the job ends, so a long job hashes and writes once, not per window.
                job["final"] = (path, snapshot, values.get("configs") or {})
        except Exception as error:                          # never break a render
            patches.STATE.stats["save_skipped"] += 1
            _log(f"latents not saved: {error!r}")
        return result

    return record_file_metadata


def _finish_job(job):
    """Write the job's single entry, for save_scope "final"."""
    pending = job.pop("final", None)
    if pending is None:
        return
    path, snapshot, configs = pending
    if not os.path.isfile(path):
        # The window's file was replaced by a later one that was not itself
        # snapshotted (post-processing, say); nothing to attach an entry to.
        _log(f"latents not saved: {os.path.basename(path)} is no longer on disk")
        patches.STATE.stats["save_skipped"] += 1
        return
    _store_snapshot(job, path, snapshot, configs)


def _store_snapshot(job, path, snapshot, configs):
    store = patches.STORE
    sha = store.sha256_file(path)
    manifest = dict(snapshot["manifest"])
    manifest.update({"video_name": os.path.basename(path),
                     "video_path": os.path.abspath(path),
                     "model_type": configs.get("model_type") or job.get("model_type"),
                     "base_model_type": job.get("base_model_type"),
                     "model_filename": configs.get("model_filename")})
    store.save(sha, snapshot["arrays"], manifest)
    patches.STATE.stats["saved"] += 1
    size = os.path.getsize(store.path_for(sha)) / 1024 ** 2
    _log(f"saved latents for {os.path.basename(path)} ({size:.1f} MB)")

    # A sliding-window job records one file per window.  When Wan2GP replaced
    # the previous window's file, that entry now points at nothing.
    previous, previous_path = job.get("last_sha"), job.get("last_path")
    if previous and previous != sha and previous_path and not os.path.exists(previous_path):
        store.delete(previous)
    job["last_sha"], job["last_path"] = sha, os.path.abspath(path)

    limit = int(patches.CONFIG.store_limit_gb * 1024 ** 3)
    removed = store.enforce_limit(limit, keep={sha})
    if removed:
        _log(f"removed {removed} least recently used saved latent file(s) to stay "
             f"under {patches.CONFIG.store_limit_gb:g} GB")


# ---------------------------------------------------------------------- install

def _is_ours(function):
    """Whether this plugin's wrapper is anywhere in function's __wrapped__ chain."""
    for _ in range(64):
        if function is None:
            return False
        if getattr(function, _MARK, False):
            return True
        function = getattr(function, "__wrapped__", None)
    return False


def _wrap_global(name):
    current = _NAMESPACE.get(name)
    if not callable(current):
        return False
    wrapper = _HOOKS[name](current)
    setattr(wrapper, _MARK, True)
    _NAMESPACE[name] = wrapper
    _INSTALLED[name] = wrapper
    return True


def _wrap_source_modules():
    """Wrap the functions where they are defined, as well as wgp.py's globals."""
    for key, (module_name, attribute) in _SOURCE_MODULES.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        current = getattr(module, attribute, None)
        if not callable(current) or _is_ours(current):
            continue
        wrapper = _HOOKS[key](current)
        setattr(wrapper, _MARK, True)
        setattr(module, attribute, wrapper)


def ensure_installed():
    """Wrap again any wgp.py function another plugin replaced.  Returns the names."""
    if _NAMESPACE is None:
        return []
    _wrap_source_modules()
    redone = [name for name in _HOOKS
              if callable(_NAMESPACE.get(name)) and not _is_ours(_NAMESPACE.get(name))
              and _wrap_global(name)]
    if redone:
        _log(f"Wan2GP's {', '.join(redone)} had been replaced by another plugin that "
             f"loaded after this one (H3 Latent Continue does this). Wrapped again on "
             f"top of it; saving and restoring are active.")
    return redone


def _on_generate():
    """Called by patches at the start of every generate(): heal, then make sure a job exists."""
    redone = ensure_installed()
    if patches.JOB.get() is None and (redone or _NAMESPACE is not None):
        # generate_media was not ours when this job started, so there is no job
        # context.  Saving still works from here on; the source file is unknown,
        # so this job's first window cannot restore.
        patches.JOB.set(new_job(fallback=True))


def status():
    """What is and is not wired up, for the panel's Diagnostics button."""
    report = {"namespace": _NAMESPACE is not None, "hooks": {}, "resolver": False}
    for name in ("generate_media", "save_video", "record_file_metadata"):
        if _NAMESPACE is not None:
            current = _NAMESPACE.get(name)
            report["hooks"][name] = ("missing from wgp.py" if not callable(current)
                                     else "ours" if _is_ours(current)
                                     else "replaced by another plugin")
        elif name in _INSTALLED:
            report["hooks"][name] = "installed without access to wgp.py's globals"
        else:
            report["hooks"][name] = "not installed"
    try:
        from shared.prompt_enhancer import images
        report["resolver"] = _is_ours(images.resolve_continue_video) or hasattr(
            images.resolve_continue_video, "__wrapped__")
    except Exception:
        report["resolver"] = False
    return report


def install(plugin):
    """Wrap the wgp.py functions.  Returns a list of problems (empty on success)."""
    global _NAMESPACE
    problems = []
    _HOOKS.update({
        "generate_media": lambda f: wrap_generate_media(
            f, getattr(plugin, "get_base_model_type", None)),
        "save_video": wrap_save_video,
        "record_file_metadata": wrap_record_file_metadata,
    })
    setter = getattr(plugin, "_set_wgp_global_func", None)
    namespace = getattr(setter, "__globals__", None)
    if isinstance(namespace, dict):
        _NAMESPACE = namespace
        _wrap_source_modules()
        for name in _HOOKS:
            if not callable(namespace.get(name)):
                problems.append(f"Wan2GP has no {name}")
            elif not _is_ours(namespace.get(name)):
                _wrap_global(name)
        patches.ON_GENERATE = _on_generate
    else:
        # No route to wgp.py's globals: fall back to the injected references.
        # This cannot detect or repair another plugin replacing the wrappers.
        problems.append("cannot reach wgp.py's globals; if another plugin wraps the "
                        "same functions after this one, saving may stop silently")
        for name, make in _HOOKS.items():
            if name in _INSTALLED:
                continue
            current = getattr(plugin, name, None)
            if not callable(current):
                problems.append(f"Wan2GP did not provide {name}")
                continue
            wrapper = make(current)
            setattr(wrapper, _MARK, True)
            plugin.set_global(name, wrapper)
            _INSTALLED[name] = wrapper

    if "resolve_continue_video" not in _INSTALLED:
        try:
            from shared.prompt_enhancer import images
            original = images.resolve_continue_video
            wrapper = wrap_resolve_continue_video(original)
            setattr(wrapper, _MARK, True)
            images.resolve_continue_video = wrapper
            _INSTALLED["resolve_continue_video"] = images.resolve_continue_video
        except Exception as error:
            problems.append(f"'Continue last video' cannot be followed ({error!r}); "
                            f"pick the file in Video to Continue instead")
    return problems
