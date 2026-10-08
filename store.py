"""Saved latents, kept inside the plugin folder and found by the video they belong to.

Each entry is one safetensors file named after the SHA-256 of the finished video
it was captured with:

    <plugin>/latents/<sha256 of the mp4>.safetensors

That name is the whole index.  Continuing a video means hashing the file in the
Video to Continue field and checking whether an entry of that name exists, so
entries are selected automatically, survive the video being moved or renamed,
and can never be paired with the wrong file: any edit, trim or re-encode changes
the hash and the lookup simply misses.

The module is numpy + safetensors only - no torch - so it can be tested without
a GPU stack and so a corrupt entry can never execute code (no pickle).

What an entry holds (all float32):

    video             [1, C, T, h, w]   the last T latents of the final window
    signatures        [k, 16, 16]       fingerprints of that window's last k decoded frames
    signature         [16, 16]          fingerprint of decoded frame -2      (optional)
    audio             [..., A]          the last A audio latents              (optional)
    audio_signature   [32]              envelope of the last 0.5s of audio   (optional)

plus a JSON manifest in the safetensors metadata.  See _validate for the rules.
"""

import hashlib
import json
import math
import os
import tempfile
import time

import numpy as np

FORMAT = "wan2gp.h3-latent-carry"
FORMAT_VERSION = 1
SUFFIX = ".safetensors"
MAX_ENTRY_BYTES = 512 * 1024 ** 2       # a real entry is ~5-20 MB; refuse anything absurd
REQUIRED = ("video", "signatures")
HALF = ("video", "audio")
OPTIONAL = ("signature", "audio", "audio_signature")

_HASH_CACHE = {}                        # (path, size, mtime_ns) -> sha256


def sha256_file(path):
    """SHA-256 of a file, cached on (path, size, mtime) for repeat lookups."""
    stat = os.stat(path)
    key = (os.path.abspath(path), stat.st_size, stat.st_mtime_ns)
    cached = _HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 ** 2), b""):
            digest.update(chunk)
    result = digest.hexdigest()
    if len(_HASH_CACHE) > 256:
        _HASH_CACHE.clear()
    _HASH_CACHE[key] = result
    return result


def _is_sha(text):
    return isinstance(text, str) and len(text) == 64 and all(c in "0123456789abcdef" for c in text)


class LatentStore:
    # Exposed on the instance so patches.py can hash without importing this
    # module (it is loaded standalone by some tests).
    sha256_file = staticmethod(sha256_file)

    def __init__(self, root):
        self.root = os.path.abspath(root)

    # ------------------------------------------------------------------ paths

    def path_for(self, sha):
        if not _is_sha(sha):
            raise ValueError(f"not a SHA-256 digest: {sha!r}")
        return os.path.join(self.root, sha + SUFFIX)

    def has(self, sha):
        return _is_sha(sha) and os.path.isfile(self.path_for(sha))

    # ------------------------------------------------------------------ write

    def save(self, sha, arrays, manifest):
        """Write an entry atomically.  Replaces an entry for the same video.

        Same hash means byte-identical video, so replacing is always correct:
        it can only happen when the same file is recorded again.
        """
        _validate_arrays(arrays)
        os.makedirs(self.root, exist_ok=True)
        from safetensors.numpy import save_file
        # Latents go to float16: they are VAE-space conditioning, where the
        # precision is far below what float32 records, and it halves an entry.
        # Fingerprints stay float32 - they are compared against a tolerance.
        payload = {name: np.ascontiguousarray(
                       value, dtype=np.float16 if name in HALF else np.float32)
                   for name, value in arrays.items() if value is not None}
        info = dict(manifest)
        info["video_sha256"] = sha
        info.setdefault("saved_at", time.time())
        metadata = {"format": FORMAT, "version": str(FORMAT_VERSION),
                    "manifest": json.dumps(info, ensure_ascii=False, default=str)}
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=SUFFIX, dir=self.root)
        os.close(fd)
        try:
            save_file(payload, tmp, metadata=metadata)
            os.replace(tmp, self.path_for(sha))
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return self.path_for(sha)

    # ------------------------------------------------------------------ read

    def load(self, sha):
        """Return (manifest, arrays) or None if there is no entry.

        Raises ValueError for an entry that exists but is not usable, so the
        caller can report it instead of silently re-encoding.
        """
        if not self.has(sha):
            return None
        path = self.path_for(sha)
        if os.path.getsize(path) > MAX_ENTRY_BYTES:
            raise ValueError(f"saved latents are implausibly large ({os.path.getsize(path)} bytes)")
        from safetensors import safe_open
        with safe_open(path, framework="np") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("format") != FORMAT:
                raise ValueError("not an H3 Latent Carry entry")
            if metadata.get("version") != str(FORMAT_VERSION):
                raise ValueError(f"entry format version {metadata.get('version')} is not "
                                 f"supported (this version reads {FORMAT_VERSION})")
            arrays = {name: handle.get_tensor(name) for name in handle.keys()}
            manifest = json.loads(metadata.get("manifest") or "{}")
        if manifest.get("video_sha256") != sha:
            raise ValueError("entry is filed under a different video hash than it records")
        _validate_arrays(arrays)
        _validate_manifest(manifest)
        return manifest, arrays

    def signatures(self, sha):
        """Just the fingerprint grid of an entry, without reading the latents."""
        from safetensors import safe_open
        with safe_open(self.path_for(sha), framework="np") as handle:
            if (handle.metadata() or {}).get("format") != FORMAT:
                raise ValueError("not an H3 Latent Carry entry")
            return handle.get_tensor("signatures")

    def touch(self, sha):
        """Mark an entry as recently used, so eviction keeps it longer."""
        try:
            os.utime(self.path_for(sha))
        except OSError:
            pass

    # ------------------------------------------------------------------ housekeeping

    def delete(self, sha):
        try:
            os.remove(self.path_for(sha))
            return True
        except (OSError, ValueError):
            return False

    def entries(self):
        """[(sha, bytes, mtime)], newest first.  Ignores anything that is not an entry."""
        result = []
        try:
            names = os.listdir(self.root)
        except OSError:
            return result
        for name in names:
            sha = name[:-len(SUFFIX)] if name.endswith(SUFFIX) else None
            if not _is_sha(sha):
                continue
            try:
                stat = os.stat(os.path.join(self.root, name))
            except OSError:
                continue
            result.append((sha, stat.st_size, stat.st_mtime))
        result.sort(key=lambda item: item[2], reverse=True)
        return result

    def stats(self):
        entries = self.entries()
        return len(entries), sum(size for _, size, _ in entries)

    def enforce_limit(self, max_bytes, keep=()):
        """Delete least recently used entries until the store fits.  Returns count removed."""
        if max_bytes is None or max_bytes <= 0:
            return 0
        entries = self.entries()
        total = sum(size for _, size, _ in entries)
        removed = 0
        for sha, size, _ in reversed(entries):          # oldest first
            if total <= max_bytes:
                break
            if sha in keep:
                continue
            if self.delete(sha):
                total -= size
                removed += 1
        return removed

    def manifest(self, sha):
        """Just the manifest, without reading tensors.  None if unreadable."""
        try:
            from safetensors import safe_open
            with safe_open(self.path_for(sha), framework="np") as handle:
                return json.loads((handle.metadata() or {}).get("manifest") or "{}")
        except Exception:
            return None

    def prune_missing(self):
        """Delete entries whose recorded video no longer exists at its last known path.

        An entry stays valid for a moved file (the hash is what matters), so this
        is a manual tidy-up, never automatic.
        """
        removed = 0
        for sha, _, _ in self.entries():
            info = self.manifest(sha)
            path = (info or {}).get("video_path")
            if path and not os.path.exists(path):
                removed += self.delete(sha)
        return removed

    def clear(self):
        removed = 0
        for sha, _, _ in self.entries():
            removed += self.delete(sha)
        return removed


# ---------------------------------------------------------------------- validation

def _validate_arrays(arrays):
    for name in REQUIRED:
        if arrays.get(name) is None:
            raise ValueError(f"entry is missing '{name}'")
    unknown = set(arrays) - set(REQUIRED) - set(OPTIONAL)
    if unknown:
        raise ValueError(f"entry has unexpected tensors {sorted(unknown)}")
    video, signatures = arrays["video"], arrays["signatures"]
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[2] < 2 or min(video.shape[3:]) < 1:
        raise ValueError(f"video latents have shape {tuple(video.shape)}, expected [1, C, T, h, w]")
    if signatures.ndim != 3 or signatures.shape[0] < 2:
        raise ValueError(f"fingerprints have shape {tuple(signatures.shape)}, expected [k, 16, 16]")
    signature = arrays.get("signature")
    if signature is not None and tuple(signature.shape) != tuple(signatures.shape[1:]):
        raise ValueError("join fingerprint does not match the fingerprint grid")
    audio = arrays.get("audio")
    if audio is not None and (audio.ndim < 2 or audio.shape[-1] < 1):
        raise ValueError(f"audio latents have shape {tuple(audio.shape)}")
    for name, value in arrays.items():
        if value is None:
            continue
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"'{name}' is not floating point")
        if not np.isfinite(value).all():
            raise ValueError(f"'{name}' contains NaN or infinity")


def _validate_manifest(manifest):
    fps = manifest.get("fps")
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 0 < fps <= 240:
        raise ValueError(f"entry records an invalid fps ({fps!r})")
    for key in ("total_latents", "width", "height"):
        value = manifest.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"entry records an invalid {key} ({value!r})")
