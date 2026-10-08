#!/usr/bin/env python3
"""The latent store: what it accepts, what it refuses, and housekeeping.

numpy + safetensors only.

WHAT IS BEING CHECKED
---------------------
1. A round trip preserves every tensor exactly and records the video hash.
2. Loading refuses, with a reason, anything that is not a usable entry: a
   foreign safetensors file (a LoRA, an H3 Latent Continue checkpoint), a newer
   format version, an entry filed under the wrong hash, NaN values, missing or
   unexpected tensors, and nonsense geometry.  Refusal is an exception, so the
   caller can say why instead of silently re-encoding.
3. The hash follows the file's content: editing the file changes the hash, so a
   re-encoded or trimmed video can never pick up the original's latents.
4. Saving leaves no temporary files; entries() ignores anything that is not an
   entry; prune_missing and clear do what they say.

    python tests/test_store.py
"""

import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile
import time

import numpy as np
from safetensors.numpy import save_file

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("h3lc_store", ROOT / "store.py")
store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(store)

FAILURES = []


def check(label, condition, detail=""):
    print(f"  [{'pass' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def refused(action, fragment):
    try:
        action()
    except ValueError as error:
        return fragment in str(error), str(error)
    return False, "accepted"


def arrays(seed=0):
    rng = np.random.default_rng(seed)
    return {"video": rng.standard_normal((1, 24, 17, 48, 84)).astype(np.float32),
            "signatures": rng.random((12, 16, 16)).astype(np.float32),
            "signature": rng.random((16, 16)).astype(np.float32),
            "audio": rng.standard_normal((1, 32, 2, 192)).astype(np.float32),
            "audio_signature": rng.random(32).astype(np.float32)}


MANIFEST = {"fps": 24.0, "width": 1344, "height": 768, "total_latents": 37,
            "expected_latents": 37, "frames": 124}

print(__doc__.strip().splitlines()[0])
print()

work = pathlib.Path(tempfile.mkdtemp(prefix="h3lc-store-"))
latents = store.LatentStore(str(work / "latents"))
video = work / "clip.mp4"
video.write_bytes(os.urandom(4096))
sha = store.sha256_file(str(video))

print("round trip")
data = arrays()
latents.save(sha, data, {**MANIFEST, "video_path": str(video)})
manifest, loaded = latents.load(sha)
check("tensors survive the round trip",
      all(np.allclose(loaded[k], v, atol=1e-2) for k, v in data.items())
      and all(np.array_equal(loaded[k], data[k]) for k in ("signatures", "signature")))
check("the manifest records the video hash", manifest["video_sha256"] == sha)
check("no temporary files are left behind",
      sorted(p.name for p in (work / "latents").iterdir()) == [sha + ".safetensors"])
size_mb = os.path.getsize(latents.path_for(sha)) / 1024 ** 2
check("a 1344x768 entry is a sensible size", 2 < size_mb < 12, f"{size_mb:.1f} MB")
check("latents are stored as float16, fingerprints as float32",
      loaded["video"].dtype == np.float16 and loaded["signatures"].dtype == np.float32)
check("an unknown hash is simply absent", latents.load("0" * 64) is None)
print()

print("refusals")


def write_raw(name, tensors, metadata):
    path = work / "latents" / f"{name}.safetensors"
    save_file(tensors, str(path), metadata=metadata)


fake = "1" * 64
write_raw(fake, {"lora_down": np.zeros((4, 4), np.float32)}, {"format": "pt"})
check("a foreign safetensors file (e.g. a LoRA)", *refused(lambda: latents.load(fake), "not an H3 Latent Carry entry"))

write_raw(fake, {"video": np.zeros((1, 24, 3, 4, 4), np.float32)},
          {"format": "wan2gp.h3.latent-continuation", "version": "2", "manifest": "{}"})
check("an H3 Latent Continue checkpoint", *refused(lambda: latents.load(fake), "not an H3 Latent Carry entry"))

write_raw(fake, data, {"format": store.FORMAT, "version": "99", "manifest": "{}"})
check("a newer format version", *refused(lambda: latents.load(fake), "version 99"))

shutil.copy(latents.path_for(sha), latents.path_for(fake))
check("an entry filed under a different hash", *refused(lambda: latents.load(fake), "different video hash"))


def save_variant(label, fragment, **changes):
    variant = {**arrays(), **changes}
    variant = {k: v for k, v in variant.items() if v is not None}
    check(label, *refused(lambda: latents.save(fake, variant, MANIFEST), fragment))


bad = arrays()["video"].copy()
bad[0, 0, 0, 0, 0] = np.nan
save_variant("NaN in the latents", "NaN", video=bad)
save_variant("missing fingerprints", "missing 'signatures'", signatures=None)
save_variant("a video tensor of the wrong rank", "expected [1, C, T, h, w]",
             video=np.zeros((24, 17, 48, 84), np.float32))
save_variant("an integer tensor", "not floating point",
             audio=np.zeros((1, 32, 2, 8), np.int32))
check("an unexpected tensor",
      *refused(lambda: latents.save(fake, {**arrays(), "extra": np.zeros(1, np.float32)}, MANIFEST),
               "unexpected tensors"))

latents.save(fake, arrays(), {**MANIFEST, "fps": 0})
check("nonsense geometry in the manifest", *refused(lambda: latents.load(fake), "invalid fps"))
latents.delete(fake)
print()

print("the hash follows the content")
before = store.sha256_file(str(video))
time.sleep(0.01)
with open(video, "ab") as handle:
    handle.write(b"x")                        # one byte appended, e.g. by a remux
after = store.sha256_file(str(video))
check("editing the file changes its hash", before != after)
check("so the edited file finds no entry", latents.load(after) is None)
print()

print("housekeeping")
(work / "latents" / "notes.txt").write_text("not an entry")
(work / "latents" / ("a" * 63 + ".safetensors")).write_bytes(b"")
check("entries() ignores anything that is not an entry",
      [e[0] for e in latents.entries()] == [sha])
gone = work / "gone.mp4"
gone.write_bytes(b"soon deleted")
gone_sha = store.sha256_file(str(gone))
latents.save(gone_sha, arrays(1), {**MANIFEST, "video_path": str(gone)})
gone.unlink()
check("prune_missing removes entries whose video is gone, and only those",
      latents.prune_missing() == 1 and latents.has(sha) and not latents.has(gone_sha))
check("clear removes every entry", latents.clear() == 1 and latents.stats() == (0, 0))
check("path_for refuses anything but a digest",
      *refused(lambda: latents.path_for("../escape"), "not a SHA-256"))
print()

shutil.rmtree(work, ignore_errors=True)
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
