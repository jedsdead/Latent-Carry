#!/usr/bin/env python3
"""Window-join statistics for a MiniMax H3 sliding-window render in Wan2GP.

Measures what happens at each window join, and how the picture and sound
drift down a chain of windows. Written for the Sliding Window Latents A/B
protocol: render the same seed and prompt with the plugin off and on, run
this on both, and compare.

    python measure_joins.py shot.mp4 --window 362 --overlap 18
    python measure_joins.py stock.mp4 --baseline carried.mp4 --window 362 --overlap 18
    python measure_joins.py shot.mp4 --window 362 --overlap 18 --json out.json

Companion to measure_tone.py, which reports tone frame by frame. This one is
about the joins specifically, and it reads audio as well as picture.

Needs ffmpeg and ffprobe on PATH (Wan2GP already requires them) and numpy.
Nothing else - no torch, no opencv, no GPU.


WHAT IT MEASURES AND WHY
------------------------
Wan2GP muxes sliding windows into a single file, so the audio either side of
a join is contiguous by construction. The open questions are therefore about
steps and drift rather than about whether one clip continues another:

  1. Does the join show a step - in tone, in level, in spectral balance -
     that the surrounding frames do not?
  2. Does the picture or the sound degrade progressively from window to
     window? That is the round-trip cost the plugin exists to remove.
  3. Does the motion stumble at the join - a repeated or skipped instant,
     which is what a coordinate skew in the history block looks like from
     the outside?

Measurement 3 is the one to read when testing H3LC_FIX_COORDS.


READING THE OUTPUT
------------------
Joins:      a step at the join that is large relative to the within-window
            baseline is the round trip showing up. Compare the same number
            between two runs; the absolute value alone means little.

Chain:      per-window means and the slope across windows. A negative HF
            slope is the top end being worn away link by link. If latent
            carry is doing its job the slope flattens.

Continuity: ratio of the frame-to-frame difference at the join to the
            local median. About 1.0 is a clean join.
              >> 1  the model jumped - a skipped instant.
              << 1  the model repeated an instant it had already shown.
            A history block placed too early tends to produce the second.
            The ratio shows the magnitude of a skew, not its sign in
            frames; use the repeat/skip verdict for the direction.

            It is confounded when appearance changes sharply at the same
            join, so read it across several joins and between two runs
            rather than from any single number.
"""

import argparse
import json
import shutil
import subprocess
import sys

import numpy as np


# --------------------------------------------------------------------------
# media loading
# --------------------------------------------------------------------------

def _require_tools():
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on PATH; it ships with Wan2GP's "
                             f"requirements, so activate that environment first")


def probe(path):
    """Container metadata via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", path],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"ffprobe failed on {path}:\n{out.stderr.strip()}")
    data = json.loads(out.stdout)

    info = {"path": path, "fps": None, "width": None, "height": None,
            "sample_rate": None, "channels": 0}
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and info["fps"] is None:
            rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
            num, _, den = rate.partition("/")
            try:
                info["fps"] = float(num) / float(den or 1)
            except (ValueError, ZeroDivisionError):
                info["fps"] = None
            info["width"] = stream.get("width")
            info["height"] = stream.get("height")
        elif stream.get("codec_type") == "audio" and info["sample_rate"] is None:
            info["sample_rate"] = int(stream.get("sample_rate", 0)) or None
            info["channels"] = int(stream.get("channels", 0))
    return info


def load_luma(path, width):
    """Decode to single-channel luma at a reduced width. Returns [N, H, W] float32."""
    meta = probe(path)
    if meta["width"] is None:
        raise SystemExit(f"{path} has no video stream")
    height = max(2, int(round(width * meta["height"] / meta["width"])) // 2 * 2)
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-vf", f"scale={width}:{height}", "-pix_fmt", "gray",
         "-f", "rawvideo", "-"],
        capture_output=True)
    if out.returncode != 0:
        raise SystemExit(f"ffmpeg video decode failed:\n{out.stderr.decode()[:400]}")
    frames = np.frombuffer(out.stdout, dtype=np.uint8)
    count = frames.size // (width * height)
    if count == 0:
        raise SystemExit(f"no frames decoded from {path}")
    return frames[:count * width * height].reshape(count, height, width).astype(np.float32)


def load_audio(path, sample_rate=32000):
    """Decode to mono float32 at sample_rate. Returns None when there is no audio."""
    meta = probe(path)
    if not meta["channels"]:
        return None, sample_rate
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"],
        capture_output=True)
    if out.returncode != 0 or not out.stdout:
        return None, sample_rate
    return np.frombuffer(out.stdout, dtype=np.float32).copy(), sample_rate


# --------------------------------------------------------------------------
# join positions
# --------------------------------------------------------------------------

def joins_from_schedule(window, overlap, total_frames):
    """Join frame indices, using the same convention as measure_tone.py.

    stride = window - overlap; the first join sits at the last carried
    frame (overlap - 1) and they repeat every stride after that.
    """
    stride = window - overlap
    if stride < 1:
        raise SystemExit("overlap must be smaller than window")
    positions, boundary = [], max(overlap - 1, 1)
    while boundary < total_frames - 1:
        positions.append(boundary)
        boundary += stride
    return positions


def joins_auto(luma, minimum_gap=24, count=None):
    """Locate joins from the picture alone, for when the schedule is unknown.

    Frame-to-frame difference, high-pass filtered against a local median, then
    the strongest peaks kept subject to a minimum spacing.
    """
    motion = np.abs(np.diff(luma, axis=0)).mean(axis=(1, 2))
    if motion.size < 3 * minimum_gap:
        return []
    pad = minimum_gap // 2
    padded = np.pad(motion, pad, mode="edge")
    local = np.array([np.median(padded[i:i + minimum_gap]) for i in range(motion.size)])
    excess = motion - local
    scale = np.median(np.abs(excess - np.median(excess))) or 1e-6
    score = excess / (1.4826 * scale)

    order = np.argsort(score)[::-1]
    picked = []
    limit = count if count else max(1, motion.size // (minimum_gap * 4))
    for index in order:
        if score[index] < 3.0 and len(picked) >= 1 and count is None:
            break
        if all(abs(int(index) - p) >= minimum_gap for p in picked):
            picked.append(int(index) + 1)
        if len(picked) >= limit:
            break
    return sorted(picked)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def spectral_features(segment, sample_rate, split_hz=4000.0):
    """(rms_db, hf_ratio, centroid_hz) for one audio segment."""
    if segment.size < 64:
        return None
    rms = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
    rms_db = 20.0 * np.log10(max(rms, 1e-9))

    windowed = segment.astype(np.float64) * np.hanning(segment.size)
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(segment.size, 1.0 / sample_rate)
    total = spectrum.sum()
    if total <= 0:
        return rms_db, 0.0, 0.0
    hf_ratio = float(spectrum[freqs >= split_hz].sum() / total)
    centroid = float((spectrum * freqs).sum() / total)
    return rms_db, hf_ratio, centroid


def luma_percentiles(frames):
    """p10 / median / p90 per frame."""
    flat = frames.reshape(frames.shape[0], -1)
    return np.percentile(flat, [10, 50, 90], axis=1).T


def detail_energy(frames):
    """Per-frame high-frequency picture detail: mean |Laplacian|.

    A VAE round trip softens fine detail before it touches overall tone, so
    this falls earlier down a chain than the luma percentiles do.
    """
    centre = frames[:, 1:-1, 1:-1]
    laplacian = (frames[:, :-2, 1:-1] + frames[:, 2:, 1:-1]
                 + frames[:, 1:-1, :-2] + frames[:, 1:-1, 2:] - 4.0 * centre)
    return np.abs(laplacian).mean(axis=(1, 2))


def step(series, at, span):
    """Mean after minus mean before, plus the within-window baseline volatility."""
    before = series[max(at - span, 0):at]
    after = series[at:at + span]
    if before.size < 2 or after.size < 2:
        return None
    baseline = float(np.median(np.abs(np.diff(np.concatenate((before, after))))))
    return float(after.mean() - before.mean()), baseline


def continuity(luma, at):
    """Frame-to-frame difference at the join against the local median.

    Also reports whether the join looks like a repeated instant or a skipped
    one, by comparing the join difference to the one-frame-wider gap.
    """
    diffs = np.abs(np.diff(luma, axis=0)).mean(axis=(1, 2))
    if at < 3 or at >= diffs.size - 2:
        return None
    local = np.concatenate((diffs[max(at - 9, 0):at - 1], diffs[at + 1:at + 9]))
    if local.size < 4:
        return None
    median = float(np.median(local)) or 1e-9
    ratio = float(diffs[at - 1] / median)

    wider = float(np.abs(luma[at + 1] - luma[at - 2]).mean())
    expected = 3.0 * median
    if ratio < 0.55:
        verdict = "repeated instant"
    elif ratio > 1.8:
        verdict = "skipped instant"
    else:
        verdict = "smooth"
    return {"ratio": ratio, "verdict": verdict,
            "join_delta": float(diffs[at - 1]), "local_median": median,
            "wider_gap": wider, "wider_expected": expected}


def slope_per_window(values):
    """Least-squares slope of a per-window series, and its total change."""
    if len(values) < 2:
        return None, None
    x = np.arange(len(values), dtype=np.float64)
    slope = float(np.polyfit(x, np.asarray(values, dtype=np.float64), 1)[0])
    return slope, float(values[-1] - values[0])


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def analyse(path, args):
    meta = probe(path)
    fps = args.fps or meta["fps"] or 24.0
    luma = load_luma(path, args.scale)
    audio, sample_rate = load_audio(path)

    if args.joins:
        joins = [int(v) for v in args.joins.split(",") if v.strip()]
    elif args.auto_joins or not args.window:
        joins = joins_auto(luma)
    else:
        joins = joins_from_schedule(args.window, args.overlap, luma.shape[0])
    joins = [j for j in joins if 2 < j < luma.shape[0] - 2]

    percentiles = luma_percentiles(luma)
    detail = detail_energy(luma)

    result = {"path": path, "frames": int(luma.shape[0]), "fps": fps,
              "resolution": f"{meta['width']}x{meta['height']}",
              "analysed_at": f"{luma.shape[2]}x{luma.shape[1]}",
              "has_audio": audio is not None, "joins": joins,
              "join_detail": [], "chain": {}}

    span = args.span

    # ---- per join --------------------------------------------------------
    for at in joins:
        entry = {"frame": at, "time": at / fps}

        picture = step(detail, at, span)
        if picture:
            entry["detail_step"], entry["detail_baseline"] = picture
        shadows = step(percentiles[:, 0], at, span)
        if shadows:
            entry["p10_step"], _ = shadows
        median = step(percentiles[:, 1], at, span)
        if median:
            entry["median_step"], _ = median

        entry["continuity"] = continuity(luma, at)

        if audio is not None:
            width = int(args.audio_window * sample_rate)
            centre = int(round(at / fps * sample_rate))
            before = audio[max(centre - width, 0):centre]
            after = audio[centre:centre + width]
            f_before, f_after = (spectral_features(before, sample_rate),
                                 spectral_features(after, sample_rate))
            if f_before and f_after:
                entry["rms_step_db"] = f_after[0] - f_before[0]
                entry["hf_step"] = f_after[1] - f_before[1]
                entry["centroid_step_hz"] = f_after[2] - f_before[2]
            guard = int(0.012 * sample_rate)
            local = audio[max(centre - width, 0):centre + width]
            if local.size > 2 * guard and guard > 1:
                edge = audio[max(centre - guard, 0):centre + guard]
                rms = float(np.sqrt(np.mean(local.astype(np.float64) ** 2))) or 1e-9
                entry["click_score"] = float(np.abs(np.diff(edge)).max() / rms)

        result["join_detail"].append(entry)

    # ---- chain trend -----------------------------------------------------
    bounds = [0] + joins + [luma.shape[0]]
    segments = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    segments = [(a, b) for a, b in segments if b - a >= 8]

    per_window_detail, per_window_hf, per_window_centroid, per_window_p10 = [], [], [], []
    for a, b in segments:
        per_window_detail.append(float(detail[a:b].mean()))
        per_window_p10.append(float(percentiles[a:b, 0].mean()))
        if audio is not None:
            chunk = audio[int(a / fps * sample_rate):int(b / fps * sample_rate)]
            features = spectral_features(chunk, sample_rate)
            if features:
                per_window_hf.append(features[1])
                per_window_centroid.append(features[2])

    chain = {"windows": len(segments), "detail": per_window_detail,
             "p10": per_window_p10, "hf_ratio": per_window_hf,
             "centroid_hz": per_window_centroid}
    for key in ("detail", "p10", "hf_ratio", "centroid_hz"):
        slope, total = slope_per_window(chain[key])
        chain[f"{key}_slope"], chain[f"{key}_total"] = slope, total
    result["chain"] = chain
    return result


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _fmt(value, spec="+7.3f"):
    return "     --" if value is None else format(value, spec)


def report(result, args):
    print(f"\n{'=' * 74}")
    print(f"{result['path']}")
    print(f"{'=' * 74}")
    print(f"  {result['frames']} frames at {result['fps']:.3f} fps"
          f"  ({result['frames'] / result['fps']:.2f}s), {result['resolution']}"
          f", analysed at {result['analysed_at']}")
    print(f"  audio: {'yes' if result['has_audio'] else 'none'}"
          f"   joins: {len(result['joins'])}"
          + (f" at {result['joins']}" if len(result['joins']) <= 12 else ""))

    if not result["joins"]:
        print("\n  No joins located. Pass --window/--overlap, or --joins a,b,c.")
        return

    print("\n  JOINS   (step across the join; baseline is the within-window "
          "frame-to-frame median)")
    print(f"  {'frame':>7} {'time':>7} {'detail':>8} {'base':>7} "
          f"{'p10':>7} {'median':>7} | {'rms dB':>7} {'HF':>7} {'centroid':>9} {'click':>6}")
    for entry in result["join_detail"]:
        print(f"  {entry['frame']:7d} {entry['time']:6.2f}s "
              f"{_fmt(entry.get('detail_step'), '+8.3f')} "
              f"{_fmt(entry.get('detail_baseline'), '7.3f')} "
              f"{_fmt(entry.get('p10_step'), '+7.2f')} "
              f"{_fmt(entry.get('median_step'), '+7.2f')} | "
              f"{_fmt(entry.get('rms_step_db'), '+7.2f')} "
              f"{_fmt(entry.get('hf_step'), '+7.4f')} "
              f"{_fmt(entry.get('centroid_step_hz'), '+9.1f')} "
              f"{_fmt(entry.get('click_score'), '6.2f')}")

    print("\n  CONTINUITY   (join frame-difference / local median; ~1.0 is a clean join)")
    for entry in result["join_detail"]:
        cont = entry.get("continuity")
        if not cont:
            continue
        print(f"  {entry['frame']:7d}  ratio {cont['ratio']:5.2f}   {cont['verdict']}")
    ratios = [e["continuity"]["ratio"] for e in result["join_detail"] if e.get("continuity")]
    if ratios:
        print(f"  {'mean':>7}  ratio {np.mean(ratios):5.2f}"
              f"   (spread {np.std(ratios):.2f})")

    chain = result["chain"]
    print(f"\n  CHAIN   ({chain['windows']} windows; slope is per window)")
    rows = (("picture detail", "detail", "{:8.3f}", "{:+8.4f}"),
            ("shadows (p10)", "p10", "{:8.2f}", "{:+8.4f}"),
            ("audio HF ratio", "hf_ratio", "{:8.4f}", "{:+8.5f}"),
            ("audio centroid", "centroid_hz", "{:8.1f}", "{:+8.2f}"))
    for label, key, value_fmt, slope_fmt in rows:
        series = chain.get(key) or []
        if len(series) < 2:
            continue
        print(f"    {label:16s} first {value_fmt.format(series[0])}"
              f"   last {value_fmt.format(series[-1])}"
              f"   slope {slope_fmt.format(chain[f'{key}_slope'])}"
              f"   total {slope_fmt.format(chain[f'{key}_total'])}")


def compare(primary, baseline):
    print(f"\n{'=' * 74}")
    print("COMPARISON")
    print(f"{'=' * 74}")
    print(f"  A: {primary['path']}")
    print(f"  B: {baseline['path']}")
    print("  Positive 'delta' means B is larger than A.\n")

    def mean_of(result, key):
        values = [e[key] for e in result["join_detail"] if e.get(key) is not None]
        return float(np.mean(values)) if values else None

    print(f"  {'metric':24s} {'A':>10} {'B':>10} {'delta':>10}")
    for label, key in (("join detail step", "detail_step"),
                       ("join p10 step", "p10_step"),
                       ("join rms step dB", "rms_step_db"),
                       ("join HF step", "hf_step"),
                       ("join centroid step", "centroid_step_hz"),
                       ("join click score", "click_score")):
        a, b = mean_of(primary, key), mean_of(baseline, key)
        if a is None or b is None:
            continue
        print(f"  {label:24s} {a:10.4f} {b:10.4f} {b - a:+10.4f}")

    for label, key in (("chain detail slope", "detail_slope"),
                       ("chain p10 slope", "p10_slope"),
                       ("chain HF slope", "hf_ratio_slope"),
                       ("chain centroid slope", "centroid_hz_slope")):
        a, b = primary["chain"].get(key), baseline["chain"].get(key)
        if a is None or b is None:
            continue
        print(f"  {label:24s} {a:10.4f} {b:10.4f} {b - a:+10.4f}")

    ratios = []
    for result in (primary, baseline):
        values = [e["continuity"]["ratio"] for e in result["join_detail"]
                  if e.get("continuity")]
        ratios.append(float(np.mean(values)) if values else None)
    if all(r is not None for r in ratios):
        print(f"  {'continuity ratio':24s} {ratios[0]:10.4f} {ratios[1]:10.4f} "
              f"{ratios[1] - ratios[0]:+10.4f}")
        print("\n  Continuity nearest 1.00 is the better-aligned run. A run that "
              "sits\n  clearly below the other is repeating an instant at every join, "
              "which\n  is what a history block placed too early produces.")


def main():
    parser = argparse.ArgumentParser(
        description="Window-join statistics for Wan2GP H3 sliding-window renders.")
    parser.add_argument("video")
    parser.add_argument("--baseline", help="second render to compare against")
    parser.add_argument("--window", type=int, help="sliding window size in frames")
    parser.add_argument("--overlap", type=int, default=18, help="sliding window overlap")
    parser.add_argument("--joins", help="explicit join frames, comma separated")
    parser.add_argument("--auto-joins", action="store_true",
                        help="locate joins from the picture instead of the schedule")
    parser.add_argument("--fps", type=float, help="override detected fps")
    parser.add_argument("--scale", type=int, default=384,
                        help="analysis width in pixels (default 384)")
    parser.add_argument("--span", type=int, default=8,
                        help="frames averaged each side of a join (default 8)")
    parser.add_argument("--audio-window", type=float, default=0.5,
                        help="seconds of audio each side of a join (default 0.5)")
    parser.add_argument("--json", help="write full results to this path")
    args = parser.parse_args()

    _require_tools()

    primary = analyse(args.video, args)
    report(primary, args)

    baseline = None
    if args.baseline:
        baseline = analyse(args.baseline, args)
        report(baseline, args)
        compare(primary, baseline)

    if args.json:
        payload = {"primary": primary}
        if baseline:
            payload["baseline"] = baseline
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\n  wrote {args.json}")
    print()


if __name__ == "__main__":
    sys.exit(main())
