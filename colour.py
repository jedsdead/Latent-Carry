"""Colour correction applied to carried latents instead of to decoded pixels.

Why in latent space
-------------------
Correcting decoded pixels and carrying latents are incompatible. The
correction happens after `decode`; the latents are captured during it. So a
pixel correction never reaches the conditioning, the next window is
conditioned on content that no longer matches the video it continues, and
the feedback loop that made pixel correction converge - stock Wan2GP slices
`pre_video_guide` out of the *corrected* sample and re-encodes it - is
exactly the loop latent carry removes.

Correcting the latents closes it again, and costs one 24x24 matrix multiply
per window instead of a VAE round trip.

How
---
Wan2GP ships a linear latent-to-RGB map for H3 previews, in
`shared/RGB_factors.py`. Signed RGB is `latent @ M + bias`, with M 24x3. It
is surjective onto RGB and well conditioned (singular values 0.847, 0.381,
0.126; condition number 6.7), so `pinv(M)` always finds a latent delta that
produces a requested RGB delta.

Because the map is linear, an affine colour correction in RGB collapses to
an affine in latent space. With `s` the signed RGB of a latent, a correction
`s' = Q s + 2o` gives

    L' = (I + P^T D M^T) L + P^T [D (bias + 1) + 2 o]        D = Q - I

- a single 24x24 matrix and a 24-vector, built once per window and applied
with one einsum. The same collapse the pixel version does from Y'CbCr into a
3x3, one step further down.

Measurement is in Rec. 709 Y'CbCr, so the four axes stay separable: the luma
mean is brightness, its spread is contrast, the chroma means are colour cast
and their shared spread is saturation. One scale for both chroma channels,
not one each, because scaling them separately shifts hue while claiming to
change only saturation.

What this does not do
---------------------
It does not touch the saved video. This is the latent equivalent of anchor
-only scope: it stops drift compounding into later windows without altering
frames already written. Whole-window pixel correction remains the only thing
that fixes what is already on disk, and it is not compatible with carrying -
see the module docstring above.

The linear map is a preview approximation of the real decoder, not the
decoder. Offsets - brightness and cast - are on firm ground. Gains -
contrast and saturation - are multiplicative and lean on the approximation
harder, so they are off by default. Measure before trusting them.
"""

import numpy as np

# Fallback copy of shared/RGB_factors.py, "minimax_h3", so the maths is
# testable without Wan2GP present. load_factors() prefers the live values.
_EMBEDDED_M = np.array([
    (-0.152104, -0.232543, -0.224404), (0.139305, 0.183498, 0.102793),
    (0.200355, 0.102748, -0.014897), (0.205527, 0.060373, -0.223634),
    (0.074193, 0.143338, -0.056514), (0.118000, 0.103517, -0.028515),
    (0.087893, 0.075594, 0.083708), (0.082730, 0.091851, 0.066146),
    (0.046948, 0.062517, 0.070342), (0.372102, 0.297104, 0.314785),
    (-0.119610, -0.122069, -0.075437), (-0.091787, -0.068970, -0.008514),
    (0.032219, 0.024462, 0.053078), (-0.000277, 0.025318, 0.015307),
    (0.029826, 0.048534, 0.071107), (-0.028803, -0.041618, -0.063091),
    (0.008493, 0.006525, -0.003508), (-0.045912, -0.048118, -0.039270),
    (-0.039653, -0.045122, -0.054826), (0.090955, 0.084697, 0.097391),
    (0.007149, 0.005895, 0.005149), (-0.013688, -0.014178, -0.019584),
    (0.001140, 0.019586, 0.020279), (-0.018252, -0.017087, -0.007325),
], dtype=np.float64)
_EMBEDDED_BIAS = np.array([0.167441, 0.103166, 0.056984], dtype=np.float64)

# Rec. 709 luma, with Cb/Cr centred on zero for unit-range RGB.
_KR, _KB = 0.2126, 0.0722
_KG = 1.0 - _KR - _KB
RGB_TO_YCBCR = np.array([
    [_KR, _KG, _KB],
    [-_KR / (2 * (1 - _KB)), -_KG / (2 * (1 - _KB)), 0.5],
    [0.5, -_KG / (2 * (1 - _KR)), -_KB / (2 * (1 - _KR))],
], dtype=np.float64)
YCBCR_TO_RGB = np.linalg.inv(RGB_TO_YCBCR)

AXES = ("brightness", "contrast", "saturation", "cast")


def load_factors():
    """The live latent-to-RGB map, or the embedded copy if Wan2GP is absent.

    Returns (M, bias, source). A shape change upstream is reported rather
    than absorbed: a differently sized map would silently mean something
    else.
    """
    try:
        from shared.RGB_factors import get_rgb_factors
        factors, bias = get_rgb_factors("minimax_h3")
        matrix = np.asarray(factors, dtype=np.float64)
        offset = np.asarray(bias, dtype=np.float64)
        if matrix.shape != _EMBEDDED_M.shape or offset.shape != (3,):
            return _EMBEDDED_M, _EMBEDDED_BIAS, (
                f"embedded (upstream map is {matrix.shape}, expected "
                f"{_EMBEDDED_M.shape})")
        return matrix, offset, "wan2gp"
    except Exception as error:
        return _EMBEDDED_M, _EMBEDDED_BIAS, f"embedded ({type(error).__name__})"


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def ycbcr_stats(rgb01):
    """Per-channel mean and standard deviation in Y'CbCr.

    `rgb01` is (..., 3) in [0, 1]. Returned as plain float arrays so a
    window's statistics can be kept after its frames are gone - which is the
    only thing that survives an ordinary window join.
    """
    flat = np.asarray(rgb01, dtype=np.float64).reshape(-1, 3)
    ycc = flat @ RGB_TO_YCBCR.T
    return {"mean": ycc.mean(axis=0), "std": ycc.std(axis=0),
            "count": int(flat.shape[0])}


def _shrink(observed, noise):
    """Hold back a measurement that is not clearly larger than the wobble.

    Real footage moves, and on dark or low-chroma material that movement is
    often larger than the drift being looked for. Correcting it anyway does
    not average out over a long generation - each window inherits the last,
    so chasing noise accumulates into exactly the drift the correction
    exists to prevent.
    """
    magnitude = np.abs(observed)
    floor = np.abs(noise)
    scale = np.zeros_like(magnitude)
    live = magnitude > floor
    scale[live] = 1.0 - (floor[live] / magnitude[live])
    return observed * scale


def measure(reference, current, noise=None, strength=1.0,
            scene_threshold=0.06, max_correction=0.15, axes=AXES):
    """Y'CbCr gain and offset that would pull `current` onto `reference`.

    Both are ycbcr_stats dicts. Returns (gain3, offset3, report); gain and
    offset are None when the measurement was rejected.
    """
    ref_mean, cur_mean = reference["mean"], current["mean"]
    ref_std, cur_std = reference["std"], current["std"]
    report = {"rejected": None}

    luma_step = float(abs(ref_mean[0] - cur_mean[0]))
    chroma_step = float(np.abs(ref_mean[1:] - cur_mean[1:]).max())
    report["luma_step"], report["chroma_step"] = luma_step, chroma_step
    if max(luma_step, chroma_step) > scene_threshold:
        # Drift is around a percent per window. Something several times
        # larger is not severe drift, it is evidence the two sides are not
        # showing the same moment. Discard rather than cap: capping and
        # applying drags a new scene bodily toward the grade of the old one.
        report["rejected"] = (f"scene change (luma {luma_step:.4f}, chroma "
                              f"{chroma_step:.4f} > {scene_threshold})")
        return None, None, report

    safe_std = np.where(cur_std < 1e-6, 1e-6, cur_std)
    raw_gain = ref_std / safe_std
    raw_gain[1] = raw_gain[2] = np.sqrt(max(raw_gain[1] * raw_gain[2], 1e-12))
    raw_offset = ref_mean - raw_gain * cur_mean

    if not {"contrast"} & set(axes):
        raw_gain[0] = 1.0
    if not {"saturation"} & set(axes):
        raw_gain[1] = raw_gain[2] = 1.0
    if not {"brightness"} & set(axes):
        raw_offset[0] = 0.0
    if not {"cast"} & set(axes):
        raw_offset[1] = raw_offset[2] = 0.0

    gain_error, offset_error = raw_gain - 1.0, raw_offset
    if noise is not None:
        noise_gain = np.abs(noise["std"] / np.where(reference["std"] < 1e-6,
                                                    1e-6, reference["std"]))
        gain_error = _shrink(gain_error, noise_gain)
        offset_error = _shrink(offset_error, np.abs(noise["mean"]))
        report["noise"] = {"gain": noise_gain.tolist(),
                           "offset": np.abs(noise["mean"]).tolist()}

    gain_error *= float(strength)
    offset_error *= float(strength)
    gain_error = np.clip(gain_error, -max_correction, max_correction)
    offset_error = np.clip(offset_error, -max_correction, max_correction)

    gain, offset = 1.0 + gain_error, offset_error
    report["gain"], report["offset"] = gain.tolist(), offset.tolist()
    report["is_noop"] = bool(np.allclose(gain, 1.0) and np.allclose(offset, 0.0))
    return gain, offset, report


# --------------------------------------------------------------------------
# the collapse into latent space
# --------------------------------------------------------------------------

def within_window_change(head, tail, scene_threshold=0.06):
    """Is a window uniform enough for a head measurement to describe its tail?

    The correction is measured between the reference - the previous window's
    tail - and this window's opening frames, because those are the two things
    that are supposed to be continuous. But in `latents` scope it is applied to
    this window's *closing* latents, and in `both` scope to the whole window.

    So a cut part way through a window puts the measurement and its target on
    opposite sides of it. Everything after the cut belongs to a scene the
    reference never saw and has nothing valid to be corrected toward, and the
    carried block is entirely on that side. Refuse rather than apply: unlike a
    pixel correction there is no partial version available, since the unit
    being corrected is the single block of tail latents.

    Comparing the window's own head against its own tail is enough. A cut that
    does not change the grade needs no guarding, and one that does shows up
    here. A slow deliberate lighting change will not trip it - drift runs
    around a percent per window against a 0.06 default.

    Returns (ok, detail).
    """
    luma = float(abs(head["mean"][0] - tail["mean"][0]))
    chroma = float(np.abs(np.asarray(head["mean"][1:]) - np.asarray(tail["mean"][1:])).max())
    if max(luma, chroma) > scene_threshold:
        return False, (f"window is not uniform (head to tail luma {luma:.4f}, "
                       f"chroma {chroma:.4f} > {scene_threshold}); a cut inside "
                       f"the window puts the measurement and the latents it "
                       f"would correct on opposite sides of it")
    return True, None


def compose(total, new, max_gain=0.25, max_offset=0.25):
    """Accumulate a correction onto the running total, in Y'CbCr.

    Applying `new` after `total` gives `g = g_total * g_new` and
    `o = g_new * o_total + o_new`.

    This matters more than it looks. Correcting the carried latents does not
    rewrite the video, so the reference a window is matched against - the
    previous window's tail - is itself uncorrected and keeps moving. A
    per-window correction therefore cancels one window's drift, measures
    none the next, and lets the following one through: the drift comes back
    at half rate in a visible stair-step. Carrying the total forward holds
    the grade instead.

    The total is clamped, not the step, since it is the total that reaches
    the model.
    """
    gain_total, offset_total = (np.ones(3), np.zeros(3)) if total is None else total
    gain_new, offset_new = np.asarray(new[0], dtype=np.float64), np.asarray(new[1], dtype=np.float64)
    gain = np.asarray(gain_total, dtype=np.float64) * gain_new
    offset = gain_new * np.asarray(offset_total, dtype=np.float64) + offset_new
    gain = np.clip(gain, 1.0 - max_gain, 1.0 + max_gain)
    offset = np.clip(offset, -max_offset, max_offset)
    return gain, offset


def rgb_affine(gain, offset):
    """The Y'CbCr gain/offset as a 3x3 matrix and 3-vector acting on RGB.

    Y'CbCr is a linear transform of RGB, so the whole correction is one
    matrix - the same collapse the pixel implementation makes.
    """
    Q = YCBCR_TO_RGB @ np.diag(np.asarray(gain, dtype=np.float64)) @ RGB_TO_YCBCR
    o = YCBCR_TO_RGB @ np.asarray(offset, dtype=np.float64)
    return Q, o


def latent_correction(gain, offset, matrix=None, bias=None):
    """Build (A, b) so that `L @ A.T + b` carries the RGB correction.

    Derivation, with s the signed RGB of a latent and D = Q - I:

        s      = M^T L + bias
        s'     = Q s + 2 o          (offsets double: unit -> signed)
        L'     = L + P^T (s' - s)
               = (I + P^T D M^T) L + P^T [D (bias + 1) + 2 o]
    """
    if matrix is None or bias is None:
        loaded_m, loaded_b, _ = load_factors()
        matrix = loaded_m if matrix is None else matrix
        bias = loaded_b if bias is None else bias
    M = np.asarray(matrix, dtype=np.float64)          # 24 x 3
    c = np.asarray(bias, dtype=np.float64)            # 3
    P = np.linalg.pinv(M)                             # 3 x 24

    Q, o = rgb_affine(gain, offset)
    D = Q - np.eye(3)
    A = np.eye(M.shape[0]) + P.T @ D @ M.T
    b = P.T @ (D @ (c + 1.0) + 2.0 * o)
    return A, b


def latent_to_signed_rgb(latents, matrix=None, bias=None):
    """Signed RGB estimate of a latent tensor, without decoding it.

    `latents` is (..., C) on the last axis. Used for measurement when no
    decoded frames are to hand, and by the tests.
    """
    if matrix is None or bias is None:
        loaded_m, loaded_b, _ = load_factors()
        matrix = loaded_m if matrix is None else matrix
        bias = loaded_b if bias is None else bias
    return np.asarray(latents, dtype=np.float64) @ np.asarray(matrix) + np.asarray(bias)


def apply_numpy(latents, A, b):
    """Reference implementation: `latents` is (..., C) on the last axis."""
    return np.asarray(latents, dtype=np.float64) @ np.asarray(A).T + np.asarray(b)


def apply_torch(latents, A, b):
    """Apply (A, b) to a [B, C, T, H, W] latent tensor.

    One einsum over 24 channels; the spatial and temporal extents are
    untouched, so cost does not scale with resolution the way a VAE pass
    does.
    """
    import torch
    matrix = torch.as_tensor(A, dtype=latents.dtype, device=latents.device)
    shift = torch.as_tensor(b, dtype=latents.dtype, device=latents.device)
    out = torch.einsum("dc,bcthw->bdthw", matrix, latents)
    return out + shift.view(1, -1, 1, 1, 1)
