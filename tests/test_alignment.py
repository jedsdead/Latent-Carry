#!/usr/bin/env python3
"""Tests for the join-alignment search.

Numpy only - no Wan2GP, no torch, no GPU. `_check_alignment` is reimplemented
here against the same constants, because the real one takes torch tensors and
the logic under test is the search, not the tensor plumbing.

THE FAILURE BEING GUARDED
-------------------------
The next window hands back `continuation[:, -overlap:-1]`, so its last frame is
the previous window's decoded frame -2. If the cached latents end somewhere
else, the carried block gets placed as though it ended at the join, and the
model is handed footage from *after* the join as though it came before - and
renders it again. That is a repeated shot, not a stutter.

The old check compared one fingerprint at one offset against a 3% tolerance.
The tests below show why that cannot work: on a slow shot, frames several
apart differ by well under 3%, so a displaced tail passes and nothing reports
it. Searching the run finds where the join actually sits and reports the skew.

    python tests/test_alignment.py
"""

import sys

import numpy as np

ALIGN_EXPECTED_OFFSET = -2
ALIGN_DECISIVE = 0.6
MATCH_TOLERANCE = 0.03

FAILURES = []
rng = np.random.default_rng(5)


def check(label, condition, detail=""):
    print(f"  [{'pass' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def check_alignment(signatures, incoming, align_search=True,
                    tolerance=MATCH_TOLERANCE):
    """Mirror of patches._check_alignment for the searching path."""
    errors = np.abs(signatures - incoming[None]).mean(axis=(1, 2))
    depth = signatures.shape[0]
    expected_row = depth + ALIGN_EXPECTED_OFFSET
    expected_error = float(errors[expected_row])
    best_row = int(errors.argmin())
    best_error = float(errors[best_row])
    skew = best_row - expected_row

    if align_search and skew != 0 and best_error < expected_error * ALIGN_DECISIVE:
        return False, skew
    if expected_error > tolerance:
        return False, None
    return True, 0


def tail(frames, drift, start=0.5, noise=0.0):
    """A decoded tail as 16x16 fingerprints, each frame drifting by `drift`."""
    base = rng.normal(0.0, 0.08, (16, 16))
    out = []
    for i in range(frames):
        f = np.clip(start + base + drift * i, 0.0, 1.0)
        if noise:
            f = np.clip(f + rng.normal(0.0, noise, (16, 16)), 0.0, 1.0)
        out.append(f)
    return np.stack(out)


print(__doc__.strip().splitlines()[0])
print()

DEPTH = 12

# ------------------------------------------------------- correct alignment ----
print("a correctly aligned tail")
for label, drift in (("fast motion", 0.02), ("slow shot", 0.001), ("static shot", 0.0)):
    frames = tail(DEPTH, drift)
    incoming = frames[DEPTH + ALIGN_EXPECTED_OFFSET]      # the real join frame
    ok, skew = check_alignment(frames, incoming)
    check(f"{label:14s} accepted with zero skew", ok and skew == 0, f"skew {skew}")
print()

# ------------------------------------------------------ displaced tails -------
print("a displaced tail is detected and the skew reported")
for label, drift in (("fast motion", 0.02), ("slow shot", 0.001)):
    frames = tail(DEPTH, drift)
    # Only +1 is reachable upward: the cached fingerprints stop at offset -1.
    # A cache holding frames from after the join shows as a NEGATIVE skew, and
    # that is the direction that causes a repeat.
    for planted in (+1, -1, -3, -6):
        row = DEPTH + ALIGN_EXPECTED_OFFSET + planted
        assert 0 <= row < DEPTH, planted
        incoming = frames[row]
        ok, skew = check_alignment(frames, incoming)
        check(f"{label:14s} skew {planted:+d} rejected and measured",
              (not ok) and skew == planted, f"reported {skew}")
print()

# ------------------------------------- why one offset was not enough ----------
print("the single-offset check these replace")
frames = tail(DEPTH, 0.001)                    # slow shot
row = DEPTH + ALIGN_EXPECTED_OFFSET - 4        # cache holds 4 frames past the join
incoming = frames[row]
single = float(np.abs(frames[DEPTH + ALIGN_EXPECTED_OFFSET] - incoming).mean())
check("a 4-frame overrun slips under the 3% tolerance",
      single <= MATCH_TOLERANCE,
      f"difference {single:.5f} <= {MATCH_TOLERANCE}")
ok, skew = check_alignment(frames, incoming)
check("the search catches the same case", (not ok) and skew == -4, f"skew {skew}")

ok_disabled, _ = check_alignment(frames, incoming, align_search=False)
check("H3LC_ALIGN_SEARCH=0 restores the old permissive behaviour", ok_disabled)
print()

# ---------------------------------------------- a genuinely different video ---
print("a different video is still rejected")
frames = tail(DEPTH, 0.002)
other = np.clip(rng.normal(0.5, 0.25, (16, 16)), 0.0, 1.0)
ok, _ = check_alignment(frames, other)
check("unrelated content rejected", not ok)
print()

# ------------------------------------------------- static shot ambiguity ------
print("a static shot stays usable rather than being refused on noise")
frames = tail(DEPTH, 0.0, noise=0.002)
incoming = frames[DEPTH + ALIGN_EXPECTED_OFFSET]
accepted = sum(check_alignment(tail(DEPTH, 0.0, noise=0.002),
                               tail(DEPTH, 0.0, noise=0.002)[DEPTH - 2])[0]
               for _ in range(1))
ok, skew = check_alignment(frames, incoming)
check("the true join is accepted despite a flat, noisy tail", ok, f"skew {skew}")
print("      (on a flat tail no offset beats expected by the decisive margin,")
print("       so the search abstains and the tolerance decides, as before)")
print()

if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
