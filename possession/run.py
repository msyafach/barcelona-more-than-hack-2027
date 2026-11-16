"""Regenerate the submission from the provided data.

    python -m possession.run                      # -> possession.csv
    python -m possession.run --out <team_name>.csv
"""

import argparse
import os

import numpy as np

from . import features, model, shape
from .io import GAME, ROOT, attack_direction_first_half, load_smart_tagging, load_tracking
from games import half_video_blocks, require


def build(params=None, use_team_priors=False, use_tagging=True, use_shape=True,
          use_broadcast=False, report=None):
    """Two passes: the first gives pseudo-labels for the self-supervised shape model, the second
    decodes with shape evidence added in the frames the ball does not inform."""
    tracking = load_tracking()
    extra = None
    if use_broadcast:
        from .experimental import build_broadcast_ball
        extra = build_broadcast_ball.load()
    f = features.compute(tracking, extra_ball=extra)
    events = load_smart_tagging() if use_tagging else ()
    halves = half_video_blocks(GAME)
    first = model.label(f, halves, events, params, use_team_priors)
    if not use_shape:
        return first
    p = dict(model.PARAMS, **(params or {}))
    cH, cA = model.control_evidence(f, p)
    home_dir = attack_direction_first_half(require(["atd_meta"])[0])[0]
    logit = shape.gap_evidence(tracking, f, halves, home_dir, cH, cA, np.isin(first, ["H", "A"]),
                               report=report)
    return model.label(f, halves, events, params, use_team_priors, shape_logit=logit)


def write(path, labels):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


def summary(labels, halves):
    """Possession count and mean duration over graded frames; needs no ground truth."""
    spells, frames = 0, 0
    for _, first, last in halves:
        seg = labels[first:last + 1]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate(([0], change))
        for s in starts:
            if seg[s] in ("H", "A"):
                spells += 1
        frames += np.isin(seg, ["H", "A"]).sum()
    return spells, frames / max(spells, 1) / 25.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "possession.csv"),
                        help="output CSV (default: possession.csv in the solution folder)")
    parser.add_argument("--team-priors", action="store_true",
                        help="ablation: add Smart Tagging phase-team priors (off in submission)")
    parser.add_argument("--no-shape", action="store_true", help="ablation: drop team-shape evidence")
    parser.add_argument("--broadcast", action="store_true",
                        help="experimental: fill the ball from the aligned broadcast (off in submission)")
    args = parser.parse_args()

    labels = build(use_team_priors=args.team_priors, use_shape=not args.no_shape,
                   use_broadcast=args.broadcast)
    path = args.out
    write(path, labels)
    n, dur = summary(labels, half_video_blocks(GAME))
    counts = {c: int((labels == c).sum()) for c in "HADX"}
    print("wrote", path, counts)
    print("full match: {} possessions, mean {:.1f} s".format(n, dur))
