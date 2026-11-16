"""Align the broadcast ATD to the tactical timeline and cache its ball on tactical frames.

    python -m possession.build_broadcast_ball
"""

import os
import time

import numpy as np

from . import align
from .io import cache_dir, load_tracking

PATH = os.path.join(cache_dir(), "BAR-ATH_broadcast_ball_on_tactical.npz")


def build():
    t = load_tracking()
    b = load_tracking(source="broadcast")
    n = len(t["players"])
    t0 = time.time()
    offsets, windows = align.align(b, t, n)
    costs = align.frame_costs(b, t, offsets, n)
    ball = align.ball_on_tactical(b, offsets, costs, n)
    np.savez(PATH, ball=ball, offsets=offsets, costs=costs, windows=np.array(windows))
    print("aligned in {:.0f} s".format(time.time() - t0))
    return ball


def load():
    if not os.path.isfile(PATH):
        return build()
    return np.load(PATH)["ball"]


if __name__ == "__main__":
    build()
