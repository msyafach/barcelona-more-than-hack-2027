"""Per-frame features from the tracking, in metres and seconds."""

import numpy as np

from .io import GAME, GAMES

FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]


def interpolate_gaps(values, max_gap):
    """Linear interpolation of NaN runs no longer than max_gap frames, per column."""
    out = values.copy()
    n = len(values)
    for col in range(values.shape[1]):
        v = out[:, col]
        ok = np.isfinite(v)
        if ok.sum() < 2:
            continue
        idx = np.arange(n)
        filled = np.interp(idx, idx[ok], v[ok])
        # length of the NaN run each frame belongs to
        run_id = np.cumsum(ok)
        gap = ~ok
        run_len = np.bincount(run_id[gap], minlength=run_id.max() + 1)
        inner = gap & (run_id > 0) & (run_id < run_id.max())  # bounded on both sides
        short = inner & (run_len[run_id] <= max_gap)
        v[short] = filled[short]
    return out


def centred_mean(x, half):
    """Centred moving mean ignoring NaN; NaN where the window has no data."""
    ok = np.isfinite(x)
    s = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    c = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    cnt = c[hi] - c[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (s[hi] - s[lo]) / cnt, np.nan)


def compute(tracking, ball_gap=12, extra_ball=None):
    """dict of per-frame arrays. Ball gaps up to `ball_gap` frames are linearly bridged.

    extra_ball: optional (n, 2) ball from another source on this timeline (the aligned
    broadcast), used only on frames where this source has no ball.
    """
    players = tracking["players"].astype(np.float64)
    team = tracking["team"]
    ball_raw = tracking["ball"][:, :2].astype(np.float64)
    if extra_ball is not None:
        fill = ~np.isfinite(ball_raw[:, 0]) & np.isfinite(extra_ball[:, 0])
        ball_raw[fill] = extra_ball[fill]
    ball_seen = np.isfinite(ball_raw[:, 0])
    ball = interpolate_gaps(ball_raw, ball_gap)
    ball_ok = np.isfinite(ball[:, 0])

    bx, by = ball[:, 0] * LENGTH, ball[:, 1] * WIDTH
    px, py = players[:, :, 0] * LENGTH, players[:, :, 1] * WIDTH
    dist = np.hypot(px - bx[:, None], py - by[:, None])  # NaN where either is missing

    feats = {"ball_seen": ball_seen, "ball_ok": ball_ok,
             "ball_x": ball[:, 0], "ball_y": ball[:, 1]}
    for code, name in ((0, "H"), (1, "A")):
        d = np.where(team[None, :] == code, dist, np.nan)
        with np.errstate(all="ignore"):
            feats["d" + name] = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1)
        feats["d" + name][~np.isfinite(feats["d" + name])] = np.nan
        feats["n" + name] = np.sum(np.isfinite(px) & (team[None, :] == code), axis=1)
        feats["cx" + name] = _nanmean(px, team == code)
        feats["cy" + name] = _nanmean(py, team == code)

    # Ball speed in m/s from the bridged track, lightly smoothed.
    vx = np.gradient(bx) * FPS
    vy = np.gradient(by) * FPS
    feats["ball_speed"] = centred_mean(np.hypot(vx, vy), 2)

    # Collective activity: mean player speed (provided per slot, m/s).
    speeds = players[:, :, 2]
    with np.errstate(all="ignore"):
        feats["player_speed"] = np.nanmean(np.where(np.isfinite(speeds), speeds, np.nan), axis=1)
    feats["n_players"] = np.sum(np.isfinite(px), axis=1)

    # Out-of-bounds: ball beyond the lines by a small margin (metres).
    margin = 0.5
    feats["ball_out"] = ball_ok & ((bx < -margin) | (bx > LENGTH + margin)
                                   | (by < -margin) | (by > WIDTH + margin))
    return feats


def _nanmean(values, mask):
    sub = values[:, mask]
    with np.errstate(all="ignore"):
        return np.nanmean(sub, axis=1)
