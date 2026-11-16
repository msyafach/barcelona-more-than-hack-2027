"""Map the broadcast tracking onto the tactical timeline, then use its ball where tactical has none.

Both recordings report players in the same normalised pitch frame, so a broadcast frame and the
tactical frame showing the same instant must put every visible player of each team next to a
tracked player of that team. The broadcast camera sees only part of the pitch, so the cost is
one-sided: for each broadcast player, the distance to the nearest same-team tactical player.

Alignment is done per window of broadcast frames, because the TV feed cuts and replays: within a
window the offset is constant, between windows it can jump. Windows are tracked (search near the
previous offset first) and re-acquired with a global search when the local fit is poor. A replay
legitimately maps back to the earlier instant it shows, the positions really are those of that
instant, but it is only ever used to fill frames tactical cannot see.
"""

import numpy as np

from .features import LENGTH, WIDTH

STEP = 5             # broadcast frames between samples inside a window
WIN = 25             # samples per window (125 frames = 5 s)
MIN_PLAYERS = 6      # broadcast players needed in a sample for it to count
LOCAL = 125          # local search, +/- frames around the previous offset
GOOD = 1.2           # m: median one-sided NN distance accepted as a match
FRAME_GOOD = 1.5
BACK = 3000          # frames a replay may map back
AHEAD = 30000        # frames the feed may skip forward (halftime)     # m: per-frame check before a broadcast ball is used


def _xy(players):
    return np.stack([players[..., 0] * LENGTH, players[..., 1] * WIDTH], axis=-1)


def window_cost(bxy, bteam_mask, txy, tteam, t_frames):
    """Median one-sided NN distance for one broadcast sample against many tactical frames.

    bxy: (k, 2) visible broadcast players; bteam_mask: (k,) team codes.
    txy: (n_t, slots, 2) tactical positions; t_frames: candidate tactical frames (m,)
    Returns (m,) cost.
    """
    T = txy[t_frames]  # (m, slots, 2)
    d = np.linalg.norm(T[:, None, :, :] - bxy[None, :, None, :], axis=-1)  # (m, k, slots)
    same = (tteam[None, None, :] == bteam_mask[None, :, None])
    d = np.where(same & np.isfinite(d), d, np.inf)
    nn = d.min(axis=2)  # (m, k)
    nn = np.where(np.isfinite(nn), nn, 50.0)
    return np.median(nn, axis=1)


def align(broadcast, tactical, n_tactical):
    """(offset per broadcast frame or NaN, cost per broadcast frame). tactical = b + offset."""
    bxy = _xy(broadcast["players"].astype(np.float64))
    txy = _xy(tactical["players"].astype(np.float64))
    bteam, tteam = broadcast["team"], tactical["team"]
    nb = len(bxy)
    visible = np.isfinite(bxy[..., 0])
    offsets = np.full(nb, np.nan)
    window_costs = []
    prev = None
    span = STEP * WIN
    for w0 in range(0, nb - span, span):
        samples = [b for b in range(w0, w0 + span, STEP) if visible[b].sum() >= MIN_PLAYERS]
        if len(samples) < WIN // 3:
            continue

        def cost_for(cands):
            total = np.zeros(len(cands))
            for b in samples:
                v = visible[b]
                t = np.clip(b + cands, 0, n_tactical - 1)
                total += window_cost(bxy[b][v], bteam[v], txy, tteam, t)
            return total / len(samples)

        best = None
        if prev is not None:
            cands = np.arange(prev - LOCAL, prev + LOCAL + 1)
            c = cost_for(cands)
            i = int(np.argmin(c))
            if c[i] < GOOD:
                best = (cands[i], c[i])
        if best is None:
            # The TV feed runs forward with the match: a replay maps back at most a couple of
            # minutes, halftime can skip ~15 minutes ahead. Without a previous lock, search all.
            if prev is None:
                cands = np.arange(-w0, n_tactical - span - w0, 10)
            else:
                cands = np.arange(prev - BACK, prev + AHEAD, 10)
            cands = cands[(w0 + cands >= 0) & (w0 + cands < n_tactical - span)]
            if len(cands) == 0:
                continue
            # coarse on a subset of samples for speed, then refine
            sub = samples[::max(1, len(samples) // 5)]
            total = np.zeros(len(cands))
            for b in sub:
                v = visible[b]
                total += window_cost(bxy[b][v], bteam[v], txy, tteam,
                                     np.clip(b + cands, 0, n_tactical - 1))
            coarse = cands[np.argsort(total)[:5]]
            fine = np.unique(np.concatenate([np.arange(c - 10, c + 11) for c in coarse]))
            c = cost_for(fine)
            i = int(np.argmin(c))
            best = (fine[i], c[i])
        window_costs.append((w0, best[0], best[1]))
        if best[1] < GOOD:
            offsets[w0:w0 + span] = best[0]
            prev = best[0]
    return offsets, window_costs


def frame_costs(broadcast, tactical, offsets, n_tactical):
    """Per broadcast frame one-sided NN cost at its assigned offset (NaN where unassigned)."""
    bxy = _xy(broadcast["players"].astype(np.float64))
    txy = _xy(tactical["players"].astype(np.float64))
    out = np.full(len(bxy), np.nan)
    for b in np.flatnonzero(np.isfinite(offsets)):
        v = np.isfinite(bxy[b, :, 0])
        if v.sum() < 3:
            continue
        t = int(b + offsets[b])
        if not 0 <= t < n_tactical:
            continue
        out[b] = window_cost(bxy[b][v], broadcast["team"][v], txy, tactical["team"],
                             np.array([t]))[0]
    return out


def ball_on_tactical(broadcast, offsets, costs, n_tactical):
    """(n_tactical, 2) broadcast ball mapped to tactical frames, NaN where not trustworthy."""
    out = np.full((n_tactical, 2), np.nan)
    ok = (np.isfinite(offsets) & np.isfinite(costs) & (costs < FRAME_GOOD)
          & np.isfinite(broadcast["ball"][:, 0]))
    b = np.flatnonzero(ok)
    t = (b + offsets[b]).astype(np.int64)
    keep = (t >= 0) & (t < n_tactical)
    out[t[keep]] = broadcast["ball"][b[keep], :2]
    return out
