"""Per-frame H/A/D labels from tracking evidence, decoded with a 3-state Viterbi.

The idea, in football terms
---------------------------
* Possession changes only when the other team gains *control* of the ball: one of its players is
  close to a slow-enough ball, clearly closer than any opponent, for a sustained run of frames.
  A touch, a deflection or a ball flying past a player is not enough evidence to pay the switch
  cost. While the ball is loose or invisible there is no evidence either way, so the decoder keeps
  the last team, exactly the "possession stays with the last team that had it" rule.
* The ball is dead when it has left the pitch, when play visibly stops (the whole field walks and
  the ball sits still or is not seen), and in the run-up to every restart the automatic Smart
  Tagging flags. Restarts end dead stretches.
* Entering or leaving a dead stretch, or swapping teams, each costs a fixed penalty, so a label
  only changes when the accumulated evidence pays for it. That is what kills flicker.

All constants below are set from football reasoning (distances in metres, durations in seconds),
not fitted to any ground truth.
"""

import numpy as np

from .features import FPS, centred_mean

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])

PARAMS = dict(
    # control evidence
    control_radius=1.5,       # m: a player this close to the ball can be playing it
    control_margin=0.5,       # m: and must be this much closer than the nearest opponent
    control_max_ball_speed=10.0,  # m/s: faster than this the ball is travelling, not controlled
    control_weight=1.0,       # log-score per frame of control
    soft_control=True,
    soft_r1=1.0,              # m: full control evidence inside this distance
    soft_r2=3.0,              # m: none beyond this
    soft_margin=1.0,          # m: closer-than-opponent margin at which evidence saturates
    switch_cost=10.0,         # H<->A: ~10 frames (0.4 s) of net control by the other team
    # team-shape evidence (possession.shape), per frame of log-odds, used where control is silent.
    # Consecutive frames are far from independent, so it is heavily discounted: a switch inside a
    # ball gap needs several seconds of consistent shape.
    shape_weight=0.05,
    shape_clip=3.0,
    # dead-ball evidence
    walk_speed=1.4,           # m/s: mean player speed below this reads as a stoppage
    walk_scale=0.4,           # m/s per unit of evidence
    walk_weight=0.6,
    activity_window_s=1.0,
    pass_speed=6.0,           # m/s: a ball this fast within +/-0.5 s means someone played it
    out_weight=3.0,           # per frame the ball is beyond the lines
    out_hold_weight=1.0,      # per frame between the ball leaving and the restart
    out_hold_max_s=45.0,
    out_min_frames=3,         # ignore single-frame excursions (position jitter on the line)
    dead_cost=25.0,           # alive<->dead
    # Smart Tagging restarts: the restart happens ~3.4 s after a tag starts (both kickoffs)
    # (kickoffs: +3.4 s; the mean tracking profile around all set-piece tags, players slowest,
    # ball least visible, runs from ~-6 s to ~+4 s around the tag start)
    restart_offset_s=4.0,
    pre_restart_dead_s=8.0,   # the seconds before a restart are dead
    pre_restart_weight=1.0,
    post_restart_alive_s=1.0,
    post_restart_weight=1.5,
)


def control_evidence(f, p):
    """(ctrl_H, ctrl_A) per-frame evidence in [0, 1]."""
    if p.get("soft_control"):
        return soft_control_evidence(f, p)
    dH, dA = f["dH"], f["dA"]
    slow = np.nan_to_num(f["ball_speed"], nan=np.inf) < p["control_max_ball_speed"]
    ok = f["ball_ok"] & slow & ~f["ball_out"]
    dHn = np.nan_to_num(dH, nan=np.inf)
    dAn = np.nan_to_num(dA, nan=np.inf)
    cH = ok & (dHn < p["control_radius"]) & (dHn + p["control_margin"] < dAn)
    cA = ok & (dAn < p["control_radius"]) & (dAn + p["control_margin"] < dHn)
    return cH.astype(float), cA.astype(float)


def soft_control_evidence(f, p):
    """Graded version: full weight within soft_r1 of the ball, fading to 0 at soft_r2, scaled by
    how much closer the team's nearest player is than the opponent's (saturating at 1 m).

    Automatic ball positions lag and jitter by a metre or more, so a hard 1.5 m gate throws away
    most frames of a dribble and delays every detected change of possession.
    """
    dH = np.nan_to_num(f["dH"], nan=np.inf)
    dA = np.nan_to_num(f["dA"], nan=np.inf)
    dmin = np.minimum(dH, dA)
    close = np.clip((p["soft_r2"] - dmin) / (p["soft_r2"] - p["soft_r1"]), 0.0, 1.0)
    side = np.clip(np.nan_to_num(dA - dH, nan=0.0, posinf=1.0, neginf=-1.0)
                   / p["soft_margin"], -1.0, 1.0)
    speed = np.nan_to_num(f["ball_speed"], nan=np.inf)
    gate = np.where(speed < p["control_max_ball_speed"], 1.0, 0.0)
    ok = f["ball_ok"] & ~f["ball_out"]
    e = np.where(ok, close * side * gate, 0.0)
    return np.clip(e, 0, 1), np.clip(-e, 0, 1)


def dead_evidence(f, p, restarts=()):
    """Per-frame log-odds of dead vs alive (added to the D state's score)."""
    n = len(f["ball_ok"])
    half = int(round(p["activity_window_s"] * FPS))
    activity = centred_mean(f["player_speed"], half)
    walk = np.clip((p["walk_speed"] - np.nan_to_num(activity, nan=p["walk_speed"]))
                   / p["walk_scale"], -1.0, 1.0)
    fast = np.nan_to_num(f["ball_speed"], nan=0.0) > p["pass_speed"]
    ball_still = ~(centred_mean(fast.astype(float), int(FPS // 2)) > 0)
    # Slow players only count as a stoppage when the ball is not moving; a slow team passing the
    # ball around at the back is alive.
    walk = np.where((walk > 0) & ~ball_still, 0.0, walk)
    score = p["walk_weight"] * walk
    score = score + p["out_weight"] * f["ball_out"]
    score = score + p["out_hold_weight"] * out_hold(f, p)

    for frame in restarts:
        lo = max(0, frame - int(p["pre_restart_dead_s"] * FPS))
        hi = min(n, frame + int(p["post_restart_alive_s"] * FPS))
        score[lo:frame] += p["pre_restart_weight"]
        score[frame:hi] -= p["post_restart_weight"]
    return score


def out_hold(f, p):
    """1 from the moment the ball leaves the pitch until it is next played inside it.

    Once the ball is out it stays dead until a throw-in, corner, goal kick or kick-off, and the
    tracker often loses it in the meantime, so 'ball out' alone marks only the first frames of the
    stoppage. The restart is the first frame the ball is seen back inside the lines moving at pass
    speed. Capped, so one missed restart cannot swallow a minute of play.
    """
    out = f["ball_out"]
    n = len(out)
    inside_fast = (f["ball_seen"] & ~out
                   & (np.nan_to_num(f["ball_speed"], nan=0.0) > p["pass_speed"]))
    hold = np.zeros(n)
    cap = int(p["out_hold_max_s"] * FPS)
    minrun = p["out_min_frames"]
    idx = np.flatnonzero(out)
    if len(idx) == 0:
        return hold
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    next_fast = np.full(n + 1, n)
    fast_idx = np.flatnonzero(inside_fast)
    pos = np.searchsorted(fast_idx, np.arange(n))
    next_fast[:n] = np.where(pos < len(fast_idx), fast_idx[np.minimum(pos, len(fast_idx) - 1)], n)
    for r in runs:
        if len(r) < minrun:
            continue
        start, end = r[0], r[-1] + 1
        stop = min(next_fast[end] if end < n else n, start + cap)
        hold[start:stop] = 1.0
    return hold


def restart_frames(events, p):
    """Video frames of restarts implied by Smart Tagging set pieces."""
    off = int(round(p["restart_offset_s"] * FPS))
    return sorted(e["start"] + off for e in events if e["code"] == "SET PIECES")


def kickoff_tags(events):
    """Video frames where Smart Tagging opens a 'Kick Off (Start)' set piece, one per half."""
    return sorted(e["start"] for e in events
                  if e["code"] == "SET PIECES" and e["type"].startswith("kick off (start)"))


def viterbi(emis, switch_cost, dead_cost, start=None):
    """emis: (n, 3) log-scores. Returns best state path. start: forced first state or None."""
    n = len(emis)
    trans = np.array([[0.0, -switch_cost, -dead_cost],
                      [-switch_cost, 0.0, -dead_cost],
                      [-dead_cost, -dead_cost, 0.0]])
    back = np.zeros((n, 3), dtype=np.int8)
    score = emis[0].copy()
    if start is not None:
        score = np.full(3, -1e9)
        score[start] = emis[0, start]
    for t in range(1, n):
        cand = score[:, None] + trans  # from x to
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + emis[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def team_priors(events, n, weight):
    """Weak per-frame team evidence from Smart Tagging phases with a team tag."""
    prior = np.zeros((n, 2))
    attack = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
              "BALL IN FINAL THIRD", "BALL IN THE BOX", "LONG BALL"}
    defend = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
              "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}
    for e in events:
        team = {"barcelona": H, "atletic club": A}.get(e["team"])
        if team is None:
            continue
        if e["code"] in attack:
            prior[e["start"]:e["end"], team] += weight
        elif e["code"] in defend:
            prior[e["start"]:e["end"], 1 - team] += weight
    return prior


def label(f, halves, events=(), params=None, use_team_priors=False, team_prior_weight=0.02,
          shape_logit=None):
    """Full-length label array (H/A/D/X letters). X outside the halves."""
    p = dict(PARAMS, **(params or {}))
    n = len(f["ball_ok"])
    cH, cA = control_evidence(f, p)
    restarts = restart_frames(events, p) if events else []
    dead = dead_evidence(f, p, restarts)

    emis = np.zeros((n, 3))
    w = p["control_weight"]
    emis[:, H] = w * cH - w * cA
    emis[:, A] = w * cA - w * cH
    # Control says which team, not whether the ball is alive: a player standing over a ball placed
    # for a free kick is "in control" too. So it is neutral between D and the controlling team.
    emis[:, D] = dead + w * np.maximum(cH, cA)
    if shape_logit is not None and p["shape_weight"] > 0:
        # Only where control says nothing (ball unseen, or in flight): shape breaks the tie that
        # would otherwise be settled arbitrarily by the decoder.
        quiet = np.maximum(cH, cA) < 0.05
        s = p["shape_weight"] * np.clip(shape_logit, -p["shape_clip"], p["shape_clip"]) * quiet
        emis[:, H] += s / 2
        emis[:, A] -= s / 2
    if use_team_priors and events:
        prior = team_priors(events, n, team_prior_weight)
        emis[:, H] += prior[:, H] - prior[:, A]
        emis[:, A] += prior[:, A] - prior[:, H]

    # Nothing is in play before a half kicks off. Until Smart Tagging opens that half's kick-off
    # set piece, the only allowed state is D: players warming up or walking to their positions
    # near the ball are not a possession.
    kickoffs = kickoff_tags(events) if events else []
    for _, first, last in halves:
        for k in kickoffs:
            if first < k <= last:
                emis[first:k, H] = emis[first:k, A] = -1e9

    out = np.full(n, "X", dtype="<U1")
    for _, first, last in halves:
        sl = slice(first, last + 1)
        path = viterbi(emis[sl], p["switch_cost"], p["dead_cost"], start=p.get("start_state"))
        out[sl] = LETTERS[path]
    return out
