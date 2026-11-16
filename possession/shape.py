"""Team-shape evidence for frames where the ball is not tracked (41% of graded frames).

When the ball is invisible the only information left is how the 22 players are arranged. Football
gives a few robust tendencies: the team in possession spreads out (width and depth) while the
defending team compacts, and the attacking block pushes toward the opponent's goal.

The weights are learned SELF-SUPERVISED: the targets are the model's own confident control frames
(ball visible, a player of one team clearly on it), produced by an algorithm with no human label
and no ground truth. The fit is checked on the other half of the match (train on one half,
test on the other) so its accuracy is an honest out-of-sample number.
"""

import numpy as np

from .features import FPS, LENGTH, WIDTH, centred_mean


def shape_features(tracking, f, attack_sign):
    """(n, k) feature matrix, oriented so positive = 'looks like H has the ball'.

    attack_sign: (n,) +1 where H attacks toward x=1, -1 where toward x=0, 0 outside play.
    """
    P = tracking["players"].astype(np.float64)
    team = tracking["team"]
    x = P[:, :, 0] * LENGTH
    y = P[:, :, 1] * WIDTH
    spd = P[:, :, 2]
    cols = []
    stats = {}
    for code in (0, 1):
        m = team == code
        with np.errstate(all="ignore"):
            sx = np.nanstd(np.where(m, x, np.nan), axis=1)
            sy = np.nanstd(np.where(m, y, np.nan), axis=1)
            cx = np.nanmean(np.where(m, x, np.nan), axis=1)
            sp = np.nanmean(np.where(m, spd, np.nan), axis=1)
        stats[code] = dict(sx=sx, sy=sy, cx=cx, sp=sp)
    half = FPS  # 1 s smoothing
    sm = lambda v: centred_mean(v, half)
    # width / depth: possession spreads
    cols.append(sm(stats[0]["sy"] - stats[1]["sy"]))
    cols.append(sm(stats[0]["sx"] - stats[1]["sx"]))
    # territory: block position along H's attacking direction (both teams' mean)
    allx = (stats[0]["cx"] + stats[1]["cx"]) / 2.0
    cols.append(sm((allx - LENGTH / 2.0) * attack_sign))
    # forward motion of the whole block along H's attacking direction (m/s)
    vx = np.gradient(np.nan_to_num(sm(allx), nan=LENGTH / 2.0)) * FPS
    cols.append(sm(vx * attack_sign))
    # speed difference
    cols.append(sm(stats[0]["sp"] - stats[1]["sp"]))
    X = np.column_stack(cols)
    return np.nan_to_num(X, nan=0.0)


def attack_sign_array(n, halves, home_dir_first_half):
    """+1 where H attacks toward x=1. Teams swap ends at halftime."""
    s = np.zeros(n)
    first = 1.0 if home_dir_first_half == "left_to_right" else -1.0
    for half, a, b in halves:
        s[a:b + 1] = first if half == 1 else -first
    return s


def fit_logistic(X, y, l2=1.0, iters=300):
    """Plain Newton-step logistic regression with standardisation. Returns (mu, sd, w, b)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    Z1 = np.column_stack([Z, np.ones(len(Z))])
    w = np.zeros(Z1.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Z1 @ w))
        g = Z1.T @ (p - y) + l2 * np.r_[w[:-1], 0]
        Hs = (Z1 * (p * (1 - p))[:, None]).T @ Z1 + l2 * np.diag(np.r_[np.ones(len(w) - 1), 0])
        step = np.linalg.solve(Hs, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return mu, sd, w


def predict_logit(model, X):
    mu, sd, w = model
    Z = (X - mu) / sd
    return np.column_stack([Z, np.ones(len(Z))]) @ w


def pseudo_labels(cH, cA, alive_mask, thresh=0.6):
    """Frames where control evidence is clear: y=1 for H, 0 for A; mask of usable frames."""
    use = alive_mask & ((cH > thresh) | (cA > thresh))
    y = (cH > cA).astype(float)
    return use, y


def gap_evidence(tracking, f, halves, home_dir, cH, cA, alive_mask, report=None):
    """Per-frame log-odds H vs A from shape, fitted on the other half (cross-fitted)."""
    n = len(f["ball_ok"])
    X = shape_features(tracking, f, attack_sign_array(n, halves, home_dir))
    use, y = pseudo_labels(cH, cA, alive_mask)
    out = np.zeros(n)
    for half, a, b in halves:
        test = np.zeros(n, bool)
        test[a:b + 1] = True
        train = use & ~test & (np.abs(attack_sign_array(n, halves, home_dir)) > 0)
        model = fit_logistic(X[train], y[train])
        logit = predict_logit(model, X)
        out[test] = logit[test]
        if report is not None:
            held = use & test
            acc = float(((logit[held] > 0) == (y[held] > 0.5)).mean())
            base = float(max(y[held].mean(), 1 - y[held].mean()))
            report.append(dict(half=half, n_train=int(train.sum()), n_test=int(held.sum()),
                               acc=acc, majority=base, w=model[2].round(3).tolist()))
    return out
