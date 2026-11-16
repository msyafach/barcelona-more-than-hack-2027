"""Every metric the README quotes, regenerated from the data in one command.

    python -m possession.report            # prints Markdown tables

Sections:
  1. the submission itself (counts, possessions, shares), which needs no ground truth
  2. training of the one learned component, the team-shape model (self-supervised, cross-fitted)
  3. full-match self-checks against the automatic Smart Tagging, for the submission and ablations
  4. the public 5-minute window, scored by the organisers' own metric code (a pipeline check only)

Section 4 needs evaluation/gt/BAR-ATH_possession_gt_public.csv, copied from the challenge
repository; it is skipped if the file is not there.
"""

import os
import sys

import numpy as np

from . import proxy_eval, run
from .io import GAME, ROOT, load_smart_tagging
from games import half_video_blocks

FEATURES = ("width difference (H minus A)", "depth difference (H minus A)",
            "block position toward A's goal", "block forward speed toward A's goal",
            "mean player speed difference (H minus A)", "intercept")
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def _graded(labels):
    m = np.zeros(len(labels), bool)
    for _, a, b in half_video_blocks(GAME):
        m[a:b + 1] = True
    return m


def section_submission(labels):
    g = _graded(labels)
    n_poss, dur = run.summary(labels, half_video_blocks(GAME))
    h, a, d = ((labels[g] == c).sum() for c in "HAD")
    print("### 1. The submission\n")
    print("| quantity | value |\n|---|---|")
    print("| rows | {:,} |".format(len(labels)))
    print("| graded frames (two halves) | {:,} |".format(int(g.sum())))
    print("| H / A / D on graded frames | {:,} / {:,} / {:,} |".format(h, a, d))
    print("| X (only outside the halves) | {:,} |".format(int((labels == "X").sum())))
    print("| possessions | {} |".format(n_poss))
    print("| mean possession length | {:.1f} s |".format(dur))
    print("| dead-ball share of graded frames | {:.1%} |".format(d / g.sum()))
    print("| home share of ball-alive frames | {:.1%} |".format(h / (h + a)))
    print()


def section_training(report):
    print("### 2. Training of the team-shape model\n")
    print("Logistic regression on pseudo-labels from pass 1 (ball visible, one team clearly on "
          "it). Cross-fitted: the model for each half is trained on the other half only. "
          "Accuracy is measured on the held-out half.\n")
    print("| applied to | train frames | test frames | held-out accuracy | majority-team baseline | lift |")
    print("|---|---|---|---|---|---|")
    for r in report:
        print("| half {} | {:,} | {:,} | {:.1%} | {:.1%} | +{:.1f} pts |".format(
            r["half"], r["n_train"], r["n_test"], r["acc"], r["majority"],
            100 * (r["acc"] - r["majority"])))
    print("\nStandardised weights (positive means \"looks like H has the ball\"):\n")
    print("| feature | trained on half 2 | trained on half 1 |\n|---|---|---|")
    w1, w2 = report[0]["w"], report[1]["w"]
    for name, a, b in zip(FEATURES, w1, w2):
        print("| {} | {:+.3f} | {:+.3f} |".format(name, a, b))
    print()


def section_proxy(variants, events):
    print("### 3. Full-match self-checks (no ground truth)\n")
    print("| variant | Smart Tagging team agreement | possessions | mean length | dead-ball share |")
    print("|---|---|---|---|---|")
    for name, labels in variants:
        r = proxy_eval.proxy(labels, events)
        print("| {} | {:.3f} | {} | {:.1f} s | {:.1%} |".format(
            name, r["team_agree"], r["poss"], r["dur"], r["dead_share"]))
    print("\nThe team-priors variant uses the same Smart Tagging tags as this check, so its "
          "agreement grades its own input and is not comparable.\n")


def section_public(variants):
    if not os.path.isfile(GT_PATH):
        print("### 4. Public window\n\nskipped: copy the public ground truth into evaluation/gt/\n")
        return
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    import possession_metrics as pm
    gt = pm.label_codes(pm.read_label_csv(GT_PATH))
    print("### 4. Public 5-minute window (organisers' metric code)\n")
    print("| variant | macroF1 | spellF1 | spellF1 90% interval | possessions (truth 17) | alive/dead F1 | team accuracy |")
    print("|---|---|---|---|---|---|---|")
    main = None
    for name, labels in variants:
        r = pm.evaluate(gt, pm.label_codes(labels), bootstrap_n=1000)
        main = main or r
        print("| {} | {:.4f} | {:.4f} | {:.2f} to {:.2f} | {} | {:.4f} | {:.4f} |".format(
            name, r["macro_f1"], r["spell_f1_mean"], r["spell_bootstrap_low"],
            r["spell_bootstrap_high"], r["n_pred_spells"], r["alive_dead_f1"],
            r["team_acc_gt_alive"]))
    r = main
    print("\nSubmission, per class:\n")
    print("| class | precision | recall | F1 | support |\n|---|---|---|---|---|")
    for c in ("H", "A", "D"):
        k = r["per_class"][c]
        print("| {} | {:.4f} | {:.4f} | {:.4f} | {:,} |".format(
            c, k["precision"], k["recall"], k["f1"], k["support"]))
    print("\nSubmission, confusion (rows truth, columns prediction):\n")
    print("| | H | A | D |\n|---|---|---|---|")
    for i, c in enumerate("HAD"):
        row = r["confusion"][i]
        print("| {} | {:,} | {:,} | {:,} |".format(c, row[0], row[1], row[2]))
    print("\nSubmission, spellF1 per IoU threshold:\n")
    print("| IoU | " + " | ".join(str(t) for t in sorted(r["spell_f1"])) + " |")
    print("|---|" + "---|" * len(r["spell_f1"]))
    print("| F1 | " + " | ".join("{:.4f}".format(r["spell_f1"][t]["f1"])
                                  for t in sorted(r["spell_f1"])) + " |")
    print("\nSubmission, possession changes found within a tolerance:\n")
    print("| tolerance | precision | recall | F1 |\n|---|---|---|---|")
    for tol in sorted(r["transitions"]):
        t = r["transitions"][tol]
        print("| {} frames ({:.1f} s) | {:.4f} | {:.4f} | {:.4f} |".format(
            tol, tol / 25.0, t["precision"], t["recall"], t["f1"]))
    print()


if __name__ == "__main__":
    report = []
    submission = run.build(report=report)
    variants = [
        ("submission", submission),
        ("without team-shape model", run.build(use_shape=False)),
        ("with team priors (dropped)", run.build(use_team_priors=True)),
    ]
    section_submission(submission)
    section_training(report)
    section_proxy(variants, load_smart_tagging())
    section_public(variants)
