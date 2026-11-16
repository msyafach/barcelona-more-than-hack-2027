"""Full-match self-checks that need no ground truth.

Smart Tagging is an independent *automatic* output (not human-labelled, not derived from the
ground truth), so its agreement with a submission over the whole match is a usable, noisy --
proxy for choosing between model variants, unlike the 17 possessions of the public window.
Keep the team priors off in any variant judged this way, or the proxy grades its own input.

    python -m possession.proxy_eval evaluation/results/*.csv
"""

import sys

import numpy as np
import pandas as pd

from .io import GAME, load_smart_tagging
from .run import summary
from games import half_video_blocks

ATTACK = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD", "BALL IN THE BOX"}
TEAMS = {"barcelona": "H", "atletic club": "A"}


def proxy(labels, events):
    n = len(labels)
    want = np.full(n, "", dtype="<U1")
    for e in events:
        if e["code"] in ATTACK and e["team"] in TEAMS:
            want[e["start"]:e["end"]] = TEAMS[e["team"]]
    m = (want != "") & np.isin(labels, ["H", "A"])
    team_agree = float((labels[m] == want[m]).mean())

    dead = np.zeros(n, bool)
    alive_after = np.zeros(n, bool)
    for e in events:
        if e["code"] == "SET PIECES":
            dead[e["start"] - 100:e["start"] + 50] = True
            alive_after[e["start"] + 200:e["start"] + 400] = True
    graded = labels != "X"
    d_recall = float((labels[dead & graded] == "D").mean())
    alive_ok = float((labels[alive_after & graded] != "D").mean())
    spells, dur = summary(labels, half_video_blocks(GAME))
    return dict(team_agree=team_agree, sp_dead=d_recall, sp_alive_after=alive_ok,
                poss=spells, dur=dur,
                dead_share=float((labels[graded] == "D").mean()))


if __name__ == "__main__":
    events = load_smart_tagging()
    print("{:<28} {:>9} {:>8} {:>8} {:>5} {:>6} {:>7}".format(
        "file", "teamAgree", "spDead", "spAlive", "poss", "dur", "deadSh"))
    for path in sys.argv[1:]:
        labels = pd.read_csv(path)["label"].values.astype("<U1")
        r = proxy(labels, events)
        print("{:<28} {:>9.3f} {:>8.3f} {:>8.3f} {:>5} {:>6.1f} {:>7.3f}".format(
            path.replace("\\", "/").split("/")[-1], r["team_agree"], r["sp_dead"],
            r["sp_alive_after"], r["poss"], r["dur"], r["dead_share"]))
