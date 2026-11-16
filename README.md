# Ball Possession Segmentation for More than a Hack 2027

Our entry for Phase 1 of More than a Hack 2027: for every frame of FC Barcelona vs Athletic Club, it works out whether Barcelona has the ball (H), Athletic has it (A), or the ball is dead (D).

![Pipeline overview](docs/pipeline.svg)

## Features
- **Possessions, not flickers.** The match is modelled as three states (H, A, D) and decoded with Viterbi. Switching teams has a cost, so a touch, a deflection or a ball flying past someone doesn't start a new possession.
- **Control that tolerates noisy tracking.** The tracked ball wobbles by about a metre, so instead of a hard distance cut-off we use a sliding scale: full credit within 1 m of the ball, nothing beyond 3 m, weighted by how much closer one team is than the other.
- **Knows when the ball is dead.** It looks for the whole field walking, the ball going over a line (and staying dead until the restart), and the set pieces flagged by Metrica's automatic Smart Tagging.
- **Copes with the missing ball.** The ball isn't tracked on 41% of graded frames. For those stretches a small model reads the shape of the two teams instead. It learns only from the pipeline's own confident frames, never from human labels.
- **Same file every time.** Run it twice and you get a byte-identical `possession.csv`.
- **Checked with the official tools.** The organisers' scorer and validator are included so you can test the output locally.

## Prerequisites
- Python 3.12 or newer (the pinned numpy 2.5.5 needs it). We tested on Python 3.13 under Windows 11.
- The match data from the organisers. It isn't in this repository; step 3 below explains where to put it.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url> solution
   cd solution
   ```
2. Install dependencies:
   ```bash
   python -m venv .venv
   .venv/Scripts/activate
   pip install -r requirements.txt
   ```
   On macOS or Linux, activate with `source .venv/bin/activate` instead.
3. Point it at the match data. Any of these works:
   - unpack the bundle into `./data`, so you have `data/tactical/...` and `data/broadcast/...`;
   - leave it in the folder that contains this repository (`../tactical/...`);
   - or set `MTAH_DATA_DIR`:
     ```bash
     export MTAH_DATA_DIR=/path/to/mtah2027
     ```

The data is licensed for the competition only, so `.gitignore` keeps it (and anything cached from it) out of git.

## Usage

Rebuild the submission:
```bash
python -m possession.run
```
That writes `possession.csv` (199,319 rows of `frame,label`) in a few seconds. The first run also saves a parsed copy of the tracking under `<data dir>/cache/`, which makes later runs faster.

Check the file with the organisers' validator:
```bash
python evaluation/validate_submission.py possession.csv
```

The full write-up (approach, what we used and dropped, the possession rules, limitations) is in [docs/TECHNICAL.md](docs/TECHNICAL.md).

## Results and training metrics

All the numbers in this section come out of one command, so you can check them yourself:
```bash
python -m possession.report
```
Section 4 needs the public ground truth, which we don't redistribute. Copy `BAR-ATH_possession_gt_public.csv` from the challenge repository into `evaluation/gt/` first.

Two things to keep in mind when reading them:
- **Nothing here was trained on labels from this match.** The rules don't allow it, and we didn't. The only part that learns is the team-shape model (section 2), and it learns from the pipeline's own guesses. Each half is always scored by a model that was trained on the other half.
- **Nobody knows the final score yet.** The ranking uses the organisers' private answers for the whole match. The public five minutes contain just 17 possessions, which makes the spellF1 there uncertain by about ±0.17. They tell us the pipeline works; they can't tell us which version is better. To compare versions we used the full-match checks in section 3.

### 1. The submission

| | |
|---|---|
| rows | 199,319 |
| graded frames (the two halves) | 158,857 |
| H / A / D on graded frames | 55,256 / 32,792 / 70,809 |
| X (only before kick-off, at half-time and after the final whistle) | 40,462 |
| possessions | 338 |
| average possession length | 10.4 s |
| share of graded frames with the ball dead | 44.6% |
| Barcelona's share of live-ball time | 62.8% |

### 2. How the team-shape model trained

It's a logistic regression that guesses which team has the ball from how the players are spread out. Its training examples are frames where the ball is visible and one team is clearly on it. The model used on each half is trained on the other half, and the accuracy below is measured on the half it never saw.

| used on | training frames | test frames | test accuracy | always guessing the team with more of the ball | gain |
|---|---|---|---|---|---|
| first half | 15,208 | 16,382 | 86.1% | 67.5% | +18.6 points |
| second half | 16,382 | 15,208 | 83.9% | 62.0% | +21.9 points |

What it learned (standardised weights; positive means "looks like Barcelona has the ball"):

| feature | model for the first half | model for the second half |
|---|---|---|
| width difference (Barcelona minus Athletic) | +1.340 | +1.614 |
| depth difference (Barcelona minus Athletic) | +0.346 | +0.203 |
| how far the whole block sits toward Athletic's goal | -0.183 | -0.035 |
| how fast the block moves toward Athletic's goal | +0.519 | +0.205 |
| average speed difference (Barcelona minus Athletic) | -0.712 | -0.484 |
| intercept | +0.712 | +1.121 |

The biggest signal by far is width: the team with the ball spreads out. Both halves learned the same signs, which suggests it's picking up something real about the game rather than memorising one half.

### 3. Full-match checks (no ground truth needed)

| version | agrees with Smart Tagging's attacking team | possessions | average length | ball dead |
|---|---|---|---|---|
| submission | 0.792 | 338 | 10.4 s | 44.6% |
| without the team-shape model | 0.776 | 326 | 10.8 s | 44.7% |
| with Smart Tagging team priors (dropped) | 0.857 | 324 | 11.0 s | 44.2% |

The last row looks best, but don't be fooled: that version feeds on the same Smart Tagging tags this check compares against, so it's grading its own homework.

### 4. Public five minutes (scored with the organisers' own code)

| version | macroF1 | spellF1 | spellF1, 90% range | possessions (truth: 17) | live/dead F1 | right team |
|---|---|---|---|---|---|---|
| submission | 0.8740 | 0.5176 | 0.34 to 0.68 | 17 | 0.9642 | 0.9030 |
| without the team-shape model | 0.8462 | 0.4706 | 0.31 to 0.64 | 17 | 0.9648 | 0.8594 |
| with Smart Tagging team priors (dropped) | 0.8453 | 0.4588 | 0.29 to 0.62 | 17 | 0.9663 | 0.8748 |

How the submission does on each label:

| label | precision | recall | F1 | frames |
|---|---|---|---|---|
| H | 0.8803 | 0.8900 | 0.8851 | 3,454 |
| A | 0.8724 | 0.9205 | 0.8958 | 2,577 |
| D | 0.8999 | 0.7897 | 0.8412 | 1,469 |

Where the mistakes go (rows are the truth, columns what we said):

| | H | A | D |
|---|---|---|---|
| H | 3,074 | 273 | 107 |
| A | 183 | 2,372 | 22 |
| D | 235 | 74 | 1,160 |

spellF1 at each overlap threshold:

| overlap needed | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| F1 | 0.6471 | 0.5882 | 0.5882 | 0.4706 | 0.2941 |

How often we place a change of possession close to the real one:

| within | precision | recall | F1 |
|---|---|---|---|
| 12 frames (0.5 s) | 0.2000 | 0.2000 | 0.2000 |
| 25 frames (1.0 s) | 0.3000 | 0.3000 | 0.3000 |

Our biggest weakness is timing: the right possessions, but their edges land late or early. That's why spellF1 falls off at the stricter overlap thresholds.

### 5. How we got here

Each version builds on the one before. The organisers' baseline is the bar to clear.

| version | what changed | public macroF1 | public spellF1 | possessions (full match) | agrees with Smart Tagging |
|---|---|---|---|---|---|
| organisers' baseline (smooth50) | nearest player, then a majority vote | 0.6207 | 0.3256 | 594 | 0.702 |
| v1 | three-state Viterbi, hard control cut-off, first dead-ball rules | 0.6873 | 0.4000 | 315 | 0.769 |
| v2 | standing near a dead ball no longer counts as play; pass check over a short window | 0.8380 | 0.4471 | 273 | 0.788 |
| v3 | sliding-scale control (full at 1 m, none at 3 m) | 0.8462 | 0.4706 | 328 | 0.765 |
| v4 | team-shape model for ball gaps | 0.8738 | 0.5176 | 342 | 0.787 |
| v5 (submitted) | ball stays dead from leaving the pitch until the restart | 0.8740 | 0.5176 | 338 | 0.792 |

On the full match, the organisers' baseline scores spellF1 0.2433. That's the published floor.

### 6. How sensitive it is

We nudged each main setting by roughly half its value to see whether the result hangs on a lucky number. For most of them, at most 2% of graded frames change:

| setting | default | tried | frames changed | possessions |
|---|---|---|---|---|
| cost of switching teams | 10 | 6 / 15 | 0.9% / 1.8% | 399 / 270 |
| cost of entering or leaving a stoppage | 25 | 15 / 40 | 1.6% / 1.4% | 349 / 328 |
| distance where control fades out | 3.0 m | 2.5 m | 1.0% | 306 |
| weight of the team-shape model | 0.05 | 0.02 / 0.10 | 1.6% / 1.3% | 328 / 344 |
| walking-speed threshold | 1.4 m/s | 1.2 / 1.6 m/s | 8.6% / 6.0% | 349 / 325 |

The one to watch is the walking-speed threshold. Between 1.2 and 1.6 m/s, the share of the match we call dead moves from 37% to 50%.

## Repository layout
```
solution/
├─ README.md
├─ LICENSE                       MIT, for our code
├─ LICENSE-organisers-MIT        MIT, for games.py and evaluation/ (from the organisers)
├─ requirements.txt              pinned numpy and pandas
├─ games.py                      match constants and data paths (from the organisers; it has to stay at the root)
├─ possession/                   the pipeline
│  ├─ run.py                     start here: builds possession.csv
│  ├─ io.py                      reads the tracking and Smart Tagging into cached numpy arrays
│  ├─ features.py                per-frame features in metres and seconds
│  ├─ model.py                   control and dead-ball evidence, and the Viterbi decoder
│  ├─ shape.py                   the team-shape model for ball gaps
│  ├─ proxy_eval.py              full-match checks that need no ground truth
│  ├─ report.py                  prints every number in this README
│  └─ experimental/              things we tried and dropped (switched on with --broadcast)
│     ├─ align.py                lines up the broadcast tracking with the tactical video
│     └─ build_broadcast_ball.py caches the aligned broadcast ball
├─ evaluation/                   the organisers' scorer and validator, untouched
│  └─ gt/                        put the public ground truth here; not committed
└─ docs/                        the technical write-up and the diagram above
```

## License
Our code is released under the [MIT](https://choosealicense.com/licenses/mit/) License. `games.py` and everything in `evaluation/` come from the organisers' repository, which is also MIT (see `LICENSE-organisers-MIT`). Neither licence covers the match data.
