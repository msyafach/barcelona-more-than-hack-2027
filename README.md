# Ball Possession Segmentation for More than a Hack 2027

An algorithmic pipeline that labels every frame of the FC Barcelona vs Athletic Club tactical video as home possession (H), away possession (A) or dead ball (D), built for Phase 1 of the More than a Hack 2027 challenge.

![Pipeline overview](docs/pipeline.svg)

## Features
- Three-state H/A/D model decoded with Viterbi, so a possession only changes when the evidence pays a fixed switch cost. A touch, a deflection or a ball flying past a player does not flip the label.
- Graded control evidence from the automatic tracking: full weight within 1 m of the ball, fading to zero at 3 m, scaled by how much closer one team is than the other.
- Dead-ball detector built from collective player speed, ball out of the pitch (held until the restart), and the restarts implied by the automatic Smart Tagging set pieces.
- Self-supervised team-shape model (width, depth, territory, forward motion) that fills the 41% of graded frames where the ball is not tracked. It is trained on the pipeline's own confident frames, never on any human label.
- Fully deterministic: the same data always produces the same `possession.csv`, byte for byte.
- Ships the organisers' official scorer and validator so the output can be checked locally.

## Prerequisites
- Python 3.13 (tested on Windows 11). Python 3.10 or newer should work with the pinned packages.
- The gated match bundle from the organisers (not included, see Installation step 3).

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
   On macOS or Linux, activate with `source .venv/bin/activate`.
3. Provide the match data. Use one of these:
   - unpack the bundle into `./data` (`data/tactical/...`, `data/broadcast/...`), or
   - keep it in the folder that contains this repository (`../tactical/...`), or
   - point `MTAH_DATA_DIR` at it:
     ```bash
     export MTAH_DATA_DIR=/path/to/mtah2027
     ```

The data is licensed for the competition only. `.gitignore` keeps it, and every cache derived from it, out of git.

## Usage

Regenerate the submission:
```bash
python -m possession.run
```
This writes `possession.csv` (199,319 rows, `frame,label`) in a few seconds. The first run also writes a parsed cache of the tracking under `<data dir>/cache/`.

Write the file under the team name for the submission form:
```bash
python -m possession.run --out <team_name>.csv
```

Validate the file with the organisers' checks:
```bash
python evaluation/validate_submission.py possession.csv
```

Score it against the public 5-minute ground truth (a pipeline check only, never used for tuning):
```bash
mkdir -p evaluation/results && cp possession.csv evaluation/results/
python evaluation/run_evaluation.py
```

Ablations and self-checks:
```bash
python -m possession.run --no-shape --out evaluation/results/no_shape.csv
python -m possession.run --team-priors --out evaluation/results/team_priors.csv
python -m possession.proxy_eval evaluation/results/*.csv
```

The approach, the assets used and dropped, the possession rules and the limitations are described in [TECHNICAL.md](TECHNICAL.md), also provided as `TECHNICAL.pdf`. To rebuild the PDF (black-and-white Mermaid diagram, needs Chrome or Edge):
```bash
pip install -r docs/requirements-docs.txt
python docs/build_pdf.py --out <team_name>.pdf
```

## Repository layout
```
possession/
  io.py                   read the tracking and Smart Tagging into cached numpy arrays
  features.py             per-frame features in metres and seconds
  model.py                control and dead-ball evidence, Viterbi decoder
  shape.py                self-supervised team-shape model for ball gaps
  align.py                broadcast-to-tactical alignment (experimental, off by default)
  build_broadcast_ball.py cache the aligned broadcast ball (experimental)
  proxy_eval.py           full-match self-checks that need no ground truth
  run.py                  entry point, writes possession.csv
docs/build_pdf.py         renders TECHNICAL.md to PDF
games.py                  match constants and data paths (from the organisers' repository)
evaluation/               organisers' scorer, validator and public ground truth, unchanged
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. `games.py` and `evaluation/` come from the organisers' MIT-licensed repository (see `LICENSE-organisers-MIT`). The match data is not covered by any licence in this repository.
