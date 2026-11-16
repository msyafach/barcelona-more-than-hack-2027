"""Load the tactical ATD and Smart Tagging into numpy arrays, cached next to the data.

The cache lives under MTAH_DATA_DIR/cache, outside the repository, because a frame-level export of
the tracking is derived data that must not be published.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, require, source_frames  # noqa: E402

N_SLOT_FIELDS = 3  # x, y, speed per player slot; ball is x, y, z


def cache_dir():
    path = os.path.join(DATA_DIR, "cache")
    os.makedirs(path, exist_ok=True)
    return path


def team_of_each_track(meta_path):
    """0 for home slots, 1 for away; the first Team in the file is home (FIFA/EPTS convention)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    return np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def attack_direction_first_half(meta_path):
    """{team_index: raw value of attack_direction_first_half} straight from the metadata."""
    root = ET.parse(meta_path).getroot()
    out = {}
    for index, team in enumerate(root.iter("Team")):
        for param in team.iter("ProviderParameter"):
            if param.findtext("Name") == "attack_direction_first_half":
                out[index] = param.findtext("Value")
    return out


def _parse_raw(raw_path, n_slots, n_frames):
    """Stream the raw ATD into (players[n, slots, 3], ball[n, 3]) float32, NaN where missing."""
    players = np.full((n_frames, n_slots, N_SLOT_FIELDS), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    width = 1 + n_slots * N_SLOT_FIELDS + 3
    with open(raw_path) as handle:
        chunk = []
        for line in handle:
            chunk.append(line.replace(":", ",").replace(";", ","))
            if len(chunk) == 20000:
                _store(chunk, players, ball, n_slots, width)
                chunk = []
        if chunk:
            _store(chunk, players, ball, n_slots, width)
    return players, ball


def _store(lines, players, ball, n_slots, width):
    values = np.array(",".join(line.strip() for line in lines).split(","), dtype=np.float64)
    values = values.reshape(-1, width)
    frames = values[:, 0].astype(np.int64) - 1  # ATD frame_id counts from 1
    players[frames] = values[:, 1:1 + n_slots * N_SLOT_FIELDS].reshape(-1, n_slots, N_SLOT_FIELDS)
    ball[frames] = values[:, 1 + n_slots * N_SLOT_FIELDS:]


def load_tracking(game=GAME, refresh=False, source="tactical"):
    """dict(players, ball, team) for one source, in normalised 0-1 pitch coordinates."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, source=source)
    cache = os.path.join(cache_dir(), "{}_{}_atd.npz".format(game, source))
    team = team_of_each_track(meta_path)
    if os.path.isfile(cache) and not refresh:
        data = np.load(cache)
        return {"players": data["players"], "ball": data["ball"], "team": team}
    players, ball = _parse_raw(raw_path, len(team), source_frames(source, game))
    np.savez(cache, players=players, ball=ball)
    return {"players": players, "ball": ball, "team": team}


def load_smart_tagging(game=GAME):
    """[{code, start, end (video frames, end exclusive), team, type, half}]."""
    path = require(["smart_tagging"], game)[0]
    fps = GAMES[game]["fps"]
    out = []
    for inst in ET.parse(path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        out.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start": int(round(float(inst.findtext("start")) * fps)),
            "end": int(round(float(inst.findtext("end")) * fps)),
            "team": (tags.get("Team") or "").strip().lower(),
            "type": (tags.get("Type") or "").strip().lower(),
            "half": (tags.get("Half") or "").strip(),
        })
    return out
