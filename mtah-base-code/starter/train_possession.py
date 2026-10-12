"""Train a first supervised baseline for MTAH 2027 possession labels.

Run from the root of metrica-sports/more-than-a-hack-2027.
Requires the authorized match bundle under data/ (or MTAH_DATA_DIR).
The public ground truth labels only the first five minutes; rows labelled X are
unlabelled outside that window and are deliberately excluded from training.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

LABELS = ("H", "A", "D")
MISSING = "N"
FEATURE_COLUMNS = [
    "ball_seen", "ball_x_m", "ball_y_m", "ball_speed",
    "home_players_seen", "away_players_seen",
    "home_mean_speed", "away_mean_speed",
    "nearest_home_m", "nearest_away_m", "nearest_distance_diff_m",
    "nearest_distance_min_m", "nearest_home_speed", "nearest_away_speed",
]


def track_teams(meta_path: str) -> np.ndarray:
    root = ET.parse(meta_path).getroot()
    teams = [team.get("id") for team in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError(f"Expected exactly two teams in {meta_path}; found {len(teams)}")
    home_id = teams[0]
    player_team_ids = [player.get("teamId") for player in root.iter("Player")]
    if not player_team_ids:
        raise ValueError(f"No player/track slots found in {meta_path}")
    return np.asarray([0 if team_id == home_id else 1 for team_id in player_team_ids], dtype=np.int8)


def parse_point(token: str):
    if not token or token.startswith(MISSING):
        return None
    try:
        x, y, speed = token.split(",")[:3]
        vals = np.asarray([float(x), float(y), float(speed)], dtype=float)
        if not np.isfinite(vals).all():
            return None
        return vals
    except (ValueError, IndexError):
        return None


def extract_features(meta_path: str, raw_path: str, n_frames: int, pitch) -> pd.DataFrame:
    """Create one row per tactical video frame from the official ATD text format."""
    teams = track_teams(meta_path)
    pitch_length, pitch_width = map(float, pitch)
    rows = np.full((n_frames, len(FEATURE_COLUMNS)), np.nan, dtype=np.float32)
    # Initial defaults are useful to distinguish a missing ball from a valid position.
    rows[:, 0] = 0.0
    rows[:, 1:3] = np.nan
    rows[:, 3:] = np.nan

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) != 3:
                continue
            try:
                frame = int(parts[0]) - 1
            except ValueError:
                continue
            if frame < 0 or frame >= n_frames:
                continue

            home_xy, away_xy, home_s, away_s = [], [], [], []
            for slot, token in enumerate(parts[1].split(";")):
                if slot >= len(teams):
                    break
                point = parse_point(token)
                if point is None:
                    continue
                x_m, y_m, speed = point[0] * pitch_length, point[1] * pitch_width, point[2]
                if teams[slot] == 0:
                    home_xy.append((x_m, y_m)); home_s.append(speed)
                else:
                    away_xy.append((x_m, y_m)); away_s.append(speed)

            ball = parse_point(parts[2])
            rows[frame, 4] = len(home_xy)
            rows[frame, 5] = len(away_xy)
            rows[frame, 6] = float(np.mean(home_s)) if home_s else np.nan
            rows[frame, 7] = float(np.mean(away_s)) if away_s else np.nan
            if ball is None:
                continue

            bx, by, bs = ball[0] * pitch_length, ball[1] * pitch_width, ball[2]
            rows[frame, 0:4] = (1.0, bx, by, bs)
            if home_xy:
                hxy = np.asarray(home_xy)
                hd = np.sqrt(np.sum((hxy - [bx, by]) ** 2, axis=1))
                rows[frame, 8] = np.min(hd)
                rows[frame, 12] = home_s[int(np.argmin(hd))]
            if away_xy:
                axy = np.asarray(away_xy)
                ad = np.sqrt(np.sum((axy - [bx, by]) ** 2, axis=1))
                rows[frame, 9] = np.min(ad)
                rows[frame, 13] = away_s[int(np.argmin(ad))]
            if np.isfinite(rows[frame, 8]) and np.isfinite(rows[frame, 9]):
                rows[frame, 10] = rows[frame, 8] - rows[frame, 9]
                rows[frame, 11] = min(rows[frame, 8], rows[frame, 9])

    frame_index = pd.RangeIndex(n_frames, name="frame")
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS, index=frame_index)


def read_labels(path: str, n_frames: int) -> np.ndarray:
    labels = np.full(n_frames, "X", dtype="<U1")
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"frame", "label"}.issubset(reader.fieldnames):
            raise ValueError("Ground truth CSV must contain frame,label columns")
        for row in reader:
            i = int(row["frame"])
            if 0 <= i < n_frames:
                label = row["label"].strip().upper()
                if label in (*LABELS, "X"):
                    labels[i] = label
    return labels


def playable_mask(n_frames: int) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[max(0, first):min(n_frames, last + 1)] = True
    return mask


def smooth_labels(labels: np.ndarray, window: int, valid_mask: np.ndarray) -> np.ndarray:
    if window <= 0:
        out = labels.copy()
        out[~valid_mask] = "X"
        return out
    # Efficient centred majority vote; only H/A/D are considered inside play.
    codes = {"H": 0, "A": 1, "D": 2}
    numeric = np.asarray([codes.get(str(x), -1) for x in labels], dtype=np.int8)
    idx = np.arange(len(labels))
    lo = np.maximum(0, idx - window)
    hi = np.minimum(len(labels), idx + window + 1)
    counts = []
    for c in range(3):
        cs = np.concatenate(([0], np.cumsum(numeric == c)))
        counts.append(cs[hi] - cs[lo])
    winner = np.argmax(np.vstack(counts), axis=0)
    reverse = np.asarray(["H", "A", "D"])
    out = reverse[winner]
    out[~valid_mask] = "X"
    return out


def write_submission(path: str, labels: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["frame", "label"])
        writer.writerows((i, label) for i, label in enumerate(labels))


def load_features() -> pd.DataFrame:
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    return extract_features(meta_path, raw_path, source_frames("tactical"), GAMES[GAME]["pitch"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("--gt", default="evaluation/gt/BAR-ATH_possession_gt_public.csv")
    parser.add_argument("--model", default="artifacts/possession_rf.joblib")
    parser.add_argument("--out", default="evaluation/results/possession_rf.csv")
    parser.add_argument("--window", type=int, default=25, help="smoothing half-window in frames; 25 = +/- 1 second")
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()

    n_frames = source_frames("tactical")
    valid_mask = playable_mask(n_frames)
    print("Extracting tactical ATD features...")
    X = load_features()

    if args.mode == "train":
        y_all = read_labels(args.gt, n_frames)
        train_mask = np.isin(y_all, LABELS)
        if int(train_mask.sum()) == 0:
            raise ValueError(f"No labelled H/A/D frames found in {args.gt}")
        y = y_all[train_mask]
        print(f"Training on {len(y):,} labelled frames; counts: " + str({k: int(np.sum(y == k)) for k in LABELS}))
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("classifier", RandomForestClassifier(
                n_estimators=args.trees, class_weight="balanced_subsample",
                min_samples_leaf=5, max_features="sqrt", n_jobs=-1,
                random_state=2027,
            )),
        ])
        model.fit(X.loc[train_mask], y)
        os.makedirs(os.path.dirname(args.model) or ".", exist_ok=True)
        joblib.dump({"model": model, "features": FEATURE_COLUMNS}, args.model)
        print(f"Saved model: {args.model}")
    else:
        if not os.path.isfile(args.model):
            raise FileNotFoundError(f"Model not found: {args.model}. Run mode=train first.")
        bundle = joblib.load(args.model)
        model = bundle["model"]

    predicted = model.predict(X)
    predicted = smooth_labels(predicted, args.window, valid_mask)
    write_submission(args.out, predicted)
    print(f"Wrote {args.out} ({len(predicted):,} rows)")
    print("Labels:", {k: int(np.sum(predicted == k)) for k in ("H", "A", "D", "X")})
    print("Next: python evaluation/validate_submission.py " + args.out)
    print("Then: python evaluation/run_evaluation.py")
    print("Important: public GT covers only five minutes; its score is a pipeline check, not a reliable full-match estimate.")


if __name__ == "__main__":
    main()
