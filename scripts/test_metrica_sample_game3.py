"""Benchmark and test possession models on real Metrica Sports Sample_Game_3 data.

Sample_Game_3 contains:
- 143,761 frames of optical tracking (25 fps, ~95 minutes)
- 35 tracked players in Metrica FIFA/EPTS format
- 3,620 tagged events (Pass, Carry, Shot, Ball Out, Recovery, Set Piece, etc.)
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm

DATA_DIR = os.path.join(ROOT, "sample-data", "data", "Sample_Game_3")
META_PATH = os.path.join(DATA_DIR, "Sample_Game_3_metadata.xml")
TRACK_PATH = os.path.join(DATA_DIR, "Sample_Game_3_tracking.txt")
EVENTS_PATH = os.path.join(DATA_DIR, "Sample_Game_3_events.json")

def parse_metadata():
    """Extract team IDs, player mapping, and pitch dimensions from EPTS metadata."""
    tree = ET.parse(META_PATH)
    root = tree.getroot()
    
    # Namespace handling
    ns = {"epts": "FIFADataTransferFormatEPTSNamespace/FIFADataTransferFormatEPTS"}
    teams = root.findall(".//epts:Team", ns) or root.findall(".//Team")
    team_a_id = teams[0].get("id")
    team_b_id = teams[1].get("id")
    
    players = root.findall(".//epts:Player", ns) or root.findall(".//Player")
    player_teams = np.array([0 if p.get("teamId") == team_a_id else 1 for p in players], dtype=np.int8)
    
    length_elem = root.find(".//epts:FieldLength", ns) or root.find(".//FieldLength")
    width_elem = root.find(".//epts:FieldWidth", ns) or root.find(".//FieldWidth")
    field_length = float(length_elem.text) if length_elem is not None else 105.0
    field_width = float(width_elem.text) if width_elem is not None else 68.0

    return team_a_id, team_b_id, player_teams, (field_length, field_width)

def derive_ground_truth(n_frames):
    """Build ground truth labels from 3,620 Metrica events."""
    with open(EVENTS_PATH, "r", encoding="utf-8") as f:
        events = json.load(f)["data"]
    
    gt = np.full(n_frames, "X", dtype="<U1")
    
    # 1. Fill event intervals
    for e in events:
        start_f = e.get("start", {}).get("frame")
        end_f = e.get("end", {}).get("frame")
        if start_f is None or end_f is None:
            continue
        
        start_idx = max(0, int(start_f) - 1)
        end_idx = min(n_frames, int(end_f))
        
        etype = e.get("type", {}).get("name", "")
        team = e.get("team", {}).get("name", "") if e.get("team") else ""
        
        if etype in ("PASS", "CARRY", "SHOT", "RECOVERY", "CHALLENGE"):
            lbl = "H" if team == "Team A" else ("A" if team == "Team B" else "D")
            gt[start_idx:end_idx] = lbl
        elif etype in ("BALL OUT", "SET PIECE"):
            gt[start_idx:end_idx] = "D"

    # Forward fill gaps during active play periods
    # Period 1: ~frame 361 to 69661
    # Period 2: ~frame 71000 to 143761
    for start, end in [(361, 69661), (71000, 143761)]:
        last = "D"
        for i in range(start, min(end, n_frames)):
            if gt[i] in ("H", "A", "D"):
                last = gt[i]
            else:
                gt[i] = last

    return gt

def extract_features_and_nearest(player_teams, pitch):
    """Parse raw tracking text, return nearest team codes and ML features."""
    length, width = pitch
    n_slots = len(player_teams)
    
    lines = []
    with open(TRACK_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    n_frames = len(lines)
    nearest_codes = np.full(n_frames, -1, dtype=np.int8) # 0=H, 1=A, -1=no ball
    
    # Features for ML: [ball_seen, nearest_home_m, nearest_away_m, diff_m, min_dist_m]
    features = np.zeros((n_frames, 5), dtype=np.float32)
    
    for i, line in enumerate(lines):
        parts = line.strip().split(":")
        if len(parts) != 3:
            continue
        
        ball_part = parts[2]
        if not ball_part or ball_part.startswith("NaN"):
            features[i, 0] = 0.0
            features[i, 1:] = 50.0 # large distance when ball invisible
            continue
        
        try:
            bx, by = [float(x) for x in ball_part.split(",")[:2]]
        except Exception:
            continue
        
        bx_m, by_m = bx * length, by * width
        features[i, 0] = 1.0
        
        player_tokens = parts[1].split(";")
        min_home_dist = 999.0
        min_away_dist = 999.0
        
        for slot, tok in enumerate(player_tokens[:n_slots]):
            if not tok or tok.startswith("NaN"):
                continue
            coords = tok.split(",")
            px_m, py_m = float(coords[0]) * length, float(coords[1]) * width
            dist = np.sqrt((px_m - bx_m)**2 + (py_m - by_m)**2)
            
            if player_teams[slot] == 0:
                if dist < min_home_dist:
                    min_home_dist = dist
            else:
                if dist < min_away_dist:
                    min_away_dist = dist
        
        if min_home_dist < min_away_dist:
            nearest_codes[i] = 0 # H
        else:
            nearest_codes[i] = 1 # A
            
        features[i, 1] = min_home_dist
        features[i, 2] = min_away_dist
        features[i, 3] = min_home_dist - min_away_dist
        features[i, 4] = min(min_home_dist, min_away_dist)
        
    return n_frames, nearest_codes, features

def smooth_labels(codes, window):
    """Majority vote smoothing on integer codes."""
    if window <= 0:
        return np.where(codes == 0, "H", np.where(codes == 1, "A", "D"))
    
    n = len(codes)
    numeric = np.where(codes == 0, 0, np.where(codes == 1, 1, 2))
    idx = np.arange(n)
    lo = np.maximum(0, idx - window)
    hi = np.minimum(n, idx + window + 1)
    
    counts = []
    for c in range(3):
        cs = np.concatenate(([0], np.cumsum(numeric == c)))
        counts.append(cs[hi] - cs[lo])
    winner = np.argmax(np.vstack(counts), axis=0)
    mapping = np.array(["H", "A", "D"])
    return mapping[winner]

def run_tests():
    print("=" * 75)
    print("BENCHMARK METRICA SPORTS SAMPLE_GAME_3 DATASET")
    print("=" * 75)
    
    print("\n1. Membaca Metadata EPTS Sample_Game_3...")
    team_a, team_b, player_teams, pitch = parse_metadata()
    print(f"   Team Home: {team_a}, Team Away: {team_b}")
    print(f"   Total Player Slots: {len(player_teams)}")
    print(f"   Pitch: {pitch[0]}m x {pitch[1]}m")
    
    print("\n2. Memproses Tracking Data & Ekstraksi Fitur...")
    n_frames, nearest_codes, features = extract_features_and_nearest(player_teams, pitch)
    print(f"   Total Frame: {n_frames:,} (25 fps = {n_frames/25/60:.1f} menit)")
    print(f"   Ball seen: {int(np.sum(features[:, 0])):,} frames ({100*features[:, 0].mean():.1f}%)")
    
    print("\n3. Membentuk Ground Truth dari 3,620 Metrica Events...")
    gt = derive_ground_truth(n_frames)
    gt_codes = pm.label_codes(pd.DataFrame({"frame": np.arange(n_frames), "label": gt}))
    scored_mask = gt != "X"
    print(f"   Graded in-play frames: {int(scored_mask.sum()):,} / {n_frames:,}")
    print("   Label GT distribution:")
    for l in ("H", "A", "D", "X"):
        print(f"     {l}: {int(np.sum(gt == l)):,} ({100*np.sum(gt == l)/n_frames:.1f}%)")
        
    print("\n4. Menguji Model & Menghitung Metrik Kompetisi (macroF1 & spellF1)...")
    results = []

    # Heuristic baselines (Raw, Smooth 25, 50, 125)
    for w in (0, 25, 50, 125):
        name = "nearest_raw" if w == 0 else f"nearest_smooth{w}"
        preds = smooth_labels(nearest_codes, w)
        preds[~scored_mask] = "X"
        pred_codes = pm.label_codes(pd.DataFrame({"frame": np.arange(n_frames), "label": preds}))
        metrics = pm.evaluate(gt_codes, pred_codes)
        results.append({
            "model": name,
            "macroF1": metrics["macro_f1"],
            "spellF1": metrics["spell_f1_mean"],
            "poss": metrics["n_pred_spells"],
            "dur": metrics["mean_spell_duration_s"],
        })

    # Machine Learning: Train Random Forest on first 5 minutes of play
    # Graded window: frame 361 to 7861 (7,500 frames)
    train_start, train_end = 361, 361 + 7500
    train_mask = np.zeros(n_frames, dtype=bool)
    train_mask[train_start:train_end] = True
    train_mask = train_mask & (gt != "X")
    
    print(f"\n5. Melatih Random Forest pada 5 Menit Pertama ({int(train_mask.sum())} frame)...")
    X_train = features[train_mask]
    y_train = gt[train_mask]
    
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
    ])
    rf.fit(X_train, y_train)
    
    # Predict all frames
    rf_raw = rf.predict(features)
    # Test with smoothing 25 and 50
    for w in (25, 50):
        rf_smooth = smooth_labels(np.where(rf_raw == "H", 0, np.where(rf_raw == "A", 1, 2)), w)
        rf_smooth[~scored_mask] = "X"
        pred_codes = pm.label_codes(pd.DataFrame({"frame": np.arange(n_frames), "label": rf_smooth}))
        metrics = pm.evaluate(gt_codes, pred_codes)
        results.append({
            "model": f"random_forest_smooth{w}",
            "macroF1": metrics["macro_f1"],
            "spellF1": metrics["spell_f1_mean"],
            "poss": metrics["n_pred_spells"],
            "dur": metrics["mean_spell_duration_s"],
        })

    # Print Summary Table
    df_res = pd.DataFrame(results).sort_values("spellF1", ascending=False)
    print("\n" + "=" * 75)
    print("HASIL EVALUASI MODEL PADA DATASET REAL METRICA SPORTS (SAMPLE_GAME_3)")
    print("=" * 75)
    print(f"{'Model / Konfigurasi':<26} {'macroF1':<10} {'spellF1':<10} {'Possessions':<13} {'Durasi (s)'}")
    print("-" * 75)
    for _, row in df_res.iterrows():
        print(f"{row['model']:<26} {row['macroF1']:<10.4f} {row['spellF1']:<10.4f} {row['poss']:<13} {row['dur']:.1f}")
    print("=" * 75)

    # Save output
    out_csv = os.path.join(ROOT, "evaluation", "metrica_sample_game3_benchmark.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nHasil benchmark disimpan ke: {out_csv}")

if __name__ == "__main__":
    run_tests()
