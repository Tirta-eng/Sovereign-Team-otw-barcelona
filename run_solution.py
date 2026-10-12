"""Pipeline resmi tim Sovereign-Team untuk More Than A Hack 2027.

Menjalankan alur lengkap end-to-end:
1. Ekstraksi fitur posisi & tracking dari data/tactical/
2. Melatih model Machine Learning pada ground truth publik 5 menit
3. Memprediksi 199.319 frame pertandingan
4. Menerapkan temporal smoothing window (majority vote)
5. Menyimpan output submission resmi ke evaluation/results/Sovereign-Team.csv
6. Menjalankan validasi resmi (validate_submission.py)
7. Menghitung skor resmi (run_evaluation.py)
"""

import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import games

def run():
    parser = argparse.ArgumentParser(description="End-to-end pipeline Sovereign-Team")
    parser.add_argument("--window", type=int, default=25, help="Smoothing half-window in frames (default: 25)")
    parser.add_argument("--trees", type=int, default=100, help="Number of trees for Random Forest (default: 100)")
    args = parser.parse_args()

    team_submission = os.path.join(ROOT, "evaluation", "results", "Sovereign-Team.csv")
    model_artifact = os.path.join(ROOT, "artifacts", "sovereign_model.joblib")
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    train_script = os.path.join(ROOT, "mtah-base-code", "starter", "train_possession.py")

    print("=" * 70)
    print("SOVEREIGN-TEAM: OT MEMULAI PIPELINE END-TO-END")
    print("=" * 70)

    # 1. Training & Prediction
    print("\n[Langkah 1/3] Menjalankan Training & Prediksi Model...")
    cmd_train = [
        sys.executable, train_script, "train",
        "--gt", gt_path,
        "--model", model_artifact,
        "--out", team_submission,
        "--window", str(args.window),
        "--trees", str(args.trees)
    ]
    res_train = subprocess.run(cmd_train)
    if res_train.returncode != 0:
        print("[GAGAL] Terjadi kesalahan saat training/prediksi.")
        sys.exit(1)

    # 2. Validasi Format Submisi
    print("\n[Langkah 2/3] Memvalidasi Format File Submisi Resmi...")
    validator = os.path.join(ROOT, "evaluation", "validate_submission.py")
    res_val = subprocess.run([sys.executable, validator, team_submission])
    if res_val.returncode != 0:
        print("[GAGAL] File submisi tidak lolos validasi format.")
        sys.exit(1)

    # 3. Evaluasi Skor
    print("\n[Langkah 3/3] Menghitung Skor Resmi di Scoreboard...")
    scorer = os.path.join(ROOT, "evaluation", "run_evaluation.py")
    subprocess.run([sys.executable, scorer])

    print("\n" + "=" * 70)
    print("PIPELINE SELESAI!")
    print(f"File submisi tersimpan di: {team_submission}")
    print("=" * 70)

if __name__ == "__main__":
    run()
