# MTAH 2027 — supervised baseline add-on

This folder adds an initial supervised Random Forest experiment to the official `metrica-sports/more-than-a-hack-2027` repository. It does not replace the official starter baseline or evaluation code.

## Where to put it

Copy `starter/train_possession.py` into the same `starter/` folder in the official repository. Keep the official `games.py`, `evaluation/`, `starter/baseline.py`, and official data layout unchanged.

## Environment

The official evaluation needs NumPy and pandas. This experiment additionally needs scikit-learn and joblib:

```bash
python -m pip install scikit-learn joblib
```

The match bundle must be downloaded through the competition's authorized process and extracted into `data/tactical/` (or configured with `MTAH_DATA_DIR`). A fresh clone does not include match data.

## Train

Run from the root of the official repository:

```bash
python starter/train_possession.py train \
  --gt evaluation/gt/BAR-ATH_possession_gt_public.csv \
  --model artifacts/possession_rf.joblib \
  --out evaluation/results/possession_rf.csv
```

On Windows PowerShell, this can be entered as one line:

```powershell
python starter/train_possession.py train --gt evaluation/gt/BAR-ATH_possession_gt_public.csv --model artifacts/possession_rf.joblib --out evaluation/results/possession_rf.csv
```

## Predict again after training

```powershell
python starter/train_possession.py predict --model artifacts/possession_rf.joblib --out evaluation/results/possession_rf.csv
```

Then validate and score with the official scripts:

```powershell
python evaluation/validate_submission.py evaluation/results/possession_rf.csv
python evaluation/run_evaluation.py
```

## Important methodological caveat

The public ground truth labels only the first five minutes of play. The `X` values elsewhere mean *unlabelled*, not a fourth training class, so they are excluded from training. This is a baseline for experimentation, not a competition-ready model: a Random Forest trained on one short contiguous window can overfit badly and is not guaranteed to beat `starter/baseline.py`. Compare its `spellF1` to the official baseline, and keep the official scripts as the source of truth.

The script writes exactly `frame,label`, 199,319 rows indexed from 0 through 199,318, and uses `X` outside the two in-play halves. It does not include match data in the code package.
