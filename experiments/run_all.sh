#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

START_TIME=$(date +%s)

echo "=========================================="
echo " ML-CHEMBL — pelny pipeline treningowy"
echo " Start: $(date)"
echo "=========================================="
echo ""

step() {
    local n="$1"; shift
    local desc="$*"
    echo ""
    echo "━━━ [$n/6] $desc ━━━"
    echo ""
}

# ------------------------------------------------------------------
# Krok 1: MLP baseline
# ------------------------------------------------------------------
step 1 "MLP baseline (random + scaffold)"
uv run python scripts/train.py train --model MLP --split random --epochs 200 --seed 42
uv run python scripts/train.py train --model MLP --split scaffold --epochs 200 --seed 42

# ------------------------------------------------------------------
# Krok 2: MLP enhanced
# ------------------------------------------------------------------
step 2 "MLP enhanced (deskryptory + MACCS + BatchNorm)"
uv run python scripts/train.py train --model MLP --split random --epochs 200 --seed 42 \
    --use-descriptors --use-maccs --batch-norm
uv run python scripts/train.py train --model MLP --split scaffold --epochs 200 --seed 42 \
    --use-descriptors --use-maccs --batch-norm

# ------------------------------------------------------------------
# Krok 3: GNN baseline
# ------------------------------------------------------------------
step 3 "GNN baseline (random + scaffold)"
uv run python scripts/train.py train --model GNN --split random --epochs 200 --seed 42
uv run python scripts/train.py train --model GNN --split scaffold --epochs 200 --seed 42

# ------------------------------------------------------------------
# Krok 4: Ensemble across seeds
# ------------------------------------------------------------------
step 4 "Ensemble across seeds (42, 123, 7, 99)"
for model in MLP GNN; do
    for split in random scaffold; do
        for seed in 42 123 7 99; do
            uv run python scripts/train.py train \
                --model "$model" --split "$split" --epochs 200 --seed "$seed"
        done
    done
done

# ------------------------------------------------------------------
# Krok 5: Modele per-target (opcjonalne)
# ------------------------------------------------------------------
step 5 "Modele per-target (EGFR + p38a)"
if [ -f "$SCRIPT_DIR/per_target.yaml" ]; then
    uv run python scripts/train.py campaign "$SCRIPT_DIR/per_target.yaml"
else
    echo "  Brak pliku per_target.yaml — pominieto."
    echo "  Aby odpalic, stworz plik lub uzyj notebooka (komorka 16)."
fi

# ------------------------------------------------------------------
# Krok 6: Tuning GNN z Optuna
# ------------------------------------------------------------------
step 6 "Tuning GNN scaffold z Optuna (50 triali, wznawialny)"
uv run python scripts/train.py tune \
    --split scaffold --n-trials 50 --seeds 42 123 --epochs 200

# ------------------------------------------------------------------
ELAPSED=$(( $(date +%s) - START_TIME ))
echo ""
echo "=========================================="
echo " Pipeline zakonczony pomyslnie."
echo " Czas: $((ELAPSED / 3600))h $(( (ELAPSED % 3600) / 60 ))m $((ELAPSED % 60))s"
echo "=========================================="
echo " Wyniki:"
echo "   - Notebook: notebooks/learning.ipynb (komorki 12-20)"
echo "   - Optuna:   processed_data/optuna_studies/gnn_scaffold.db"
echo "   - MLflow:   mlflow ui"
echo "=========================================="
