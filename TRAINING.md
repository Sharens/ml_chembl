# Uruchamianie trenowania modeli

## Wymagania

```bash
uv sync
```

## Szybki start — cały pipeline

```bash
bash experiments/run_all.sh
```

Odpala kolejno:
1. MLP baseline (random + scaffold)
2. MLP z deskryptorami + MACCS + BatchNorm
3. GNN baseline (random + scaffold)
4. Ensemble po 4 seedach (42, 123, 7, 99)
5. Tuning GNN scaffold z Optuna (50 triali)

## Pojedyncze komendy

```bash
# MLP
uv run python scripts/train.py train --model MLP --split random

# MLP z deskryptorami molekularnymi + MACCS + BatchNorm
uv run python scripts/train.py train --model MLP --split scaffold \
    --use-descriptors --use-maccs --batch-norm

# GNN
uv run python scripts/train.py train --model GNN --split scaffold --epochs 200

# Tuning GNN z Optuna (wznawialny — można przerwać Ctrl+C i wznowić)
uv run python scripts/train.py tune --split scaffold --n-trials 50

# Podsumowanie wyników tuningu
uv run python scripts/train.py study-summary gnn_scaffold

# Wszystkie kombinacje model/split
uv run python scripts/train.py train-all

# Kampania zdefiniowana w YAML
uv run python scripts/train.py campaign experiments/example_campaign.yaml
```

## Notebook

Do eksploracji wyników otwórz `notebooks/learning.ipynb`.
Wyniki z CLI i notebooka są współdzielone (ten sam `results_table` i cache modeli).

## MLflow

```bash
mlflow ui
```

## Gdzie są wyniki?

| Co | Gdzie |
|---|---|
| Cache modeli | `processed_data/model_cache/` |
| Tuning Optuna | `processed_data/optuna_studies/gnn_*.db` |
| Logi MLflow | `mlflow.db` / `mlruns/` |
| Dane przetworzone | `processed_data/ChEMBL_processed.parquet` |
