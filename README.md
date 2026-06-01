# ml_chembl — Bioactivity Predictor (GIN + LLM Agent)

System do przewidywania aktywności biologicznej (pIC50) związków chemicznych na
podstawie struktury (SMILES). Wykorzystuje sieć **GIN** (Graph Isomorphism
Network) z PyTorch Geometric oraz **agenta LLM** (Gemma4 przez Ollama) jako
inteligentny interfejs z tool calling.

## Wymagania

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) z modelem `gemma4:e4b`

## Szybki start

```bash
# 1. Pobierz model do Ollamy (jeśli nie masz)
ollama pull gemma4:e4b

# 2. Uruchom Ollamę w tle
ollama serve &

# 3. Zainstaluj zależności
uv sync

# 4. Uruchom aplikację Streamlit
uv run streamlit run src/agent/app.py
```

Aplikacja otworzy się w przeglądarce na `http://localhost:8501`.

## Trenowanie modeli

Kod treningowy znajduje się w `src/models/`, podzielony na osobne moduły:

| Moduł | Opis |
|---|---|
| `config.py` | Domyślne parametry (epochs, lr, batch size) oraz stałe dla GNN (listy atomów, wiązań, hybrydyzacji) |
| `models.py` | `MLPBaseline` (Morgan fingerprint → MLP) i `GNNRegressor` (GINEConv + BatchNorm) |
| `data.py` | Generowanie fingerprintów Morgana, split scaffold/random, budowa grafów PyG, fabryki DataLoaderów |
| `training.py` | `seed_everything`, `train_one_epoch`, `evaluate_loss`, `evaluate_r2` |
| `pipeline.py` | `train_and_score` (trening + early stopping + cache + MLflow) i `tune_gnn` (grid search) |

### Uruchomienie w notebooku

```bash
uv run jupyter notebook notebooks/learning.ipynb
```

Notebook importuje funkcje z `src.models` i służy wyłącznie do konfiguracji
eksperymentów:

```python
from src.models import train_and_score

df = pl.read_parquet("processed_data/ChEMBL_processed.parquet")
df_clean = ...  # filtracja fingerprintów

train_and_score(model_type="MLP", split_type="random", df_fp=df_clean, log_mlflow=True)
```

### Użycie w kodzie

```python
from src.models.pipeline import train_and_score, tune_gnn
from src.models.data import build_mlp_loaders, fp_from_smiles
from src.models.models import MLPBaseline, GNNRegressor
from src.models.training import seed_everything, get_device
```

### Grid search

```python
search_space = {
    "lr": [1e-3, 3e-4],
    "gnn_hidden_dim": [128, 192],
    "gnn_num_layers": [3, 4],
    "gnn_dropout": [0.1, 0.15],
    "weight_decay": [1e-5],
    "pooling": ["mean", "add"],
}
df_tuning = tune_gnn("scaffold", search_space, seeds=[42, 43])
```

### Wyniki

- Modele cache'owane w `processed_data/model_cache/` (automatyczne wczytywanie
  przy powtórnym uruchomieniu z tymi samymi parametrami)
- Eksperymenty logowane do MLflow (`mlflow.db`) z histogramami epok i metrykami
- Globalny rejestr `results_table` przechowuje wyniki z bieżącej sesji dla
  szybkiego porównania

## Struktura projektu

```
ml_chembl/
├── src/
│   ├── agent/
│   │   ├── model_inference.py   # Ładowanie modelu + predykcja SMILES→pIC50
│   │   ├── llm_agent.py         # Agent LLM z tool calling (Ollama)
│   │   └── app.py               # Streamlit UI
│   ├── data_processing/         # Pobieranie, przetwarzanie i czyszczenie danych
│   ├── models/                   # Trenowanie MLP i GNN
│   │   ├── config.py
│   │   ├── models.py
│   │   ├── data.py
│   │   ├── training.py
│   │   └── pipeline.py
│   └── mlflow_utils.py          # Logowanie eksperymentów MLflow
├── notebooks/
│   ├── learning.ipynb           # Trenowanie MLP i GNN
│   └── eda.ipynb                # Eksploracyjna analiza danych
├── processed_data/
│   ├── model_cache/         # Checkpointy wytrenowanych modeli
│   └── graph_cache/         # Prekompilowane grafy PyG
└── pyproject.toml
```

## Testowanie

```bash
# Test modelu (bez LLM)
uv run python -c "
from src.agent.model_inference import predict_pic50, load_model
model, _ = load_model()
print(predict_pic50('CC(=O)OC1=CC=CC=C1C(=O)O', model=model))
"

# Test agenta LLM
uv run python -c "
from src.agent.llm_agent import run_agent
print(run_agent('Predict pIC50 for CC(=O)OC1=CC=CC=C1C(=O)O'))
"
```
