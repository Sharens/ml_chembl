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
uv run streamlit run agent/app.py
```

Aplikacja otworzy się w przeglądarce na `http://localhost:8501`.

## Trenowanie modeli

Trenowanie modeli MLP i GNN odbywa się w notebooku:

```bash
uv run jupyter notebook learning.ipynb
```

Wytrenowane modele są zapisywane w `processed_data/model_cache/` i logowane do
MLflow (`mlflow.db`).

## Struktura projektu

```
ml_chembl/
├── agent/
│   ├── model_inference.py   # Ładowanie modelu + predykcja SMILES→pIC50
│   ├── llm_agent.py         # Agent LLM z tool calling (Ollama)
│   └── app.py               # Streamlit UI
├── data_fetcher/            # Pobieranie danych z ChEMBL
├── data_processing/         # Przetwarzanie i czyszczenie danych
├── learning.ipynb           # Trenowanie MLP i GNN
├── processed_data/
│   ├── model_cache/         # Checkpointy wytrenowanych modeli
│   └── graph_cache/         # Prekompilowane grafy PyG
└── pyproject.toml
```

## Testowanie

```bash
# Test modelu (bez LLM)
uv run python -c "
from agent.model_inference import predict_pic50, load_model
model, _ = load_model()
print(predict_pic50('CC(=O)OC1=CC=CC=C1C(=O)O', model=model))
"

# Test agenta LLM
uv run python -c "
from agent.llm_agent import run_agent
print(run_agent('Predict pIC50 for CC(=O)OC1=CC=CC=C1C(=O)O'))
"
```
