## I. Wstępna eksploracja
    1. Załadowanie i pierwszy rzut oka na dane
    2. Sprawdzenie podstawowych statystyk (.describe(), .info())
    3. Identyfikacja braków danych, duplikatów, outlierów
    4. Stworzenie 2-3 wizualizacji (rozkłady, korelacje)

 
## II. Inżynieria cech
    1. Które cechy wymagają transformacji? (np. skalowanie, kodowanie, transformacja logarytmiczna)
    2. Czy trzeba tworzyć nowe cechy? (agregacje, sumowanie, dociąganie z zewnętrznych baz)
    3. Co z brakującymi wartościami?
    4. Jaką miarę użyjecie do oceny ważności cech?
    
 
Co dostarczyć?
Notebook z kodem i wnioskami
Lista finalnych cech z uzasadnieniem
2-3 kluczowe spostrzeżenia z analizy

---
# Kontynuacja przedmiotu
## Notatka: Budowa modeli baseline (Warsztaty SI)

Celem zajęć jest stworzenie punktów odniesienia (baseline) dla regresji aktywności biologicznej ($IC_{50}$) związków chemicznych.

---

### 1. Model MLP (Multi-Layer Perceptron)

Podejście klasyczne oparte na deskryptorach molekularnych.

* **Dane wejściowe:** Formuły SMILES zamienione na **Morgan fingerprints** (przy użyciu RDKit).
* **Architektura:** `nn.Sequential` (PyTorch).
    * Trzy warstwy liniowe z aktywacją **ReLU** i warstwą **Dropout**.
    * Przykładowy schemat: $2048 \to 512 \to 128 \to 1$.


* **Konfiguracja:**
    * **Loss:** MSE (Mean Squared Error).
    * **Optymalizator:** Adam.
    * **Do ustalenia:** Inicjalizacja wag oraz współczynnik uczenia (learning rate).



---

### 2. Model GNN (Graph Neural Network)

Podejście grafowe, gdzie molekuła to graf: węzły (atomy) i krawędzie (wiązania).

* **Cechy węzłów:** Numer atomowy, ładunek itp.
* **Architektura:** 2-3 warstwy `GCNConv`.
* Schemat: `GCNConv(in, 64) -> ReLU` $\to$ `GCNConv(64, 64) -> ReLU` $\to$ `GCNConv(64, 64) -> ReLU`.
* **Agregacja:** `global_mean_pool` (przejście z poziomu atomów do poziomu całej molekuły).
* **Linear Head:** `Linear(64, 32) -> ReLU` $\to$ `Linear(32, 1)`.


* **Konfiguracja:** Optymalizator Adam, loss MSE.

---

### 3. Metodyka i Ewaluacja

* **Podział danych (Splitting):**
* Random split: 80/10/10.
* **Scaffold split:** Podział oparty na rdzeniach strukturalnych (RDKit) – ważny dla sprawdzenia generalizacji modelu.


* **Logika porównania:**
1. Czy MLP jest lepsze od średniej (podejście naiwne)?
2. Czy GNN (struktura grafu) daje lepsze wyniki niż MLP (fingerprinty)?


* **Narzędzia:**
* **MLflow:** Do zarządzania cyklem życia modelu i logowania wyników.


* **Raportowanie:** Wyniki należy zebrać w tabeli:
| Model | Typ Splitu | Funkcja Loss | $R^2$ |
| :--- | :--- | :--- | :--- |
| MLP | Random | MSE | ... |
| GNN | Scaffold | ... | ... |

---

### Źródła i materiały:

* Dokumentacja: [PyTorch Sequential](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/2.7.0/modules/nn.html).
* Teoria GNN: [Distill.pub - Intro to GNN](https://distill.pub/2021/gnn-intro/).
* Literatura: *Grafowe sieci neuronowe. Teoria i praktyka* (F. Wójcik, 2026).

## Na następne zajęcia:
- wytrenowane modele baseline'owe (MLP, GNN dla różnych splitów)
- przygotowana tabela porównująca modele

---

**Instalacja `uv` i zarządzanie środowiskiem (Ubuntu)**

1) Instalacja `uv` (zalecane: instalator) — w terminalu:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# zamknij i otwórz powłokę lub wykonaj wskazaną komendę (np. `source ~/.profile`)
uv --version
```

Alternatywa — `pipx`:

```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
exec $SHELL
pipx install uv
uv --version
```

2) W katalogu projektu (gdzie jest `pyproject.toml`) wygeneruj plik blokujący:

```bash
# wygeneruje lub zaktualizuje uv.lock na podstawie pyproject.toml
uv lock
```

3) Utworzenie wirtualnego środowiska i zainstalowanie zależności:

```bash
# utworzy .venv
uv venv
# zsynchronizuje środowisko z uv.lock i zainstaluje pakiety
uv sync
# aktywacja venv
source .venv/bin/activate
```

Jeśli nie masz jeszcze `pyproject.toml`, możesz zainicjować projekt:

```bash
uv init .
```

Uwagi:
- `uv lock` tworzy `uv.lock` (cross-platform lockfile) — powinien trafić do kontroli wersji.
- `uv sync` tworzy/aktualizuje `.venv` i instaluje zablokowane wersje.
- W przypadku problemów z instalacją pakietów spróbuj instalatora (pierwsza metoda) lub `pipx`.

---

## MLflow (ocena 4.0)

Notebook `learning.ipynb` ma przygotowane logowanie pełnych treningów do MLflow.

1) Uruchom UI MLflow lokalnie:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

2) W notebooku uruchom komórki treningowe z `log_mlflow=True` (MLP random/scaffold oraz finalne GNN).

3) Otwórz panel:

```text
http://127.0.0.1:5000
```

Domyślna nazwa eksperymentu to `ml_chembl_baselines`, a artefakty trafiają do `mlruns/`.

Jeśli nie widzisz runów, uruchom UI z absolutną ścieżką (eliminuje problem innego katalogu roboczego):

```bash
mlflow ui --backend-store-uri sqlite:////home/computer/Repositories/ml_chembl/mlflow.db --port 5000
```
