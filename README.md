# ML ChEMBL Data Platform

Projekt stanowi platformę danych oraz środowisko analityczne do pracy z bazą chemiczną **ChEMBL**. Łączy w sobie narzędzia do inżynierii danych (ETL), analizy eksploracyjnej (EDA) oraz uczenia maszynowego w celu predykcji aktywności biologicznej związków chemicznych.

## 🚀 Cel projektu

Głównym celem jest zautomatyzowanie pobierania danych o aktywnościach chemicznych (np. IC50) dla konkretnych celów biologicznych (Targets), ich przetwarzanie przy użyciu wydajnych bibliotek (Polars, Spark) oraz budowa modeli uczenia maszynowego.

## 🛠 Technologie i biblioteki

* **Język:** Python 3.12+.
* **Przetwarzanie danych:** `Polars` (szybka alternatywa dla Pandas), `PySpark`.
* **Cheminformatyka:** `RDKit`, `chembl_webresource_client`.
* **ML:** `scikit-learn`, `numpy`.
* **Infrastruktura:** Docker, Docker Compose (Apache Spark, Airflow).
* **Formaty danych:** Parquet (wysoka wydajność odczytu).

## 📂 Struktura repozytorium

```text
ml_chembl/
├── libs/
│   ├── datasets/           # Przetworzone pliki .parquet (np. chembl_selected_ds.parquet)
│   ├── queries/            # Zapytania SQL do bazy SQLite ChEMBL
│   ├── chembl_downloader.py # Skrypt pobierający dane z lokalnej bazy SQLite
│   ├── csv2parquet.py      # Narzędzie do konwersji plików CSV na Parquet
│   └── misc.py             # Funkcje pomocnicze (ładowanie i czyszczenie danych)
├── spark_airflow/          # Konfiguracja środowiska rozproszonego
│   ├── DockerCompose.yml   # Klaster Spark (Master + Workers)
│   └── data_platform/      # Konteneryzacja platformy danych
├── eda.ipynb               # Notebook z eksploracyjną analizą danych i pobieraniem przez API
├── main.ipynb              # Główny proces uczenia maszynowego (preprocessing i modelowanie)
├── requirements.txt        # Lista zależności Python
└── .gitignore              # Ignorowane pliki (środowiska wirtualne, duże bazy danych)

```

## ⚙️ Instalacja i Uruchomienie

### 1. Środowisko lokalne Python

Zaleca się użycie środowiska wirtualnego:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# lub
.venv\Scripts\activate     # Windows

pip install -r requirements.txt

```

### 2. Infrastruktura Docker (Spark)

Aby uruchomić lokalny klaster Spark do przetwarzania dużych zbiorów danych:

```bash
cd spark_airflow
docker-compose -f DockerCompose.yml up -d

```

Spark Master będzie dostępny pod adresem `http://localhost:8080`.

## 📈 Przepływ pracy (Workflow)

1. **Pobieranie danych:** Skrypt `libs/chembl_downloader.py` łączy się z lokalną bazą SQLite ChEMBL i eksportuje wybrane dane do formatu ramki danych.
2. **Analiza EDA:** W notebooku `eda.ipynb` sprawdzane są rozkłady standardowych wartości aktywności oraz generowane są deskryptory chemiczne za pomocą RDKit.
3. **Modelowanie:** Notebook `main.ipynb` wczytuje dane z formatu `.parquet`, dokonuje skalowania cech i trenuje modele predykcyjne.

## 📝 Notatki

* Projekt korzysta z bazy **ChEMBL 36** w formacie SQLite (wymaga pobrania i umieszczenia w `libs/`).
* Wykorzystanie biblioteki `Polars` pozwala na efektywną pracę z milionami rekordów przy niskim zużyciu pamięci RAM.
