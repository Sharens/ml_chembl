Oto odwzorowana tabela w formacie Markdown na podstawie przesłanego zdjęcia:

### KRYTERIA OCENY (ZALICZENIE PROJEKTU MVP)

**Budujemy kompletny system: Model + Narzędzia + LLM.**

| Ocena | Wymagania Techniczne | Funkcjonalność LLM / Interfejs |
| --- | --- | --- |
| **2.0** | Brak działającego modelu GNN lub brak możliwości uruchomienia kodu. | Brak integracji z LLM lub niedziałający interfejs. |
| **3.0** | Działający model MLP + GNN (bazowy). Trening na dowolnym podziale. | Prosty UI (Streamlit/Gradio). LLM odpowiada tylko tekstem (brak wywołań narzędzi). |
| **3.5** | Model GNN z poprawnym Featurization (RDKit). Zastosowany Scaffold Split. | UI wyświetla 2D strukturę cząsteczki (RDKit). LLM "wie" o istnieniu modelu. |
| **4.0** | GIN z BatchNorm/Dropout. Udokumentowany proces treningu (WandB/logi). | **Function Calling:** LLM potrafi sam wywołać model GNN dla podanego SMILES. |
| **4.5** | Wysoka jakość modelu (AUC > 0.65 na Scaffold). Obsługa błędnych SMILES. | **Tool Use:** LLM wywołuje model ORAZ RDKit (np. do obliczenia masy cząsteczkowej/LogP). |
| **5.0** | Ekspert: AUC > 0.70. Dogłębna analiza błędów modelu (mismatch analysis). | **Agent:** LLM planuje kroki, wizualizuje wyniki i interpretuje predykcję modelu w kontekście chemicznym. |

* *Warunkiem koniecznym na 4.0+ jest udowodnienie, że LLM faktycznie orkiestruje narzędzia, a nie tylko wypisuje tekst.*

---
## 2. Kryteria oceny projektu
Zaliczenie przedmiotu opiera się na stopniu zaawansowania technicznego modelu i interfejsu:

| Ocena | Wymagania techniczne (Model) | Funkcjonalność LLM / UI |
| :--- | :--- | :--- |
| **3.0** | Działający MLP + bazowy GNN (losowy podział danych). | LLM odpowiada tekstem, brak wywoływania narzędzi. |
| **3.5** | GNN z poprawną inżynierią cech (RD-Kit) + **Scaffold Split**. | Wizualizacja 2D cząsteczki; LLM "widzi" wynik modelu. |
| **4.0** | Sieć typu **GIN** + BatchNorm/Dropout + logowanie (MLflow). | **Agent LLM** samodzielnie wywołuje model GNN dla SMILES. |
| **4.5** | Model z AUC $\geq 0.65$ na Scaffold Split; obsługa błędnych SMILES. | LLM wywołuje model oraz dodatkowe narzędzia RD-Kit. |
| **5.0** | AUC $\geq 0.70$ (lub 0.65 przy głębokiej analizie błędów). | LLM planuje kroki, wizualizuje i interpretuje wyniki w kontekście chemicznym. |

---