# 29.03.2026 Projektowanie systemów uczenia maszynowego dla chemii lekowej (GNN i LLM)

Wykład koncentruje się na budowie kompletnego systemu typu MVP (Minimum Viable Product), który przewiduje aktywność biologiczną związków chemicznych ($IC_{50}$) na podstawie ich struktury. Kluczowym elementem jest wykorzystanie grafowych sieci neuronowych (GNN) oraz integracja modeli z agentami LLM jako inteligentnymi interfejsami.

---

## 1. Cel i struktura projektu (MVP)
Głównym zadaniem jest stworzenie systemu, który dla danej struktury chemicznej (zapisanej w formacie SMILES) obliczy jej potencjalną aktywność biologiczną.

*   **Zadanie:** Regresja wartości $pIC_{50}$ (ujemny logarytm stężenia hamującego aktywność białka o 50%).
*   **Modele:** Porównanie prostego modelu MLP (Multi-Layer Perceptron) oraz zaawansowanego modelu grafowego (GNN).
*   **Orkiestrator (Agent LLM):** Wykorzystanie lokalnego modelu językowego do zarządzania zapytaniami użytkownika, wywoływania narzędzi (RD-Kit) i interpretacji wyników.

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

## 3. Dlaczego sieci grafowe (GNN)?
Klasyczne sieci (MLP, CNN) zawodzą w chemii z kilku powodów:
*   **Zmienna długość wejścia:** Cząsteczki mają różną liczbę atomów.
*   **Brak inwariancji na permutację:** Zmiana kolejności numeracji atomów w MLP zmienia wynik, mimo że to ta sama cząsteczka.
*   **Nieregularna geometria:** Atomy mają różną liczbę sąsiadów (węgiel w metanie ma 4, w benzenie 3).

**Kluczowe definicje:**
*   **Inwariancja (Niezmienniczość):** Wynik modelu nie zmienia się po obrocie, przesunięciu lub zmianie numeracji węzłów.
*   **Ekwiwariancja:** Jeśli obrócimy wejście, wyjście (np. wektory sił) powinno obrócić się w ten sam sposób.

---

## 4. Reprezentacja grafu w pamięci
Formalnie graf definiujemy jako $G = (V, E)$, gdzie $V$ to zbiór węzłów (atomy), a $E$ to krawędzie (wiązania).

W bibliotece **PyTorch Geometric** stosuje się format **COO (Coordinate Format)** zamiast macierzy sąsiedztwa, co oszczędza pamięć przy rzadkich grafach:

```python
# Przykład reprezentacji listy krawędzi (edge_index) w formacie COO
# Dwie listy: [źródła], [cele]
edge_index = torch.tensor([
    [0, 1, 1, 2], # Węzeł 0 połączony z 1, węzeł 1 z 0, 1 z 2, 2 z 1
    [1, 0, 2, 1]
], dtype=torch.long)
```

**Wektor cech węzła (Node Features):** "Paszport" atomu zawierający m.in.:
*   Typ pierwiastka (One-Hot Encoding).
*   Stopień węzła (liczba sąsiadów).
*   Ładunek formalny i masę atomową.
*   Hybrydyzację i aromatyczność.

---

## 5. Mechanizm Message Passing i jego problemy
Proces uczenia na grafach polega na trzech krokach: **Konstrukcja wiadomości $\to$ Agregacja $\to$ Aktualizacja stanu**.

**Problemy i rozwiązania:**
*   **Over-smoothing:** Przy zbyt wielu warstwach (krokach) wszystkie węzły stają się do siebie podobne (uśrednione). Rozwiązanie: mniejsza liczba warstw (2-3) i normalizacja.
*   **Information Bottleneck:** Trudność w przesyłaniu informacji między odległymi węzłami.
    *   **Rozwiązanie:** **Virtual Node** – sztuczny węzeł połączony ze wszystkimi atomami, służący jako globalna pamięć robocza.
*   **Agregacja (PNA):** Zamiast zwykłej średniej, *Principal Neighbourhood Aggregation* (PNA) używa wielu statystyk (mean, max, min, std) i skalerów, aby lepiej różnicować otoczenie węzła.

---

## 6. Graph Isomorphism Network (GIN)
Rekomendowana architektura ze względu na wysoką ekspresywność (zdolność rozróżniania izomorficznych grafów).

*   **Zasada:** GIN używa **sumy** zamiast średniej w agregacji, co pozwala odróżnić grafy o różnej liczbie węzłów sąsiadujących.
*   **Wydajność:** Często przewyższa bardziej złożone sieci GAT (Graph Attention Networks) w zadaniach chemicznych.

---

## 7. Praktyczna strategia treningu (Mapa drogowa)

1.  **Wybór danych (ChEMBL):** Ogranicz zbiór do aktywności względem jednego konkretnego białka (np. receptor hERG lub białko związane z chorobą Alzheimera). Celuj w 5 000 – 10 000 związków.
2.  **Podział danych:** Zastosuj **Scaffold Split** (według szkieletu cząsteczki), aby sprawdzić, czy model potrafi generalizować wiedzę na zupełnie nowe struktury (uniknięcie *data leakage*).
3.  **Stabilizacja:** Używaj **Batch Normalization** i **Dropout**.
4.  **Learning Rate:** Zastosuj zmienny krok uczenia (scheduler) i zacznij od testu "overfittingu" na bardzo małym podzbiorze danych, aby sprawdzić poprawność architektury.

---

## 8. Podsumowanie wniosków
*   Modele GNN (szczególnie GIN) znacznie przewyższają klasyczne metody MLP oparte na fingerprintach.
*   W predykcji medycznej kluczowa jest zdolność do generalizacji (Scaffold Split), a nie tylko wysoka skuteczność na danych treningowych.
*   Nowoczesne systemy ML powinny być obudowane w interfejsy agentowe (LLM), które pozwalają osobom nietechnicznym (biologom, chemikom) na interakcję z modelem.
*   Osiągnięcie poziomu AUC = 0.70 na podziale typu Scaffold uznaje się za wynik profesjonalny.

**Termin prezentacji projektów:** 13 czerwca.