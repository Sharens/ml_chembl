# AGENT.md - instrukcje dla agentow AI (OpenCode, Codex, itp.)

Ten dokument definiuje, jak agent ma realizowac zadania w projekcie MVP dla chemii lekowej (GNN + LLM).

## 1) Cel projektu

Zbuduj system MVP, ktory dla wejscia `SMILES` przewiduje aktywnosc biologiczna jako `pIC50` (regresja).

Agent ma:
- rozwijac i porownywac modele `MLP` oraz `GNN`,
- preferowac architektury grafowe (szczegolnie `GIN`) dla danych molekularnych,
- utrzymywac interfejs LLM jako orkiestrator wywolan narzedzi i interpretacji wynikow.

## 2) Priorytety merytoryczne

1. Generalizacja > wynik na treningu.
2. Obowiazkowy podzial danych: `Scaffold Split` (unikaj data leakage).
3. Docelowa architektura: `GIN` + `BatchNorm` + `Dropout`.
4. Cechy atomowe buduj przez `RDKit` (typ atomu, stopien, ladunek, masa, hybrydyzacja, aromatycznosc).
5. Graf reprezentuj w stylu PyTorch Geometric (`edge_index` w formacie COO).

## 3) Standard pracy agenta (workflow)

Agent powinien wykonywac zadania w kolejnosci:
1. Zweryfikuj problem i dane: jeden cel biologiczny (np. hERG), preferowany zakres 5k-10k zwiazkow.
2. Przygotuj pipeline danych z walidacja `SMILES` i obsluga blednych rekordow.
3. Uruchom baseline (`MLP`) i baseline grafowy (`GNN`).
4. Przejdz do modelu docelowego (`GIN`) i stabilizacji treningu.
5. Dodaj scheduler learning rate i test overfittingu na malym podzbiorze.
6. Mierz wyniki na `Scaffold Split` i raportuj metryki.
7. Integruj wynik modelu z agentem LLM, ktory wywoluje predykcje i narzedzia chemiczne.

## 4) Wymagania wobec modelowania grafowego

- Pilnuj niezmienniczosci permutacyjnej reprezentacji grafu.
- Ogranicz liczbe warstw message passing (typowo 2-3), aby zmniejszac over-smoothing.
- Dla problemu information bottleneck rozwaz `Virtual Node`.
- Jesli agregacja jest za slaba, rozwaz podejscie `PNA` (mean/max/min/std + skalery).
- W `GIN` stosuj agregacje przez sume (nie srednia), gdy priorytetem jest ekspresywnosc.

## 5) Wymagania wobec warstwy LLM/agentowej

Agent LLM ma umiec:
- przyjmowac `SMILES`,
- uruchamiac model predykcyjny,
- uruchamiac narzedzia `RDKit` (co najmniej walidacja i wizualizacja 2D),
- zwracac wynik liczbowy + interpretacje chemiczna z zaznaczeniem niepewnosci,
- obslugiwac bledne `SMILES` i proponowac poprawke/diagnoze.

## 6) Kryteria jakosci (mapa ocen)

Minimalnie do zaliczenia:
- `3.0`: dzialajacy `MLP` + bazowy `GNN` (losowy split), prosty interfejs tekstowy.

Poziom docelowy projektu:
- `3.5`: poprawne cechy `RDKit` + `Scaffold Split`, wizualizacja 2D.
- `4.0`: `GIN` + `BatchNorm/Dropout` + logowanie eksperymentow (np. `MLflow`), LLM sam wywoluje model.
- `4.5`: `AUC >= 0.65` na `Scaffold Split`, pelna obsluga blednych `SMILES`, LLM wywoluje dodatkowe narzedzia.
- `5.0`: `AUC >= 0.70` (lub >=0.65 z poglebiona analiza bledow), LLM planuje kroki i interpretuje wyniki kontekstowo.

## 7) Zasady raportowania wynikow przez agenta

Kazdy raport powinien zawierac:
- konfiguracje danych i splitu (`Scaffold Split` vs inne),
- architekture modelu i kluczowe hiperparametry,
- metryki walidacyjne/testowe,
- krotka analize bledow i ograniczen,
- rekomendacje kolejnych iteracji.

Nie raportuj "sukcesu" bez metryk i bez jasnego opisu splitu danych.

## 8) Definicja ukonczenia zadania (DoD)

Zadanie jest zakonczone, gdy:
- pipeline przyjmuje `SMILES` i zwraca predykcje `pIC50`,
- ewaluacja jest wykonana na `Scaffold Split`,
- agent LLM potrafi wywolac model i narzedzia `RDKit`,
- bledne `SMILES` sa obslugiwane, a uzytkownik dostaje czytelny komunikat,
- wyniki sa zarejestrowane i porownywalne miedzy eksperymentami.

## 9) Kontekst organizacyjny

Termin prezentacji projektu: **13 czerwca**.

Przy decyzjach technicznych wybieraj rozwiazania, ktore maksymalizuja szanse dojscia do poziomu profesjonalnego (`AUC ~0.70` na `Scaffold Split`) i sa czytelne dla koncowego uzytkownika (chemik/biolog).
