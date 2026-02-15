## Główny cel modelu
* Predykcja aktywnoci biologicznej poszczegolnych molekul

## Przetwarzanie danych (pIC50)

Głównym celem jest ujednolicenie i oczyszczenie wartości powinowactwa chemicznego.

* **Zakres wartości:** Należy zachować jedynie rekordy, gdzie  mieści się w logicznym zakresie **3 – 12**. Wartości poza tym przedziałem należy usunąć jako potencjalne błędy lub outliery.
* **Obsługa relacji (Cenzurowanie danych):** W bazie ChEMBL dane  występują w dwóch wariantach, które należy uwzględnić w kolumnie `standard_relation`:
* **Dokładne:** np.  (relacja „=”).
* **Przedziałowe:** np.  (relacja „>”).


* **Zadanie:** Należy dodać kolumnę z relacją do datasetu, aby model wiedział, czy ma do czynienia z dokładną wartością, czy z wartością progową.

## Metryki i Funkcja Straty

Poprawka definicji matematycznych i dobór funkcji celu.

* **Metryki regresji:**
    * **MAE (Mean Absolute Error):**  – mierzy średni błąd bezwzględny.
    * **MSE (Mean Squared Error):**  – silniej karze duże błędy.
    * **RMSE (Root Mean Squared Error):**  – błąd w tych samych jednostkach co cel.


* **Funkcja Straty (Loss Function):**
    * Zaleca się stosowanie **funkcji typu "Robust"** (np. *Huber Loss* lub *Log-Cosh*), które są mniej wrażliwe na outliery (szumy w danych chemicznych) niż standardowe MSE.



## Podział danych (Splitting)

Unikanie pułapek przy dzieleniu na zbiór treningowy i testowy.

* **Metoda Random Split:** Najczęstsza, ale ryzykowna w cheminformatyce. Może prowadzić do zbyt optymistycznych wyników, jeśli podobne cząsteczki trafią do obu zbiorów.
* **Zagrożenia:**
* **Overfitting (Przeuczenie):** Model zapamiętuje szum zamiast trendów.
* **Underfitting (Niedouczenie):** Model jest zbyt prosty, by wyłapać zależności.


* **Dobre praktyki:**
* **Wersjonowanie danych:** Zawsze zapisuj, która wersja datasetu została użyta do danego treningu.
* **Eksperymentowanie z permutacjami:** Testowanie różnych ziarn losowości (seeds), aby upewnić się, że wynik nie jest dziełem przypadku.


## Walidacja i Generalizacja

Jak rzetelnie ocenić model?

* **Cross-Validation (Walidacja krzyżowa):** Podstawowe narzędzie do oceny stabilności modelu.
* **Scaffold Split (RDKit):** Zamiast losowego podziału, należy zastosować partycjonowanie według **szkieletów węglowych (scaffolds)**. Pozwala to sprawdzić, jak model radzi sobie z zupełnie nowymi klasami związków chemicznych.
* **Generalizacja:** Kluczowym testem jest sprawdzenie na zbiorze testowym, czy model potrafi przewidywać właściwości dla struktur, których „nie widział” podczas treningu.



## Mechanizm uczenia sieci neuronowych

Uczenie sieci to proces optymalizacji parametrów w celu zminimalizowania błędu predykcji.

* **Wejście (Input):** Dane wejściowe są reprezentowane jako **wektory cech** (ang. *feature vectors*). W przypadku chemii mogą to być deskryptory cząsteczkowe lub wektory stanów atomów.
* **Proces uczenia (Forward & Backward Propagation):**
* **Forward Pass:** Dane przechodzą przez kolejne warstwy sieci, gdzie wykonywane są operacje mnożenia przez **wagi** () i dodawania **biasu** ().
* **Funkcja aktywacji:** Wprowadza nieliniowość (np. ReLU, Sigmoid), co pozwala sieci uczyć się złożonych wzorców.
* **Loss Function (Funkcja straty):** Mierzy różnicę między wynikiem sieci a rzeczywistością.
* **Backpropagation:** Algorytm oblicza gradienty błędu względem wag i aktualizuje je (zwykle za pomocą optymalizatora typu *Adam* lub *SGD*), aby w kolejnej iteracji błąd był mniejszy.



---

## Reprezentacja cząsteczek jako grafów

W uczeniu maszynowym cząsteczkę chemiczną najskuteczniej reprezentuje się jako **graf nieskierowany**, gdzie atomy to wierzchołki, a wiązania to krawędzie.

### Przykład: Grupa 

Aby przekształcić tę strukturę w postać zrozumiałą dla algorytmu, definiujemy zbiory  (wierzchołki) oraz  (krawędzie).

**Definicja matematyczna grafu :**

* **Zbiór wierzchołków ( - Vertices):** Każdy atom otrzymuje unikalny indeks i etykietę typu atomu.
* : C (Węgiel)
* : N (Azot)
* : O (Tlen)


* **Zbiór krawędzi ( - Edges):** Definiuje połączenia (wiązania) między atomami.
* : Wiązanie między C i N.
* : Wiązanie między N i O.



---

## Wektory stanu węzłów (Node Embeddings)

W grafowych sieciach neuronowych (GNN) każdy węzeł (atom) nie jest tylko statyczną etykietą, ale posiada własny **wektor stanu** ().

* **Inicjalizacja:** Początkowo wektor stanu węzła  zawiera cechy fizykochemiczne atomu (np. liczbę atomową, hybrydyzację, ładunek formalny).
* **Message Passing (Przekazywanie wiadomości):** W procesie uczenia węzeł  (Azot) "rozmawia" ze swoimi sąsiadami ( i ). Wektor stanu jest aktualizowany w oparciu o informacje z otoczenia:


* **Cel:** Po kilku warstwach "przekazywania wiadomości", wektor stanu każdego atomu zawiera informację nie tylko o nim samym, ale o całym jego chemicznym sąsiedztwie.


## Budowa sieci: Od wektorów do modelu

Mając dane w postaci grafu (węzły  i krawędzie ), proces budowy sieci wygląda następująco:

### Krok 1: Inicjalizacja (Wektory stanu i przejścia)

W Twoim przykładzie atomy i wiązania mają już przypisane cechy liczbowe:

* **Wektory stanu węzłów ():** , , . To są "tożsamości" atomów.
* **Wektory przejścia (krawędzi ):** , . To są cechy wiązań (np. typ wiązania: pojedyncze, podwójne).

### Krok 2: Warstwa przekazywania wiadomości (Message Passing)

To tutaj dzieje się "magia" GNN. Sieć uczy się nowych reprezentacji poprzez wymianę informacji:

1. **Agregacja:** Węzeł  (azot) zbiera informacje od sąsiadów ( i ) oraz sprawdza typy wiązań ( i ).
2. **Aktualizacja:** Sieć oblicza nowy wektor stanu dla azotu, który "wie" już, że po lewej ma węgiel, a po prawej tlen.

---

## Sieć Grafowa (GNN) vs Sieć Wektorowa (MLP)

Zrozumienie tej różnicy jest kluczowe dla pracy z danymi chemicznymi.

| Cecha | Sieć Wektorowa (np. MLP) | Sieć Grafowa (GNN) |
| --- | --- | --- |
| **Dane wejściowe** | Stały, "płaski" wektor (np. 2048 bitów fingerprintu). | Dynamiczny graf (różna liczba atomów i połączeń). |
| **Relacje** | Ignoruje strukturę przestrzenną; traktuje cechy niezależnie. | Bezpośrednio modeluje połączenia między atomami. |
| **Niezmienniczość** | Czuła na kolejność cech w wektorze. | **Niezmiennicza względem permutacji** (wynik nie zależy od tego, który atom nazwiemy numerem 1). |
| **Zastosowanie** | Szybka predykcja na podstawie ogólnych deskryptorów. | Precyzyjne modelowanie lokalnych oddziaływań i geometrii. |

---

## Co tak naprawdę się "uczy"?

W sieciach grafowych nie uczymy się samych wektorów stanu (one są danymi), ale **funkcji (wag)**, które:

* Decydują, jak ważne są informacje od konkretnych sąsiadów (**Attention Weights**).
* Przekształcają zebrane informacje w nową, lepszą reprezentację (macierze wag ).
* Agregują dane z całej cząsteczki do jednego wektora końcowego (tzw. **Readout** lub **Global Pooling**), na podstawie którego następuje predykcja (np. aktywność ).


To zagadnienie dotyczy etapu **agregacji** i **readoutu** (odczytu) w sieciach grafowych. W tradycyjnych sieciach (MLP) wektor ma stałą długość. W grafach każda cząsteczka ma inną liczbę atomów, więc musimy te informacje „ścisnąć” do jednej postaci.

Oto jak działają te mechanizmy:

---

## Agregacja lokalna (Message Passing)

Zanim sieć „wypluje” wynik dla całej cząsteczki, każdy węzeł (atom) musi dowiedzieć się czegoś o swoich sąsiadach. Robi to właśnie poprzez operacje matematyczne na wektorach stanu:

* **Sumowanie (Sum):** Nowy stan węzła to suma wektorów jego sąsiadów. Jest to bardzo czułe na **liczbę połączeń** (węzeł z 4 sąsiadami będzie miał „silniejszy” sygnał niż ten z jednym).
* **Średnia (Mean):** Dzielimy sumę przez liczbę sąsiadów. To dobre, gdy interesuje nas **charakter otoczenia**, a nie jego wielkość.
* **Maksimum (Max):** Wybieramy najwyższą wartość z każdej cechy wektora. To pozwala sieci wychwycić **najważniejszą cechę specyficzną** (np. „czy w pobliżu jest choć jeden atom siarki?”).

---

## Global Pooling (Readout) – Tworzenie reprezentacji cząsteczki

Kiedy wszystkie atomy (węzły) już „wymieniły się informacjami”, mamy zbiór zaktualizowanych wektorów stanu. Aby sieć mogła przewidzieć np. , musi zamienić te wszystkie wektory w **jeden wspólny wektor dla całej cząsteczki**.

To tutaj sumujemy węzły z ich wektorami stanu:

Gdzie:

*  to końcowy wektor stanu -tego atomu.
*  to liczba atomów w cząsteczce.

### Dlaczego sumowanie jest lepsze niż średnia w chemii?

W chemii medycznej sumowanie często sprawdza się lepiej, ponieważ **rozmiar cząsteczki ma znaczenie**. Większa cząsteczka może mieć więcej grup funkcyjnych oddziałujących z receptorem. Średnia (Mean Pooling) „rozmywa” tę informację – duża i mała cząsteczka o podobnym składzie procentowym atomów miałyby bardzo podobne wektory średnie, co w ML byłoby błędem.

---

## Co się dzieje z wektorami przejść (krawędziami)?

Wspomniałeś o wektorach przejścia (np. ). W zaawansowanych sieciach (jak **GAT** – Graph Attention Networks lub **MPNN**):

* Wektory krawędzi działają jak **filtry** lub **wagi**.
* Podczas sumowania węzłów, sieć nie tylko dodaje , ale najpierw mnoży  przez funkcję zależną od typu wiązania ().
* Dzięki temu sieć wie, że wiązanie podwójne przekazuje informację „inaczej” niż wiązanie pojedyncze.

---

### Podsumowanie: po co to robimy?

Sumowanie wektorów stanu pozwala nam przejść z **reprezentacji lokalnej** (co wie atom) do **reprezentacji globalnej** (co wiemy o całej cząsteczce). Dopiero ten „zsumowany” wektor trafia do klasycznej sieci gęstej (Linear Layers), która podaje końcową wartość aktywności biologicznej.


W kontekście uczenia maszynowego i przetwarzania danych (szczególnie w chemii i biologii), **smoothing** (wygładzanie) to technika, która ma na celu usunięcie „szumu” z danych i zapobieganie zbyt pewnym siebie (overconfident) predykcjom modelu.


## Label Smoothing (Wygładzanie etykiet)

To najczęstsze znaczenie w uczeniu sieci neuronowych. Zamiast uczyć sieć, że wynik jest „zero-jedynkowy” (np. aktywny = 1, nieaktywny = 0), nieco modyfikujemy etykiety.

* **Problem:** Standardowa funkcja straty (Cross-Entropy) dąży do tego, aby model był w 100% pewny swojej odpowiedzi. To prowadzi do **overfittingu** – model „pamięta” konkretne przykłady zamiast uczyć się wzorców.
* **Rozwiązanie:** Zamiast uczyć na wartościach , uczymy na wartościach takich jak .
* **Efekt:** Model staje się bardziej „pokorny”. Jeśli trafi na cząsteczkę bardzo podobną do treningowej, ale o innej aktywności, nie wyrzuci drastycznie błędnego wyniku. Zwiększa to zdolność sieci do **generalizacji**.

---

## Smoothing w danych grafowych (GNN)

W sieciach grafowych (o których rozmawialiśmy wcześniej) smoothing zachodzi naturalnie podczas sumowania wektorów sąsiadów.

* **Mechanizm:** Każdy kolejny krok agregacji (*Message Passing*) sprawia, że wektor stanu atomu staje się średnią (lub sumą) informacji z coraz dalszego otoczenia.
* **Zagrożenie (Over-smoothing):** Jeśli sieć ma zbyt wiele warstw (np. 50 warstw grafowych), każdy atom w cząsteczce zaczyna mieć niemal **identyczny wektor stanu**. Informacja lokalna zanika, a graf staje się jedną „rozmytą” plamą.
* **Efekt:** Model traci zdolność odróżniania atomu węgla na jednym końcu cząsteczki od tlenu na drugim. Dlatego sieci grafowe zazwyczaj mają tylko kilka warstw (np. 3–8).

---

## Smoothing w danych eksperymentalnych (np. ChEMBL)

W chemii dane  są często obarczone błędem pomiarowym. Smoothing stosuje się tu jako formę regresji, aby wygładzić krzywe dawka-odpowiedź.

* **Zastosowanie:** Zamiast wierzyć każdemu pojedynczemu punktowi pomiarowemu, dopasowuje się funkcję (np. logistyczną), która „wygładza” skoki wynikające z zanieczyszczeń lub błędów aparatury.
* **Efekt:** Uzyskujemy stabilniejszą wartość , która lepiej oddaje rzeczywistą naturę związku.

---

### Podsumowanie: po co to robimy?

Głównym celem smoothingu jest **zwiększenie odporności (robustness)** modelu.

1. **Redukuje wpływ outlierów** (pojedynczych, dziwnych wyników).
2. **Zapobiega overfittingowi** (model nie „fiksuje się” na idealnych wartościach).
3. **Poprawia generalizację** (model lepiej radzi sobie z nowymi cząsteczkami).

Oto poprawiona wersja notatki o wektorze resztkowym, sformatowana tak, aby wzory LaTeX poprawnie renderowały się w środowisku Markdown (np. w GitHub, Obsidian czy Jupyter Notebook).

---

## Wektor resztkowy (Residual Vector)

Wektor resztkowy to różnica między wartością rzeczywistą a przewidzianą przez model. Jest to kluczowe pojęcie w regresji (np. przy przewidywaniu aktywności chemicznej) oraz w budowaniu głębokich sieci neuronowych.

---

###  Wektor resztkowy w regresji (Analiza błędu)

W kontekście przewidywania , reszta () mówi nam o precyzji modelu dla konkretnej cząsteczki.

* **Definicja:**
Jeśli  to wartość rzeczywista (np. z bazy ChEMBL), a  to predykcja modelu, to resztę obliczamy wzorem:

* **Wektor resztkowy:**
To wektor zawierający błędy dla wszystkich próbek w zbiorze danych:


> **Ważne:** Jeśli wykres wektora resztkowego wykazuje wyraźny trend (np. błędy rosną wraz z wielkością cząsteczki), oznacza to, że model jest źle dopasowany (bias).

---

###  Połączenia resztkowe w sieciach neuronowych (Skip Connections)

W głębokich sieciach (ResNets, Deep GNNs) stosuje się tzw. **Residual Learning**. Zamiast uczyć warstwę mapowania wejścia bezpośrednio w wyjście, uczymy ją tylko "poprawki" (reszty).

* **Mechanizm:**
Zamiast standardowego , stosujemy architekturę:


Gdzie:

*  – wektor wejściowy (np. początkowy stan atomu).
*  – przekształcenie wykonane przez warstwy sieci (wyuczona "reszta").
*  – wynik końcowy.

**Dlaczego to stosujemy w chemii?**
W sieciach grafowych (GNN), po wielu krokach agregacji, informacja o pierwotnych cechach atomu (np. jego typie) może ulec rozmyciu (over-smoothing). Dodanie połączenia resztkowego pozwala sieci "pamiętać" pierwotny wektor stanu atomu:


---

###  Związek z funkcją straty

Wektor resztkowy jest bezpośrednio wykorzystywany do obliczania błędu, który sieć musi zminimalizować. Najpopularniejsze funkcje straty to operacje na tym wektorze:

* **MAE** (Mean Absolute Error): Średnia z wartości bezwzględnych wektora resztowego:
* **MSE** (Mean Squared Error): Średnia kwadratów komponentów wektora resztkowego:


Oto poprawiona i uzupełniona notatka, uwzględniająca techniczne aspekty ładowania danych (batching) oraz normalizację w kontekście transformacji Laplace'a.

---

## Ładowanie danych po transformacji Laplace'a do sieci neuronowej

Efektywne trenowanie sieci na danych grafowych (np. z bazy ChEMBL) wymaga odpowiedniego przygotowania struktury danych, aby proces obliczeniowy był wydajny i stabilny.

### 1. Przetwarzanie Batchowe (Batching)

Danych grafowych nie ładujemy do pamięci GPU w całości. Zamiast tego stosujemy **ładowanie batchowe**, które w przypadku grafów różni się od klasycznego ML:

* **Łączenie grafów:** W jednym batchu łączymy wiele małych grafów (cząsteczek) w jeden duży, rozłączny graf blokowo-diagonalny. Pozwala to na równoległe przetwarzanie wielu struktur jednocześnie bez zmiany ich topologii.
* **Zaleta:** Optymalizuje wykorzystanie pamięci VRAM i stabilizuje proces uczenia poprzez uśrednianie gradientów z wielu przykładów.

### 2. Przygotowanie Macierzy i Normalizacja

Kluczowym elementem transformacji Laplace'a jest odpowiednie przygotowanie wektorów przed włożeniem ich do macierzy wejściowej sieci.

* **Znormalizowany Laplasjan:** Zamiast surowej macierzy , najczęściej ładujemy jej formę znormalizowaną symetrycznie:

Zapobiega to eksplozji wartości własnych i pomaga w stabilizacji wag sieci.
* **Wektory Stanu w Macierzy:**
1. Wyliczamy wektory własne (eigenvectors) dla każdej cząsteczki.
2. Poddajemy je **normalizacji** (zazwyczaj do normy ), aby ich skala była spójna niezależnie od rozmiaru cząsteczki.
3. Wkładamy znormalizowane wektory do macierzy cech węzłów ().

### 3. Workflow ładowania danych

Proces ten w bibliotekach takich jak PyTorch Geometric wygląda następująco:

1. **Transformacja:** Obliczenie wektorów własnych Laplasjanu jako *Positional Encodings*.
2. **Dataloader:** Utworzenie obiektu `DataLoader`, który automatycznie grupuje cząsteczki w batche.
3. **Ładowanie:** Przekazanie batcha do modelu, gdzie macierz Laplasjanu steruje przepływem informacji (Message Passing).

> **Pamiętaj:** Podczas ładowania batchowego musisz zadbać o spójność wymiarów. Jeśli cząsteczki mają różną liczbę atomów, wektory własne Laplasjanu są przycinane lub dopełniane zerami (padding) do stałej długości .


---
---

Jak ładować dane po tranfosrmacji LaPlace do sieci neuronowej?
    - DAne nie ładujemy w całości tylko batchowo
    - szykujemy sobie w batchach i ładujemy znormalizowany wektor do macierzy


- Nie, bo 
## TODO:
- Należy podjąc decyzje GNN vs GCN
- każdy ma miec feature engineering zrobiony
