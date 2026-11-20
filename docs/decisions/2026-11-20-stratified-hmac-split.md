# Decyzja architektoniczna: Deterministyczny podział grup HMAC ze stratyfikacją klasową (R-01)

**Data:** 2026-11-20  
**Status:** Zaakceptowana (Wojciech Bubółka)  
**Dotyczy:** Moduł `grzybobranie_ml.prepare_data` (Etap P1), mitygacja ryzyka **R-01**  
**Powiązane dokumenty:** `DATA_CONTRACT.md`, `docs/p0_inventory.md`, `AGENTS.md`, `docs/soul.md`

---

## 1. Kontekst i zdefiniowanie problemu (Ryzyko R-01)

W prototypowym podejściu (`ml/split.py`) zdjęcia grupowane były w spójne składowe (po autorze i ID obserwacji), a następnie każda składowa otrzymywała deterministyczną pseudolosową wartość z przedziału $[0, 1)$ wyliczaną funkcją skrótu:
$$u = \frac{\text{int}(\text{HMAC-SHA256}(\text{salt}, \text{component\_id})[:8], 16)}{2^{32}}$$
i na tej podstawie trafiała do jednego ze zbiorów:
- $u < 0.70 \implies \text{train}$,
- $0.70 \le u < 0.85 \implies \text{val}$,
- $u \ge 0.85 \implies \text{test}$.

### Wada podejścia naiwnego:
Dla gatunków o małej liczności, w szczególności krytycznego gatunku śmiertelnie trującego ***Amanita verna*** (muchomor wiosenny, $n = 67$ zdjęć w całym zbiorze po włączeniu addendum), liczba unikalnych autorów i obserwacji jest niewielka (szacunkowo kilkanaście grup).

Przy naiwnym podziale istnieje wysokie prawdopodobieństwo, że:
1. Wszystkie lub niemal wszystkie grupy danego gatunku wylosują $u < 0.85$, co spowoduje, że **w zbiorze testowym znajdzie się 0 zdjęć** ($n_{\text{test}} = 0$).
2. Zbiór walidacyjny również może otrzymać znikomą liczbę próbek ($n_{\text{val}} \le 1$).
3. Przy braku próbek w teście ewaluacja bramek wydań (Etap P8), w tym wyliczenie dolnej granicy przedziału ufności Wilsona dla czułości gatunków `high_stakes` ($\ge 0.85$), staje się niemożliwa matematycznie ($n_{\text{test}} \ge 8$ jest warunkiem koniecznym).
4. Próba ręcznego „wymuszania” lub arbitralnego przenoszenia pojedynczych zdjęć narusza nierozerwalność spójnych składowych (wyciek danych) lub psuje determinizm kryptograficzny podziału.

---

## 2. Decyzja projektowa

Zamiast naiwnego progowania hashów $u \in [0, 1)$ wprowadzamy **deterministyczny algorytm przydziału spójnych składowych ze stratyfikacją po gatunku i twardym minimum dla klas `high_stakes`**.

Algorytm gwarantuje łącznie:
1. **Zero wycieku danych (No Leakage):** Żadna spójna składowa (obserwacja lub autor) nie jest dzielona pomiędzy podzbiory. Wszystkie zdjęcia danego osobnika lub fotografa w całości trafiają do jednego zbioru (`train`, `val` albo `test`).
2. **Bezwzględny determinizm (Strict Determinism):** Przydział jest w 100% powtarzalny dla danej soli `split_salt` i danego wejściowego zbioru danych. Nie używa się generatorów liczb pseudolosowych z seedem (`random.seed`), lecz kryptograficznego porządku HMAC.
3. **Twarde minimum dla klas krytycznych:** Dla każdego gatunku `high_stakes` gwarantowane jest co najmniej **$n_{\text{test}} \ge 8$** oraz **$n_{\text{val}} \ge 8$** zdjęć (o ile gatunek spełnia warunek wejścia do głowicy $\ge 55$ zdjęć).
4. **Rozłączność ze zbiorem OOD v1:** Weryfikowana przed przystąpieniem do budowy grafu i podziału.

---

## 3. Szczegółowy algorytm postępowania

### Krok 1: Budowa grafu spójnych składowych (Connected Components)
1. Rekordy zdjęć reprezentowane są jako węzły grafu nieskierowanego $G = (V, E)$.
2. Krawędź między dwoma zdjęciami powstaje, gdy współdzielą:
   - to samo `observation_id` (np. `inat_obs_12345`, `mo_obs_6789`, `gbif_occ_111213`), LUB
   - tego samego autora/obserwatora (`inat_observer`, `author`).
3. Wyznaczane są spójne składowe $\mathcal{C} = \{C_1, C_2, \dots, C_m\}$.
4. Dla każdej składowej $C_i$ wyznaczana jest jej kanoniczna reprezentacja tekstowa:
   $$\text{key}(C_i) = \text{min-hash or sorted-unique-ids}(C_i)$$

### Krok 2: Deterministyczny ranking składowych (HMAC Scoring)
Dla każdej spójnej składowej $C_i$ obliczana jest 64-bitowa wartość skrótu:
$$\text{score}(C_i) = \text{int}(\text{HMAC-SHA256}(\text{split\_salt}, \text{key}(C_i))[:16], 16)$$
Wartość ta definiuje niezmienny, deterministyczny priorytet składowej.

### Krok 3: Kolejność przetwarzania gatunków (Priority Order)
Gatunki sortowane są według priorytetu przydziału:
1. W pierwszej kolejności gatunki **`high_stakes`** o najmniejszej liczbie zdjęć (najbardziej zagrożone niedoborem w teście):
   - `amanita_verna` ($n=67$) $\to$ `inocybe_erubescens` ($n=180$) $\to$ `lepiota_brunneoincarnata` ($n=226$) $\dots$
2. W drugiej kolejności pozostałe gatunki `high_stakes`.
3. Następnie gatunki standardowe, rosnąco po liczbie zdjęć.

### Krok 4: Wyznaczanie celów liczbowych (Quotas)
Dla gatunku $s$ o liczbie zdjęć $N_s$:
- $T_{\text{train}} = \text{round}(0.70 \times N_s)$
- $T_{\text{val}} = \text{round}(0.15 \times N_s)$
- $T_{\text{test}} = N_s - T_{\text{train}} - T_{\text{val}}$

**Reguła twardego minimum dla `high_stakes`:**
Jeśli gatunek jest `high_stakes` i $N_s \ge 55$:
$$T_{\text{test}} = \max(8, T_{\text{test}})$$
$$T_{\text{val}} = \max(8, T_{\text{val}})$$
$$T_{\text{train}} = N_s - T_{\text{val}} - T_{\text{test}}$$

Dla *Amanita verna* ($N=67$):
- $T_{\text{test}} = \max(8, \text{round}(0.15 \times 67)) = \max(8, 10) = 10$
- $T_{\text{val}} = \max(8, \text{round}(0.15 \times 67)) = \max(8, 10) = 10$
- $T_{\text{train}} = 67 - 10 - 10 = 47$ zdjęć.

### Krok 5: Przydział deterministyczny z uwzględnieniem spójności
1. Składowe zawierające zdjęcia gatunku $s$, które nie zostały jeszcze przypisane przez wcześniej procesowane gatunki, są sortowane deterministycznie według $\text{score}(C_i)$.
2. Składowe zapełniają koszyki w kolejności: `test`, `val`, `train`, dążąc do jak najmniejszego błędu kwadratowego względem wyznaczonych kwot liczbowych:
   - Koszyk `test` otrzymuje składowe aż do osiągnięcia kwoty $T_{\text{test}}$ (z gwarancją $\ge 8$ zdjęć dla `high_stakes`).
   - Koszyk `val` otrzymuje składowe aż do osiągnięcia kwoty $T_{\text{val}}$ (z gwarancją $\ge 8$ zdjęć dla `high_stakes`).
   - Pozostałe składowe trafiają do `train`.
3. Jeśli składowa zawiera zdjęcia wielu gatunków, po jej przydzieleniu stan liczników we wszystkich tych gatunkach zostaje zaktualizowany, co uniemożliwia rozerwanie składowej w kolejnych iteracjach.

---

## 4. Wymagane metryki w raporcie audytu (`data_audit.md`)

Zgodnie z wymaganiami Wojciecha, raport generowany przez `prepare_data` musi zawierać dla każdego gatunku:
1. `total_photos` — łączna liczba zdjęć po deduplikacji i filtracji licencji,
2. `unique_observations` — liczba unikalnych obserwacji terenowych,
3. `unique_authors` — liczba unikalnych autorów/obserwatorów,
4. `photos_per_obs_max` — maksymalna liczba zdjęć przypadająca na jedną obserwację,
5. `train_photos` / `val_photos` / `test_photos` — faktyczna liczba zdjęć w poszczególnych splitach,
6. `val_obs_ge2` / `val_obs_ge3` oraz `test_obs_ge2` / `test_obs_ge3` — liczba obserwacji wielozdjęciowych w `val` i `test` (niezbędna do późniejszej kalibracji fuzji wielozdjęciowej w P5).

---

## 5. Konsekwencje i korzyści

1. **Gwarancja wykonalności P8:** Bramki wydań i testy Wilsona dla czułości na *Amanita verna* i innych gatunkach śmiertelnych będą miały wystarczającą liczbę próbek testowych.
2. **Pełna czystość naukowa:** Brak jakiegokolwiek wycieku obserwacji lub autorów między podzbiorami.
3. **Powtarzalność:** Każde uruchomienie `prepare_data` z tą samą solą daje bitowo identyczny podział danych.
