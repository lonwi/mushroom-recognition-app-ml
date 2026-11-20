# Data Contract — Grzybobranie AI ML Pipeline 🍄📋

**Status:** Obowiązujący (zatwierdzony 2026-11-20)  
**Wersja:** 1.0.0  
**Dotyczy:** Pakiet `mini+` (do 67 gatunków grzybów w Polsce)  
**Zgodność:** `schemas/model-manifest.schema.json` (v1.0.1), `docs/soul.md`, `AGENTS.md`

---

## 1. Cel i zakres kontraktu

Niniejszy dokument stanowi wiążący kontrakt danych pomiędzy etapami przygotowania danych (P1), treningu (P3), kalibracji OOD (P4/P5) oraz eksportu i bramek wydań (P6/P8).

Zastępuje on historyczny, nieaktualny kontrakt prototypu (`ml/DATA_CONTRACT.md`), który zakładał 54 wyjścia z klasami zbiorczymi (`not_a_mushroom`, `unknown_mushroom`) oraz sztuczny limit `class_cap = 2500`.

---

## 2. Architektura głowicy klasyfikatora i obsługa OOD

1. **Czysty klasyfikator całych zdjęć (Whole-image classifier):**
   - Wyjścia softmaxu ($K \le 67$, cel: 67) odpowiadają **wyłącznie konkretnym, biologicznym gatunkom grzybów**.
   - **Całkowity zakaz klas zbiorczych:** ani `not_a_mushroom`, ani `unknown_mushroom` nie mogą znajdować się w wyjściach modelu ani w etykietach treningowych.
2. **Dwustopniowe odrzucanie poza zbiorem (OOD):**
   - **Bramka energii Liu et al. (2020):**  
     $$E(x) = -T_{\text{energy}} \cdot \log \sum_{k=1}^K \exp(z_k / T_{\text{energy}})$$  
     Jeśli $E(x) > \tau_{\text{energy}}$, model zwraca status **`out_of_distribution`**.
   - **Bramka pewności (`minTop1Prob`):**  
     Obliczana na skalibrowanym softmaxie $p = \text{softmax}(z / T_{\text{cal}})$. Jeśli $\max_k p_k < \tau_{\text{prob}}$, model zwraca status **`unclear`**.
3. **Zasada obsługi gatunków z niedoborem danych (`uncoveredSpecies`):**
   - Aby wejść do głowicy klasyfikatora, gatunek musi spełniać twarde progi liczbowe danych treningowych:
     - **$\ge 55$ zdjęć** dla gatunków `high_stakes` / toksycznych,
     - **$\ge 40$ zdjęć** dla pozostałych gatunków.
   - Gatunki z listy 67, które nie spełnią tych progów, **nie wchodzą do głowicy** i trafiają do tablicy `uncoveredSpecies` w manifeście.
   - **Zakaz `in_model: false`:** niedopuszczalne jest tworzenie wyjść w głowicy oznaczonych jako nieaktywne.
   - W aplikacji mobilnej Atlas wyświetla gatunki z `uncoveredSpecies`, ale skaner nigdy ich nie typuje. Jeśli gatunek z `uncoveredSpecies` ma sobowtóra w głowicy, aplikacja wyświetla ostrzeżenie w arkuszu wyników (`uncovered_look_alikes`).

---

## 3. Granice bezpieczeństwa, przechowywania i poufności

1. **Zakaz surowych danych i soli w repozytorium Git:**
   - Zdjęcia źródłowe (`.jpg`, `.png`, `.webp`), archiwa oraz wygenerowane pliki cache (`data/png_cache/`) są objęte `.gitignore` i nie mogą być commitowane do git.
   - Klucz kryptograficzny `split_salt` musi znajdować się poza repozytorium (domyślnie `~/.config/grzybobranie/split_salt` lub zmienna `GRZYBOBRANIE_SPLIT_SALT_PATH`).
2. **Lokalizacja na dysku:**
   - Wszelkie operacje dyskowe, cache oraz katalogi danych (`data/`, `artifacts/`) muszą znajdować się na rodzimym systemie plików Linux (`ext4` w WSL2). Kategoryczny zakaz używania zamontowanych wolumenów Windows (`/mnt/c/`).
3. **Prywatne repozytoria zdalne (Hugging Face):**
   - Repozytorium zbiorów danych: zmienna środowiskowa `HF_DATASET_REPO` (w `.env`: `lonwi/grzybobranie-zdjecia`).
   - Repozytorium modeli i wag: zmienna środowiskowa `HF_MODEL_REPO`.

---

## 4. Schemat metadanych zdjęć (Dataset Manifest Schema)

Każdy rekord zdjęcia przetwarzany w potoku `grzybobranie_ml.prepare_data` musi posiadać znormalizowane metadane:

### 4.1. Pola wymagane (Core Fields)

| Pole | Typ | Opis | Walidacja |
|---|---|---|---|
| `photo_id` | String | Unikalny identyfikator zdjęcia | Unikalny w ramach całego korpusu (prefiksowany źródłem, np. `inat-123456`, `gbif-987654`). |
| `species_slug` | String | Identyfikator gatunku zgodny z `pokrycie-67.csv` | Musi być jednym z 67 dozwolonych slugów (np. `amanita_phalloides`). |
| `image_url` | String | Bezpośredni URL do pliku zdjęcia | Wymagany protokół `https://`. Insecure `http://` jest odrzucane. |
| `author` | String | Autor / właściciel praw autorskich | Niepuste. Wartości zastępcze (`unknown`, `anonymous`, `brak`, `-`) są zakazane. |
| `license` | String | Znormalizowany identyfikator licencji | Musi należeć do białej listy dozwolonych licencji (sekcja 5). |
| `license_url` | String | Oficjalny URL licencji | Wymagany dla licencji Creative Commons (musi prowadzić do deed na `https://creativecommons.org/`). |
| `source_package` | String | Pakiet źródłowy na HF | Jeden z: `v3`, `new16`, `addendum`. |
| `ui_safe` | Boolean | Flaga dopuszczenia do UI aplikacji | `true` tylko dla CC0, CC BY, CC BY-SA i pd_self. `false` dla CC BY-NC. |

### 4.2. Pola grupowania i deduplikacji (Disambiguation & Graph Grouping)

Pola wykorzystywane do budowy grafu spójnych składowych (zapobieganie wyciekowi danych) oraz deduplikacji:

| Pole | Typ | Opis / Rola w grafie |
|---|---|---|
| `observation_id` | String | Zunifikowany identyfikator obserwacji (łączy zdjęcia tego samego osobnika/stanowiska). |
| `inat_observation_id` | String / Null | Identyfikator obserwacji w serwisie iNaturalist. |
| `inat_photo_id` | String / Null | Identyfikator zdjęcia w serwisie iNaturalist. |
| `mo_observation_id` | String / Null | Identyfikator obserwacji w Mushroom Observer. |
| `occurrence_key` | String / Null | Identyfikator rekordu wystąpienia w GBIF. |
| `sha256` | String | Skrót kryptograficzny pliku obrazu (deduplikacja dokładna). |
| `phash` | String | Hash percepcyjny obrazu 64-bit (deduplikacja wizualna / kadry). |

---

## 5. Zasady licencyjne i czystość prawna (Licensing Policy)

### 5.1. Dozwolone licencje

| Licencja | Kategoria | Użycie w treningu | Użycie w UI aplikacji (`ui_safe`) |
|---|---|:---:|:---:|
| `CC0 1.0` | Public Domain | ✅ Tak | ✅ Tak |
| `CC BY 1.0` .. `4.0` | Attribution | ✅ Tak | ✅ Tak |
| `CC BY-SA 1.0` .. `4.0` | Share-Alike | ✅ Tak | ✅ Tak |
| `pd_self` | Własne domeny publicznej | ✅ Tak | ✅ Tak |
| `CC BY-NC 1.0` .. `4.0` | Non-Commercial | ✅ Tak | ❌ **Zakazane** (`ui_safe: false`) |

### 5.2. Bezwzględny zakaz włączania (Prohibited Assets)

1. **Licencje ND (NoDerivatives):** Wszystkie warianty `CC BY-ND` oraz `CC BY-NC-ND` są bezwzględnie odrzucane.
2. **Zbiory i wagi DF20 / DF24:** Kategoryczny zakaz używania zdjęć oraz wag pochodzących z Danish Fungi 2020 / 2024 (w tym modeli timm z oznaczeniem `*ft_df20*`).
3. **Sztuczna inteligencja i grafika:** Kategoryczny zakaz włączania obrazów generowanych przez AI, rycin, rysunków, schematów czy ilustracji z atlasów (dozwolone wyłącznie autentyczne fotografie terenowe owocników).
4. **Wagi bazowe (Backbones):** Dozwolone wyłącznie wagi o licencjach czysto komercyjnych: `Apache-2.0`, `MIT`, `BSD`. Wagi na licencji `CC BY-NC 4.0` (np. ConvNeXt V2) oraz licencjach Apple (MobileViT, FastViT) są domyślnie zablokowane.

### 5.3. Klauzula komercjalizacji (Ryzyko R-06)
Zbiór treningowy zawiera ok. 46,5% zdjęć na licencji `CC BY-NC`.  
* Stan obecny: Aplikacja jest w 100% darmowa (bez reklam i mikropłatności), co jest w pełni zgodne z licencją NC.  
* Wymóg przyszłościowy: W przypadku podjęcia decyzji o komercjalizacji aplikacji, model musi zostać powtórnie wytrenowany wyłącznie na podzbiorze licencji komercyjnych (`CC0`, `CC BY`, `CC BY-SA`, `pd_self`) lub po uzyskaniu indywidualnych licencji komercyjnych.

---

## 6. Strategia podziału danych i zapobieganie wyciekom (Split Contract)

Podział zbioru realizowany jest w proporcjach **70% Train / 15% Validation / 15% Test** z zachowaniem bezwzględnego braku wycieku danych (Zero-Leakage Guarantee):

1. **Grupowanie spójnymi składowymi (Connected Components):**
   - Wszystkie zdjęcia powiązane tym samym autorem (`author` / `inat_observer`) lub tą samą obserwacją (`observation_id`, `inat_observation_id`, `mo_observation_id`, `occurrence_key`) tworzą nierozerwalną grupę (spójną składową).
   - Cała grupa trafia deterministycznie do jednego i tylko jednego podzbioru (`train`, `val` albo `test`).
2. **Determinizm oparty na HMAC:**
   - Przypisanie grup bazuje na skrócie HMAC-SHA256 z użyciem sekretnej soli `split_salt`.
3. **Deterministyczna stratyfikacja i twarde minimum dla `high_stakes` (Ryzyko R-01):**
   - Dla gatunków krytycznych o małej liczności (szczególnie *Amanita verna*, $n=67$), algorytm podziału stosuje deterministyczną stratyfikację grup, gwarantującą twarde minimum:
     - **Minimum 8 unikalnych zdjęć w zbiorze testowym** ($n_{\text{test}} \ge 8$) dla każdego gatunku `high_stakes`. Jest to warunek konieczny do wyliczenia miarodajnego dolnego przedziału ufności Wilsona w bramkach wydań.
4. **Rozłączność ze zbiorem OOD v1:**
   - Wszystkie identyfikatory, URL-e i hashe ze zbioru `data/ood-v1/ood_ids.txt` są bezwzględnie wykluczane z podziałów treningowych i ewaluacyjnych.
5. **Metryki fuzji wielozdjęciowej:**
   - Podczas podziału zliczane są obserwacje zawierające $\ge 2$ oraz $\ge 3$ zdjęcia w zbiorach `val` i `test`, w celu zapewnienia odpowiedniej próby do kalibracji fuzji (P5).

---

## 7. Bramki jakościowe wydań (Ship Gates Contract)

Końcowy model wyeksportowany do formatu `.tflite` podlega automatycznej weryfikacji w module `grzybobranie_ml.ship_gates` według twardych kryteriów:

1. **`confidentToxicAsEdible == 0`:** Zero przypadków, w których gatunek śmiertelnie trujący został z wysoką pewnością zaklasyfikowany jako jadalny.
2. **Błąd w parach sobowtórów $\le 0.05$:** Odsetek pomyłek w krytycznych parach z `configs/karty_sobowtory.json` nie może przekraczać 5%.
3. **`highStakesRecallMin >= 0.85`:** Czułość (Recall) dla żadnego z gatunków wysokiego ryzyka nie może spaść poniżej 85%.
4. **`tfliteAgreement >= 0.99`:** Zgodność predykcji top-1 pomiędzy modelem PyTorch a wyeksportowanym modelem LiteRT `.tflite` na zbiorze testowym musi wynosić co najmniej 99%.
5. **Kalibracja OOD i ECE:** Błąd kalibracji $ECE \le 0.10$ na zbiorze testowym.
