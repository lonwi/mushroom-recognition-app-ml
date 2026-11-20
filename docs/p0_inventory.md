# Raport Inwentaryzacji i Rozruchu Środowiska (Etap P0)

**Data sporządzenia:** 2026-11-20  
**Strefa czasowa:** Europe/Warsaw  
**Projekt:** Grzybobranie AI — Standalone ML Pipeline (`grzybobranie-ml` v0.1.0)  
**Autor:** Trenowanie modelu Grzybobranie AI (asystent Wojciecha)  
**Status zatwierdzenia:** Gotowy do przeglądu przez Wojciecha  

---

## 1. Wstęp i cele Etapu P0

Zgodnie z konstytucją bota (`docs/soul.md` §2.1, §4) oraz planem etapów (`docs/prompty.md` P0), niniejszy dokument stanowi kompleksowy raport z jednorazowego rozruchu środowiska obliczeniowego, audytu istniejących komponentów i prototypu `ml/`, inwentaryzacji zbiorów danych z repozytorium Hugging Face oraz rejestru ryzyk przed przystąpieniem do Etapu P1 (`prepare_data.py`).

Wszystkie wiążące decyzje projektowe podjęte 2026-11-10 i doprecyzowane 2026-11-16 zostały uwzględnione w konfiguracji i weryfikacji.

---

## 2. Inwentaryzacja środowiska sprzętowo-programowego

Weryfikacja środowiska obliczeniowego została przeprowadzona na natywnej maszynie zgodnie z wymaganiami `docs/soul.md` §4:

### 2.1. System operacyjny i system plików
* **Dystrybucja:** Ubuntu 26.04 LTS (x86_64 Linux 6.6.x-microsoft-standard-WSL2).
* **Środowisko:** WSL2 w systemie Windows 11.
* **System plików:** Natywny Linux `ext4`.
* **Katalog roboczy sesji:** `/home/bubolka/.gemini/antigravity/worktrees/mushroom-recognition-app-ml/analyze_project_state` (jest to dedykowany **git worktree** repozytorium `mushroom-recognition-app-ml` na gałęzi `analyze_project_state`, zsynchronizowany z głównym repozytorium na ext4).
* **Weryfikacja `/mnt/*`:** Potwierdzono brak uruchamiania repozytorium i danych na partycjach Windows DrvFS/9p (`/mnt/c`), co eliminuje spadki wydajności I/O podczas buforowania i treningu.

### 2.2. Akcelerator GPU i sterowniki
* **Karta graficzna:** NVIDIA GeForce RTX 5070 Laptop GPU (8 151 MiB VRAM / 8 GB).
* **Architektura:** Blackwell, compute capability `(12, 0)` (`sm_120`).
* **Wersja sterownika NVIDIA:** Driver Version 592.27 (instalowany wyłącznie na poziomie Windows hosta; w WSL2 nie instalowano zbędnych sterowników linuxowych `cuda-drivers`, co chroni mapowanie bibliotek CUDA z hosta).
* **Wersja CUDA sterownika:** CUDA 13.1 (zgłaszana przez `nvidia-smi 590.70`).

### 2.3. Python i środowisko wirtualne (`.venv`)
* **Python:** CPython 3.14.4 (`/usr/bin/python3.14`).
* **Wirtualne środowisko:** Utworzone wewnątrz repozytorium pod `.venv/` (`Python 3.14.4`, `pip 25.1.1`).
* **Instalacja zależności:** Wykonano pomyślnie `pip install -e ".[dev]"`. Zainstalowano m.in. `timm==1.0.31`, `numpy==2.5.5`, `scipy==1.18.1`, `pillow==13.0.0`, `huggingface-hub==2.4.0`, `jsonschema==4.26.0`, `pytest==9.2.0`, `ruff==0.17.6`.

### 2.4. Weryfikacja PyTorch i CUDA `sm_120`
* **Wersja PyTorch:** `2.15.0+cu132` (oficjalne koła ze zintegrowanym środowiskiem uruchomieniowym CUDA 13.2).
* **Dostępność CUDA (`torch.cuda.is_available()`):** `True`.
* **Nazwa urządzenia (`torch.cuda.get_device_name(0)`):** `NVIDIA GeForce RTX 5070 Laptop GPU`.
* **Wspierane architektury (`torch.cuda.get_arch_list()`):** `['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']` — obecność `sm_120` potwierdzona.
* **Wsparcie BF16 (`torch.cuda.is_bf16_supported()`):** `True`.
* **Test obliczeniowy CUDA:** Przetestowano mnożenie losowych tensorów na GPU (`torch.randn(1024, 1024, device='cuda') @ ...`). Suma kontrolna: `46739.34765625`. Brak błędów kernela CUDA.

### 2.5. Klucz kryptograficzny podziału (`split_salt`)
* **Ścieżka:** `~/.config/grzybobranie/split_salt` (poza drzewem git).
* **Fingerprint SHA-256:** `88dcce3d`.
* **Weryfikacja `.env`:** Utworzono lokalny `.env` z `.env.example`:
  ```bash
  HF_DATASET_REPO=lonwi/grzybobranie-zdjecia
  HF_MODEL_REPO=
  HF_TOKEN=
  GRZYBOBRANIE_SPLIT_SALT_PATH=~/.config/grzybobranie/split_salt
  ```
  Plik `.env` jest ignorowany w `.gitignore` i nie trafi do repozytorium.

### 2.6. Autentykacja Hugging Face
* **Status `hf whoami`:** Zalogowano jako użytkownik `lonwi`.
* **Dostęp do `HF_DATASET_REPO` (`lonwi/grzybobranie-zdjecia`):** Potwierdzono uprawnienia do odczytu prywatnego repozytorium datasetu (w repozytorium wylistowano 80 326 obiektów, w tym 865 plików w katalogu `data/`).

---

## 3. Audyt komponentów repozytorium i status modułów

### 3.1. Zaimplementowane komponenty w v0.1.0
1. **`src/grzybobranie_ml/recognition_math.py`:**
   - Referencyjna implementacja bramki energii OOD ($E(x) = -T \cdot \text{logsumexp}(z/T)$) oraz progu `minTop1Prob`.
   - Ścisła kolejność bramek (energia przed `minTop1Prob`).
   - Obsługa temperatur: `calibration.temperature` dla softmax i wag fuzji, `ood.energyTemperature` (z fallbackiem) dla energii.
   - Entropijne ważenie fuzji 1–3 zdjęć (`_fusion_weights`), obcinanie residuum numerycznego (`c_v = clamp(c_v, 0)`), równy podział przy $\Sigma c \le 10^{-15}$.
   - Zwracanie listy `candidates` (top-3 z indeksem, classId, prawdopodobieństwem i rangą) wyłącznie przy statusie `ok`.
2. **`src/grzybobranie_ml/make_p2_fixtures.py`:**
   - Generator deterministycznych zestawów testowych: `decision_cases.json` i `fusion_cases.json`.
3. **Zestaw testów (`tests/test_recognition_math.py`, `tests/test_p2_fixtures.py`):**
   - **Wynik pytest:** 25/25 testów zaliczonych ze 100% zielonym statusem (czas wykonania: 0.11 s).
   - Pełna zgodność z sumami SHA-256 w `fixtures/p2/SHA256SUMS`.
4. **`configs/karty_sobowtory.json`:**
   - Kompletny plik kart sobowtórów (3 443 wiersze) dla 67 gatunków mini+.
   - Zaktualizowano metadane `_meta` po akceptacji Wojciecha:
     - `"status": "zatwierdzone_przez_Wojciecha"`
     - `"zatwierdzone": true`
     - `"nie_zatwierdzone": false`
     - `"data": "2026-11-20"`
     - `"zatwierdzil": "Wojciech"`
5. **`configs/pokrycie-67.csv`:**
   - Oficjalna tabela pokrycia danych dla 67 gatunków (51 klas z `_v3`, 14 z `new16`, 2 z `new16+addendum`).
6. **`schemas/model-manifest.schema.json`:**
   - Oficjalny kontrakt manifestu v1.0.1 z doprecyzowaniami z 2026-11-16.

### 3.2. Brakujące moduły potoku (planowane na etapy P1–P9)
| Moduł | Etap docelowy | Planowana rola |
|---|---|---|
| `grzybobranie_ml.prepare_data` | **P1** | Ingest manifestów CSV z HF, weryfikacja licencji, deduplikacja (dokładna i perceptualna), odrzucenie OOD, podział 70/15/15 HMAC z grupami obserwacji/autorów, generowanie `data_audit.md` i `leak_report.json`. |
| `grzybobranie_ml.local_drop` | **P1 / Wsparcie** | Obsługa lokalnych zrzutów zdjęć i weryfikacja metadanych CSV w `data/local_drop/`. |
| `grzybobranie_ml.cache_manager` | **P1 / P2** | Przygotowanie i buforowanie znormalizowanych PNG w rozdzielczościach 224×224 i 256×256 w `data/png_cache/`. |
| `grzybobranie_ml.train` | **P3** | Trening i fine-tuning PyTorch + `timm` (zamrożony backbone $\to$ pełny fine-tuning, mixed precision bf16, EMA, class-balanced sampling, lokalne logowanie Trackio, zapis do `artifacts/checkpoints/`). |
| `grzybobranie_ml.calibrate_ood` | **P4 / P5** | Kalibracja temperaturą (L-BFGS / NLL), obliczanie ECE/MCE na val i teście, wyznaczanie progów bramki energii OOD oraz progów fuzji na zbiorze `data/ood-v1/`. |
| `grzybobranie_ml.export_tflite` | **P6** | Eksport modeli do formatu `.tflite` przez `litert-torch` (fp32, fp16, int8 PTQ) do `artifacts/exports/`. |
| `grzybobranie_ml.ship_gates` | **P8** | Automatyczna ewaluacja twardych bramek jakościowych i bezpieczeństwa (`confidentToxicAsEdible == 0`, pomyłki w parach $\le 0.05$, `tfliteAgreement >= 0.99`). |
| `grzybobranie_ml.manifest` | **P8** | Generowanie i walidacja końcowego pliku `model-manifest.json` względem schematu JSON Schema v1.0.1. |

---

## 4. Porównanie z prototypem `ml/` i zakres koniecznej przebudowy

W toku inspekcji zbadano prototyp z repozytorium aplikacji (`/home/bubolka/projekty/mushroom-recognition-app/ml/`, stan po zamrożeniu opisanym w `OUTDATED.md`). Prototyp ten jest fundamentalnie nieaktualny w stosunku do zatwierdzonej architektury mini+ i nie może być przeniesiony wprost poprzez prostą korektę stałych liczbowych.

### 4.1. Kluczowe różnice architektoniczne

| Cecha | Prototyp `ml/` (przestarzały) | Nowy potok `grzybobranie-ml` (obowiązujący) |
|---|---|---|
| **Wyjścia głowicy** | 54 wyjścia (52 gatunki + 2 klasy zbiorcze) | Do 67 wyjść (docelowo 67), wyłącznie konkretne gatunki |
| **Klasy zbiorcze** | `not_a_mushroom` oraz `unknown_mushroom` w głowicy | **Całkowity brak klas zbiorczych.** Odrzucanie przez bramkę energii OOD i `minTop1Prob` |
| **Ograniczenia danych (cap)** | Sztuczny `class_cap = 2500` dla klas zbiorczych | Brak sztucznych capów. Wykorzystanie pełnych dostępnych danych z paczek |
| **Gatunki bez danych** | Oznaczane jako `in_model: false` w klasach głowicy | Trafiają do listy `uncoveredSpecies` w manifeście; nie ma ich w wyjściach softmaxu |
| **Framework treningowy** | Keras / TensorFlow (MobileNetV3-Small) | PyTorch + `timm` (modele z licencją Apache-2.0/MIT/BSD: MobileNetV3/V4, EfficientNet-Lite) |
| **Eksport na urządzenie** | `tf.lite.TFLiteConverter` | `litert-torch` (LiteRT 2.1.3 / `react-native-fast-tflite` 3.0.1) |
| **Rozmiar wejścia** | Na sztywno 224×224 px | Parametryczny: 224×224 oraz 256×256 badane równolegle do testów na Xiaomi Mi 11 5G |
| **Groźne rodzaje** | Tylko 4 rodzaje w `DANGEROUS_GENERA` | Rozszerzone do min. 6 rodzajów (*Amanita*, *Galerina*, *Gyromitra*, *Inosperma*, *Cortinarius*, *Lepiota*) |
| **Pary niebezpieczne** | Ad-hoc w `recognition_math.py` | Oficjalnie zatwierdzone `DANGEROUS_PAIRS` z `configs/karty_sobowtory.json` |

### 4.2. Ocena przydatności modułów z prototypu `ml/`

| Moduł w `ml/` | Status ponownego użycia | Wymagane zmiany pod `mini+` 67 klas |
|---|---|---|
| `split.py` / `split_salt.py` | **Do adaptacji** | Zachować logikę spójnych składowych HMAC (autor, iNat obs/photo, GBIF, MO). Usunąć obsługę DF20/DF24 oraz logikę dedykowaną klasom zbiorczym. Dodać stratyfikację klasową w grupach. |
| `license_policy.py` / `licenses.py` | **Do adaptacji** | Wykorzystać reguły filtracji licencji (CC0, CC BY, CC BY-NC, CC BY-SA, pd_self; odrzucenie ND i DF20/DF24). |
| `preprocess.py` | **Do adaptacji** | Zachować pipeline baseline (EXIF, box do 448 px, resampling box z antialias, normalizacja $[-1, 1]$), sparametryzować pod wymiar $N \in \{224, 256\}$. |
| `recognition_math.py` | **Odrzucony (zastąpiony)** | Stary moduł bazował na klasach zbiorczych. Został całkowicie zastąpiony nowym `src/grzybobranie_ml/recognition_math.py`. |
| `export_tflite.py` | **Wymaga przepisania** | Oparty na TensorFlow Keras; należy napisać od nowa z wykorzystaniem `litert-torch` na bazie modelu PyTorch. |
| `ship_gates.py` | **Do adaptacji** | Przepisać pod kątem manifestu v1.0.1: usunąć metryki klas zbiorczych, wdrożyć bramki dla bramki energii OOD, `DANGEROUS_PAIRS` z `karty_sobowtory.json` oraz zero tolerancji dla pomyłek trujący $\to$ jadalny. |
| `score_delegate_dump.py` | **Do adaptacji** | Użyteczny w Etapie P7 do weryfikacji logitów z urządzeń mobilnych (Xiaomi Mi 11, Huawei P smart, Samsung S23). |

---

## 5. Inwentaryzacja zbiorów danych (Hugging Face `lonwi/grzybobranie-zdjecia`)

W prywatnym repozytorium Hugging Face `lonwi/grzybobranie-zdjecia` potwierdzono obecność wymaganych pakietów danych:

### 5.1. Paczki manifestów i zdjęć na Hugging Face
1. **Paczka `_v3` (`data/20261108-v3/`):**
   - 196 plików w katalogu na HF (192 pliki CSV).
   - Obejmuje 51 klas bazowych.
   - Liczba wierszy: **74 040** (w tym 28 043 oznaczone jako `ui_safe`).
2. **Paczka `new16` (`data/miniplus-new16-20261110/`):**
   - 51 plików w katalogu na HF (42 pliki CSV).
   - Obejmuje 16 klas rozszerzenia mini+ (łącznie 13 863 wiersze w paczce źródłowej).
   - 14 klas występuje wyłącznie w tej paczce (**13 601 wierszy**, w tym 1 993 `ui_safe`).
   - Pozostałe 2 klasy (*Amanita verna* i *Lepiota brunneoincarnata*) miały w tej paczce bazowej **262 wiersze** (22 `ui_safe`).
3. **Paczka `addendum` (`data/miniplus-new16-addendum-20261110/`):**
   - 12 plików w katalogu na HF.
   - Zawiera **31 wierszy** dodatkowych zdjęć (wyłącznie do treningu, `ui_safe = 0`), uzupełniających zbiór dla *Amanita verna* (24 wiersze) i *Lepiota brunneoincarnata* (7 wierszy).
   - Łączna liczba wierszy dla tych 2 klas po scaleniu paczki bazowej i addendum: $262 + 31 =$ **293 wiersze** (*Amanita verna*: 67, *Lepiota brunneoincarnata*: 226; łącznie 22 `ui_safe`).
4. **Zbiór `ood-v1` (`data/ood-v1/` na HF):**
   - Pliki w repozytorium HF: `ood_ids.txt`, `ood_fungi.csv`, `ood_nonfungi.csv`, `SHA256SUMS`, `README.md`, skrypty pomocnicze `build_ood_v1.py`, `validate_ood_v1.py`.
   - Zbiór służy wyłącznie do kontroli rozłączności (brak wycieku do treningu) oraz do kalibracji bramki energii OOD i progów fuzji (P5).

### 5.2. Zbiorcze zestawienie pokrycia dla 67 gatunków (`configs/pokrycie-67.csv`)

> **Wyjaśnienie arytmetyki źródeł vs klas:**  
> W pierwotnej notatce z decyzji (`docs/decisions/decyzje.md`) zestawienie mogło sprawiać wrażenie sprzeczności (13 863 $\to$ 13 601 przy 0 odfiltrowanych OOD, oraz 31 $\to$ 293).  
> W rzeczywistości **żadne zdjęcia nie zostały odfiltrowane ani policzone podwójnie**:  
> - Paczka bazowa `new16` zawiera 13 863 wiersze = 13 601 (dla 14 klas) + 262 (dla 2 klas z addendum).  
> - Paczka `addendum` zawiera 31 wierszy (dla tych samych 2 klas).  
> - Suma źródeł: $74\,040 + 13\,863 + 31 = \mathbf{87\,934}$.  
> - Suma klas po przypisaniu do głowicy: $74\,040 \text{ (51 klas)} + 13\,601 \text{ (14 klas)} + 293 \text{ (2 klasy)} = \mathbf{87\,934}$.  
> Liczba 262 wierszy to po prostu część paczki `new16` przypisana do 2 klas oznaczonych w `pokrycie-67.csv` jako `new16+addendum`.

| Grupa klas w głowicy | Liczba klas | Źródło na HF | Wiersze z paczki bazowej | Wiersze z addendum | Łącznie wierszy po scaleniu | Wiersze `ui_safe` |
|---|---|---|---|---|---|---|
| Klasy bazowe `v3` | 51 | `data/20261108-v3/` | 74 040 | 0 | 74 040 | 28 043 |
| Klasy `new16` (bez addendum) | 14 | `data/miniplus-new16-20261110/` | 13 601 | 0 | 13 601 | 1 993 |
| Klasy krytyczne `new16+addendum` | 2 | `new16` + `addendum` | 262 | 31 | 293 | 22 |
| **ŁĄCZNIE (cały model)** | **67** | **Wszystkie paczki HF** | **87 903** | **31** | **87 934** | **30 058** |

### 5.3. Rozkład licencji w zbiorze (łącznie 87 934 zdjęć)
* **CC0:** 6 795 (7,73%)
* **CC BY:** 23 263 (26,46%)
* **CC BY-NC:** 40 887 (46,50%) — dopuszczone wyłącznie do treningu modelu, nigdy do UI aplikacji
* **CC BY-SA:** 16 761 (19,06%)
* **pd_self:** 228 (0,26%)

### 5.4. Gatunki z najmniejszą liczbą zdjęć (klasy krytyczne)
Zgodnie z zasadą projektu, minimalny próg wejścia do głowicy wynosi **$\ge 55$ zdjęć** dla gatunków `high_stakes` / toksycznych oraz **$\ge 40$ zdjęć** dla pozostałych.

| Slug gatunku | Nazwa polska | Źródło | Liczba zdjęć (`total`) | Próg (`prog`) | `high_stakes` | Status wejścia |
|---|---|---|---|---|---|---|
| `amanita_verna` | muchomor wiosenny | new16+addendum | 67 | 55 | `true` | **Wchodzi** (margines: +12) |
| `inocybe_erubescens` | strzępiak ceglasty | v3 | 180 | 55 | `true` | **Wchodzi** |
| `lepiota_brunneoincarnata` | czubajeczka brązowoczerwonawa | new16+addendum | 226 | 55 | `true` | **Wchodzi** |
| `entoloma_sinuatum` | wieruszka zatokowata | v3 | 391 | 55 | `true` | **Wchodzi** |
| `russula_emetica` | gołąbek wymiotny | v3 | 699 | 55 | `true` | **Wchodzi** |
| `agaricus_moelleri` | pieczarka szarołuskowa | new16 | 782 | 55 | `true` | **Wchodzi** |
| `russula_virescens` | gołąbek zielonawy | v3 | 988 | 40 | `false` | **Wchodzi** |

Wszystkie 67 gatunków z konfiguracji `pokrycie-67.csv` spełniają minimalne progi wejścia do głowicy (`przechodzi == true`). Żaden gatunek nie trafia na starcie do `uncoveredSpecies`.

---

## 6. Rejestr ryzyk i środki zaradcze

| Identyfikator | Opis ryzyka | Prawdopodobieństwo | Wpływ | Środek zaradczy |
|---|---|---|---|---|
| **R-01** | **Skrajnie mała liczność próby testowej dla `amanita_verna` i ryzyko 0 w teście przy naiwnym HMAC**. Przy 67 zdjęciach, podział HMAC grupowany po obserwacji i autorze może dać 0 zdjęć w teście, jeśli kilka dużych obserwacji wpadnie w całości do train/val. Naiwne „wymuszanie” psuje determinizm i właściwości HMAC. | Wysokie | Krytyczny | **Deterministyczny przydział spójnych składowych ze stratyfikacją po gatunku** i twardym minimum dla klas `high_stakes` (aby zagwarantować m.in. $n \ge 8$ w test dla ewaluacji Wilsona w ship gates). Algorytm zostanie sformalizowany w decyzji architektonicznej przed implementacją w `prepare_data`. W raporcie P1 obligatoryjnie zliczamy **liczbę unikalnych obserwacji i autorów** na gatunek, a nie tylko surową liczbę zdjęć. |
| **R-02** | **Wyciek danych między zbiorem treningowym a OOD v1**. | Średnie | Wysoki | W Etapie P1 (`prepare_data.py`) obligatoryjne pobranie i porównanie wszystkich ID obserwacji, URL-i i hashy SHA-256 z plikiem `data/ood-v1/ood_ids.txt`. |
| **R-03** | **Złamanie licencji wag lub zdjęć** (użycie wag komercyjnie zastrzeżonych lub zdjęć DF20/DF24). | Niskie | Krytyczny | Walidacja licencji bazowych modeli w timm (wyłącznie Apache-2.0, MIT, BSD; blokada ConvNeXt V2 i FastViT). W skryptach danych twarde odrzucanie DF20/DF24 oraz licencji ND. |
| **R-04** | **Thermal throttling GPU i przerywanie sesji na laptopie**. | Średnie | Średni | Trening w WSL2 z checkpointem co epokę i automatycznym `resume`. Monitorowanie parametrów karty przez `nvidia-smi`. Zastosowanie `bf16` i ostrożnego `batch_size`. |
| **R-05** | **Rozbieżność formatu wejścia między Pythonem a aplikacją mobilną**. | Średnie | Wysoki | Utrzymanie podwójnego pipeline'u dla 224 i 256 px aż do pomiaru na Xiaomi Mi 11 5G. Generowanie fixture'ów P1 (`parity.json`) i P2 (`decision_cases.json`). |
| **R-06** | **Licencje CC BY-NC w zbiorze treningowym (46,50% danych) a ewentualna przyszła komercjalizacja**. Zdjęcia CC BY-NC są w pełni legalne do treningu darmowej aplikacji (bez IAP/reklam). Jednak status prawny wag modelu jako utworu zależnego w razie komercjalizacji produktu to szara strefa. | Średnie | Średni | **Formalna decyzja projektowa:** Akceptujemy to ryzyko dla obecnego etapu bezpłatnej aplikacji (zgodnie z `soul.md` §2.1 pkt 2). W przypadku decyzji o komercjalizacji produktu model musiałby zostać przetrenowany wyłącznie na podzbiorze `ui_safe` + `CC BY-SA` lub po pozyskaniu licencji komercyjnych. |

---

## 7. Uzgodniona kolejność prac i plan wdrożenia (Etapy P0 $\to$ P1)

W odpowiedzi na pytanie końcowe Etapu P0, Wojciech Bubółka zatwierdził precyzyjną, 2-etapową sekwencję wdrożenia:

1. **Krok 1: Opracowanie `DATA_CONTRACT.md` dla 67 wyjść:**
   - Krótki, wiążący dokument architektoniczny definiujący kontrakt danych dla całego potoku:
     - głowica do 67 gatunków (cel 67),
     - całkowity brak klas zbiorczych (`not_a_mushroom`, `unknown_mushroom`),
     - obsługa gatunków poniżej progów (55/40) wyłącznie przez `uncoveredSpecies` w `model-manifest.json` (nigdy `in_model: false`),
     - definicja pól metadanych (autor, obserwacja, źródło, licencja, split, ui_safe).
2. **Krok 2: Implementacja `grzybobranie_ml.prepare_data`:**
   - Adaptacja `license_policy.py` oraz `split.py` z prototypu `ml/`.
   - Zapewnienie ścisłej rozłączności z `data/ood-v1/ood_ids.txt`.
   - Deduplikacja dokładna (SHA-256) oraz perceptualna (pHash).
   - Deterministyczny podział grup spójnych składowych (HMAC) ze stratyfikacją po gatunku i twardym minimum dla klas `high_stakes` (zgodnie z decyzją R-01).
   - Wygenerowanie raportów `data_audit.md` (z liczbą zdjęć, unikalnych obserwacji i autorów) oraz `leak_report.json`.
3. **Status pozostałych modułów:**
   - `src/grzybobranie_ml/recognition_math.py` jest już w pełni zaimplementowane i przetestowane w Etapie P2.
   - `preprocess.py` oczekuje w pull requeście [PR #2](https://github.com/lonwi/mushroom-recognition-app-ml/pull/2).
   - **Z prototypu `ml/` nie przenosimy żadnych innych modułów.**

---

## 8. Podsumowanie i zamknięcie Etapu P0

Etap P0 został pomyślnie zrealizowany i zweryfikowany:
- Środowisko WSL2 (Ubuntu 26.04, Python 3.14.4, RTX 5070 Laptop sm_120, PyTorch 2.15+cu132) jest w pełni skonfigurowane i przetestowane.
- Suita testowa P2 (25/25 testów) przechodzi na zielono.
- Karty sobowtórów zostały zatwierdzone jako `DANGEROUS_PAIRS`.
- Powyższy raport uwzględnia wszystkie korekty i uwagi Wojciecha.
- Przechodzimy do realizacji Kroku 1: utworzenia `DATA_CONTRACT.md`.
