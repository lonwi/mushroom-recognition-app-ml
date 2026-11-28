# Decyzje: podział danych (R-01), raport high_stakes, baseline treningowy

> Data: 2026-11-28. Status: **potwierdzone decyzje Wojciecha z 28.11.2026**, w tym przypięcie rewizji HF i K=67 (14:15 Europe/Warsaw). Otwarte punkty są w sekcji 6.
> Powiązane: `DATA_CONTRACT.md`, `README.md` (reguły projektu), `docs/decisions/krok3-decyzje.md`.

## 1. Decyzje o danych z 22.11 i 28.11.2026

Z 22.11.2026:

1. **Zdjęcia 768 px (dłuższy bok), JPEG q85**, jeden plik na zdjęcie, przepis `jpeg_q85_max768_v1`. ML nie rekompresuje. ood-v1 jest przetwarzany tak samo. Bez powiększania (thumbnail LANCZOS tylko gdy `max(w, h) > 768`).
2. **`original_sha256` zostaje w manifeście** jako klucz deduplikacji, rozłączności z ood-v1 i stały `__key__` próbki.
3. **WebDataset budujemy po stronie ML** (`pack_shards`, po `prepare_data`), nie po stronie agenta danych.

Z 28.11.2026 (Wojciech):

4. **Grupa podziału** to spójna składowa po `author_id` **lub** `observation_id` (union-find).
5. **Progi 55/40** liczy się na **całym zbiorze po filtrach**; liczność w `train` jest raportowana osobno i nie blokuje.
6. **Destylacja do modelu mobilnego** zostanie rozstrzygnięta **po baseline**.
7. **`data/png_cache/{224,256}` usunięte** z README i z tej dokumentacji. Preprocessing P1 działa na zdekodowanym JPEG-u; cache PNG nie jest potrzebny. (Katalogi `.gitkeep` i wpis w `scripts/init_env.py` mogą jeszcze istnieć w drzewie roboczym — do posprzątania przy implementacji `prepare_data`, nie w tym PR dokumentacyjnym.)
8. **Kopie robocze danych** (HF, `data/`, shardy) są trzymane przez czas rozwoju modelu i **usuwane wraz z zamknięciem projektu** (art. 26³ ust. 3).
9. **Przypięta rewizja HF:** `lonwi/grzybobranie-zdjecia` @ `faf75dca9cb5f012023003159f9351da9258f1ec` (290 789 zdjęć, zgodność z `SHA256SUMS`; 180 slugów treningowych + 91 tylko w ood-v1 = 271 unikalnych `species_slug`; `manifest_768.parquet`, 25 kolumn, bez `usage`).
10. **Głowica K=67 zostaje** (28.11, 14:15 Europe/Warsaw). Lista klas: kolumna `class` w `configs/pokrycie-67.csv` (67 unikalnych slugów; plik używany w raporcie P0). Gatunki spoza listy to rezerwa: nie są klasami i na razie nie są materiałem OOD. `prepare_data` filtruje wyłącznie tą listą, **nigdy kolumną `set`**. Wykluczona rezerwa idzie do raportu.
11. **Podzbiór UI:** oprócz ograniczenia licencyjnego (tylko CC0 / CC BY; NC, SA i pd-self wyłącznie treningowe) UI pomija 95 wierszy z `data/species-65-20261117/train_only_over_3.csv` (złączenie po `path`; `usage=train_only`, powód `over_3_per_observation`). To nie jest filtr treningowy. Plik nie jest jeszcze w repozytorium (gitignore `*.csv`) — trzeba go dodać z wyjątkiem, bez wymyślania wierszy.

## 2. R-01: algorytm podziału train/val/test

### 2.1 Zasady (potwierdzone i obowiązujące)

- **Grupa** (potwierdzone 28.11): spójna składowa zdjęć powiązanych wspólnym `author_id` **lub** wspólnym `observation_id` (union-find). Ten sam autor albo ta sama obserwacja nigdy nie rozdzielają się między splity. Żadna grupa nie trafia do więcej niż jednego splitu.
- Puste `observation_id` (648 wierszy w bieżących danych) **nie** skleja niepowiązanych zdjęć: do tokenu union-find ML używa wyłącznie zapasowego `anon:<source>:<original_sha256[:16]>`. To nie jest prawdziwe `observation_id` (zob. `DATA_CONTRACT.md` §3).
- Proporcje docelowe **70/15/15**, liczone w **unikalnych obserwacjach** w obrębie gatunku (nie w zdjęciach). Token zapasowy liczy się jako osobna obserwacja wyłącznie na potrzeby minimów i celów.
- **Stratyfikacja po gatunku**, a w każdym z `val` i `test` co najmniej **5 unikalnych obserwacji na gatunek**. Skoro `train` też ma mieć co najmniej 5, gatunek wymaga **≥ 15 obserwacji**.
- Sól: bajty pliku z `GRZYBOBRANIE_SPLIT_SALT_PATH`, a gdy zmienna nie jest ustawiona, z `~/.config/grzybobranie/split_salt` (po `expanduser`; końcowe białe znaki obcinane). Brak pliku = błąd (sól generuje `scripts/init_env.py`, nie `prepare_data`). Do raportów trafia tylko odcisk (pierwsze 12 znaków hex SHA-256 soli). Sól nigdy nie trafia do gita ani do raportów.

### 2.2 Wejście algorytmu

Po filtrach licencji, rozłączności z ood-v1, deduplikacji i ograniczeniu do klas docelowych (zob. `DATA_CONTRACT.md` §5). Dla każdego zdjęcia: `species`, `observation_id` (lub token zapasowy do grupowania), `author_id`.

### 2.3 Algorytm (deterministyczny)

1. **Grupy.** Union-find po tokenach `author:<author_id>` i `obs:<observation_id_lub_zapasowy>`. `group_key` = najmniejszy (leksykograficznie, w bajtach UTF-8) token w składowej.
2. **Kolejność grup.** `h(g) = HMAC-SHA256(salt, group_key)`. Grupy sortowane rosnąco po `(h, group_key)`. To jedyne źródło „losowości”.
3. **Profil grupy.** `n(g, s)` = liczba unikalnych obserwacji w grupie z co najmniej jednym zdjęciem gatunku `s`. `O_s` = liczba unikalnych obserwacji gatunku `s` w całym zbiorze.
4. **Kontrola wykonalności.** Jeśli dla któregokolwiek gatunku `O_s < 15` albo gatunek występuje w mniej niż 3 grupach, wynik jest **FAILED** (zob. 2.5).
5. **Cele.** Dla gatunku `s`: `T_val(s) = T_test(s) = max(5, round_half_up(0.15 · O_s))`, `T_train(s) = O_s − 2 · T_val(s)`.
6. **Faza A: gwarancja minimów.** Gatunki w kolejności `(O_s rosnąco, slug)`, czyli **najrzadszy pierwszy**. Dla każdego gatunku i każdego splitu w kolejności `val, test, train`, dopóki `c_split(s) < 5`:
   - kandydaci: nieprzypisane grupy zawierające `s`, w kolejności HMAC,
   - **warunek wykonalności:** po przypisaniu grupy `g` do splitu każdy gatunek `s'` w `g` musi nadal mieć `pozostało(s') ≥ brakujące_minima(s')`, gdzie `brakujące_minima(s') = Σ_split max(0, 5 − c_split(s'))`,
   - najpierw wybieramy pierwszego kandydata, który **nie przekracza celu** (`n(g, s) ≤ T_split(s) − c_split(s)`), a gdy brak, pierwszego spełniającego warunek wykonalności,
   - brak kandydata = **FAILED**.
   Każde przypisanie grupy zalicza obserwacje **wszystkim** gatunkom w grupie.
7. **Faza B: wypełnienie celów.** Pozostałe grupy w kolejności HMAC. Grupa trafia do splitu o największym **względnym niedoborze** `Σ_{s ∈ g} n(g, s) · (T_split(s) − c_split(s)) / O_s` (wartości ujemne dozwolone). Remisy: `train`, potem `val`, potem `test`.
8. **Asercje końcowe:** każda grupa w dokładnie jednym splicie, brak wspólnych `author_id`, `observation_id` (niepuste), `sha256` i `original_sha256` między splitami, minima ≥ 5 w `val` i `test` (oraz w `train`).

Ten sam manifest + ta sama sól = ta sama zawartość `splits.json`. Zmiana soli zmienia podział, a `splits.json` zapisuje odcisk soli, wersję algorytmu (`R-01.v1`), SHA-256 manifestu i rewizję HF.

### 2.4 Grupa z mieszanymi gatunkami

Grupa może zawierać kilka gatunków (autor fotografuje różne gatunki, rzadziej jedna obserwacja ma zdjęcia z różnymi etykietami). Zasady:
- Grupa jest **niepodzielna**; trafia do jednego splitu razem ze wszystkimi swoimi zdjęciami.
- Przypisanie zalicza obserwacje każdemu gatunkowi w grupie (`n(g, s)` osobno dla każdego `s`).
- Rzadsze gatunki są obsługiwane pierwsze, więc grupa „wspólna” dla rzadkiego i częstego gatunku najpierw służy minimom rzadkiego.
- **Obserwacja z różnymi gatunkami** jest zgłaszana w raporcie jako anomalia danych (do agenta danych) i liczona dla każdego gatunku, którego dotyczy.
- Raport pokazuje największe składowe i ich udział (ostrzeżenie, gdy jedna grupa ma > 5% zdjęć zbioru albo > 25% obserwacji któregoś gatunku). Same ostrzeżenia nie przerywają pracy. Twardy limit udziału nie został ustalony (pytanie otwarte).

### 2.5 Gatunek z mniej niż 15 obserwacjami

**Głośna porażka, bez cichych obejść:** `prepare_data` kończy się kodem 2, zapisuje `split_report.json/.md` ze `status: FAILED` i **nie zapisuje** `splits.json`. Raport podaje dla każdego problematycznego gatunku: `O_s`, liczbę grup, liczbę zdjęć, `high_stakes`, próg z `pokrycie-67.csv` oraz powód (`obs<15`, `groups<3`, `no_candidate`). Decyzja o przeniesieniu gatunku do `uncoveredSpecies` (lub o dodaniu danych) należy do człowieka i jest zapisywana w konfiguracji klas, a nie w kodzie.

### 2.6 Raporty i progi 55/40

Dla każdego gatunku i splitu: liczba **unikalnych obserwacji**, **unikalnych autorów** i zdjęć; sumy globalne i odchylenie od 70/15/15; kontrola wycieku; kontrola rozłączności z ood-v1; anomalie (obserwacje z mieszanymi gatunkami, duże składowe).

**Progi 55/40 (potwierdzone 28.11):** kryterium blokujące to liczba zdjęć **w całym zbiorze po filtrach** (`≥ 55` dla `high_stakes`, `≥ 40` dla pozostałych). Liczność w samym `train` jest w raporcie **osobno** i nie blokuje wejścia do głowicy.

## 3. high_stakes: raport i bramka wydania

- **Osobny raport** (`artifacts/logs/high_stakes_report.{json,md}`) na zbiorach `val` i `test` (decyzje o hiperparametrach tylko na `val`):
  - **recall** każdego gatunku `high_stakes` i średni (macro) recall tej grupy, z liczbą obserwacji w mianowniku,
  - **macierz pomyłek look-alike** dla par z `configs/karty_sobowtory.json`: odsetek `A → B` i `B → A`, ze szczególnym wskazaniem pomyłek **trujący → jadalny**,
  - dwa warianty: top-1 z surowych logitów oraz tylko próbki przyjęte przez bramkę OOD (po kalibracji w P5).
- **Bramka wydania** w `ship_gates`: wydanie jest **blokowane**, gdy macro recall `high_stakes` albo odsetek pomyłek trujący → jadalny przekracza próg. **Wartość progu: TBD** po pierwszym przebiegu baseline'u (ustala ją Wojciech na podstawie raportu). Do tego czasu bramka jest **skonfigurowana jako zamknięta** (brak wartości = blokada wydania), nie jako przepuszczająca.

## 4. Baseline treningowy

- Kandydaci (`timm`): **EfficientNetV2-S** lub **ConvNeXt-Tiny** (nie ConvNeXt V2, ma licencję CC BY-NC). Wagi wstępnie wytrenowane muszą mieć licencję Apache-2.0, MIT lub BSD; konkretny tag wag jest sprawdzany przed użyciem.
- **Wejście 224 px**, preprocessing z `grzybobranie_ml.preprocess` (P1) według wpisu manifestu; dla wag `timm` normalizacja `mean_std` zgodna z konfiguracją wag. Orientacja EXIF jest już wypalona w JPEG-ach datasetu, więc `exif_transpose` jest no-opem.
- **AMP** (preferowane bf16 na RTX 5070; fp16 z `GradScaler` jako zapas), `channels_last`.
- **RTX 5070 Laptop, 8 GB VRAM:** start od batcha 32, zwiększanie tylko jeśli się mieści; w razie potrzeby gradient checkpointing lub akumulacja gradientu.
- Dane z shardów WebDataset (`pack_shards`), kolejność etykiet z `labels.json`.
- Wybór między kandydatami: ten sam budżet epok, jedno ziarno, ocena na `val` (macro F1, tie-break: recall `high_stakes`). `test` tylko do końcowego raportu.
- Baseline służy jako punkt odniesienia jakości. **Destylacja do kandydata mobilnego** (np. `mobilenetv4_conv_small` z README) zostanie **rozstrzygnięta po baseline** (decyzja z 28.11). Eksport LiteRT jest przedmiotem osobnych etapów (P2/P3).

## 5. Potwierdzone decyzje (28.11.2026)

1. Grupa = spójna składowa po `author_id` **lub** `observation_id`.
2. Progi 55/40 liczone na całym zbiorze po filtrach; `train` tylko w raporcie.
3. Destylacja do modelu mobilnego — po baseline.
4. `data/png_cache/{224,256}` usunięte z dokumentacji; trening idzie z JPEG + P1.
5. Kopie robocze trzymane przez rozwój modelu, usuwane przy zamknięciu projektu (art. 26³ ust. 3).
6. Rewizja HF przypięta: `faf75dca9cb5f012023003159f9351da9258f1ec`.
7. Głowica K=67; lista w `configs/pokrycie-67.csv`; rezerwa poza listą; bez filtrowania po `set`.
8. Podzbiór UI: CC0/CC BY oraz wykluczenie 95 wierszy `train_only_over_3.csv` (po `path`).

Nadal obowiązujące (nie były przedmiotem tej rundy, bez zmiany): cele 70/15/15 w unikalnych obserwacjach; minimum 5 obserwacji także w `train` (gatunek wymaga ≥ 15).

## 6. Pytania otwarte

**Dla Wojciecha / Architekta:**

- **Kontrola opt-out TDM:** agent danych jej nie wykonuje i nie ma kolumny `tdm_optout`. Kontrakt traktuje ten stan jako nieznany. Czy dodać sprawdzanie zastrzeżeń maszynowo odczytywalnych i kolumnę w manifeście?
- **Maksymalny udział jednej składowej:** raport ostrzega przy > 5% zdjęć zbioru lub > 25% obserwacji gatunku, ale twardego limitu nie ustalono.
- **Próg bramki wydania high_stakes:** nadal TBD po pierwszym baseline (bramka do tego czasu zamknięta).
- **Pełna tablica normalizacji 28 surowych etykiet licencji** do SPDX: czy `license_url` jest zawsze obecny, gdy w `license` brakuje wersji? ML może w razie potrzeby zaproponować tabelę przy implementacji `prepare_data`.

**Dla agenta danych:**

- Rewizja `faf75dca9cb5f012023003159f9351da9258f1ec` jest przypięta w kontrakcie.
- **Znane ryzyko ICC:** gdy konwersja do sRGB się nie powiedzie, piksele zostają bez konwersji i nie ma flagi. Czy warto dodać flagę w późniejszej rewizji manifestu?
- Plik `data/species-65-20261117/train_only_over_3.csv` (95 wierszy) nie leży jeszcze w repozytorium ML — prośba o dostarczenie kopii do śledzenia w gicie.
