# Decyzje: podział danych (R-01), raport high_stakes, baseline treningowy

> Data: 2026-11-28. Status: **do potwierdzenia przez Wojciecha** w PR (punkty oznaczone jako założenia i pytania na końcu).
> Powiązane: `DATA_CONTRACT.md`, `README.md` (reguły projektu), `docs/decisions/krok3-decyzje.md`.

## 1. Decyzje o danych z 22.11.2026

1. **Zdjęcia 768 px (dłuższy bok), JPEG q85**, jeden plik na zdjęcie. ML nie rekompresuje. ood-v1 jest przetwarzany tak samo.
2. **`original_sha256` zostaje w manifeście** jako klucz deduplikacji, rozłączności z ood-v1 i stały `__key__` próbki.
3. **WebDataset budujemy po stronie ML** (`pack_shards`, po `prepare_data`), nie po stronie agenta danych.

Skutek uboczny: planowany w README `data/png_cache/{224,256}` nie jest potrzebny do treningu (preprocessing z P1 działa na zdekodowanym JPEG-u). Pozostaje jako opcjonalny cache do testów parytetu; patrz pytania.

## 2. R-01: algorytm podziału train/val/test

### 2.1 Założenia

- Proporcje docelowe **70/15/15**, liczone w **unikalnych obserwacjach** w obrębie gatunku (nie w zdjęciach).
- **Stratyfikacja po gatunku**, a w każdym z `val` i `test` co najmniej **5 unikalnych obserwacji na gatunek**. Skoro `train` też ma mieć co najmniej 5, gatunek wymaga **≥ 15 obserwacji**.
- Zero wycieku: żadna **grupa** nie trafia do więcej niż jednego splitu. Grupa to **spójna składowa** zdjęć powiązanych wspólnym `author_id` **lub** wspólnym `observation_id` (union-find). Ten sam autor albo ta sama obserwacja nigdy nie rozdzielają się między splity.
- Sól: bajty pliku z `GRZYBOBRANIE_SPLIT_SALT_PATH`, a gdy zmienna nie jest ustawiona, z `~/.config/grzybobranie/split_salt` (po `expanduser`; końcowe białe znaki obcinane). Brak pliku = błąd (sól generuje `scripts/init_env.py`, nie `prepare_data`). Do raportów trafia tylko odcisk (pierwsze 12 znaków hex SHA-256 soli). Sól nigdy nie trafia do gita ani do raportów.

### 2.2 Wejście algorytmu

Po filtrach licencji, rozłączności z ood-v1, deduplikacji i ograniczeniu do klas docelowych (zob. `DATA_CONTRACT.md` §5). Dla każdego zdjęcia: `species`, `observation_id`, `author_id`.

### 2.3 Algorytm (deterministyczny)

1. **Grupy.** Union-find po tokenach `author:<author_id>` i `obs:<observation_id>`. `group_key` = najmniejszy (leksykograficznie, w bajtach UTF-8) token w składowej.
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
8. **Asercje końcowe:** każda grupa w dokładnie jednym splicie, brak wspólnych `author_id`, `observation_id`, `sha256` i `original_sha256` między splitami, minima ≥ 5 w `val` i `test` (oraz w `train`).

Ten sam manifest + ta sama sól = ta sama zawartość `splits.json`. Zmiana soli zmienia podział, a `splits.json` zapisuje odcisk soli, wersję algorytmu (`R-01.v1`), SHA-256 manifestu i rewizję HF.

### 2.4 Grupa z mieszanymi gatunkami

Grupa może zawierać kilka gatunków (autor fotografuje różne gatunki, rzadziej jedna obserwacja ma zdjęcia z różnymi etykietami). Zasady:
- Grupa jest **niepodzielna**; trafia do jednego splitu razem ze wszystkimi swoimi zdjęciami.
- Przypisanie zalicza obserwacje każdemu gatunkowi w grupie (`n(g, s)` osobno dla każdego `s`).
- Rzadsze gatunki są obsługiwane pierwsze, więc grupa „wspólna” dla rzadkiego i częstego gatunku najpierw służy minimom rzadkiego.
- **Obserwacja z różnymi gatunkami** jest zgłaszana w raporcie jako anomalia danych (do agenta danych) i liczona dla każdego gatunku, którego dotyczy.
- Raport pokazuje największe składowe i ich udział (ostrzeżenie, gdy jedna grupa ma > 5% zdjęć zbioru albo > 25% obserwacji któregoś gatunku). Same ostrzeżenia nie przerywają pracy.

### 2.5 Gatunek z mniej niż 15 obserwacjami

**Głośna porażka, bez cichych obejść:** `prepare_data` kończy się kodem 2, zapisuje `split_report.json/.md` ze `status: FAILED` i **nie zapisuje** `splits.json`. Raport podaje dla każdego problematycznego gatunku: `O_s`, liczbę grup, liczbę zdjęć, `high_stakes`, próg z `pokrycie-67.csv` oraz powód (`obs<15`, `groups<3`, `no_candidate`). Decyzja o przeniesieniu gatunku do `uncoveredSpecies` (lub o dodaniu danych) należy do człowieka i jest zapisywana w konfiguracji klas, a nie w kodzie.

### 2.6 Raporty

Dla każdego gatunku i splitu: liczba **unikalnych obserwacji**, **unikalnych autorów** i zdjęć; sumy globalne i odchylenie od 70/15/15; lista gatunków pod progami 55/40 (łącznie po filtrach i w samym `train`); kontrola wycieku; kontrola rozłączności z ood-v1; anomalie (obserwacje z mieszanymi gatunkami, duże składowe).

## 3. high_stakes: raport i bramka wydania

- **Osobny raport** (`artifacts/logs/high_stakes_report.{json,md}`) na zbiorach `val` i `test` (decyzje o hiperparametrach tylko na `val`):
  - **recall** każdego gatunku `high_stakes` i średni (macro) recall tej grupy, z liczbą obserwacji w mianowniku,
  - **macierz pomyłek look-alike** dla par z `configs/karty_sobowtory.json`: odsetek `A → B` i `B → A`, ze szczególnym wskazaniem pomyłek **trujący → jadalny**,
  - dwa warianty: top-1 z surowych logitów oraz tylko próbki przyjęte przez bramkę OOD (po kalibracji w P5).
- **Bramka wydania** w `ship_gates`: wydanie jest **blokowane**, gdy macro recall `high_stakes` albo odsetek pomyłek trujący → jadalny przekracza próg. **Wartość progu: TBD** po pierwszym przebiegu baseline'u (ustala ją Wojciech na podstawie raportu). Do tego czasu bramka jest **skonfigurowana jako zamknięta** (brak wartości = blokada wydania), nie jako przepuszczająca.

## 4. Baseline treningowy

- Kandydaci (`timm`): **EfficientNetV2-S** lub **ConvNeXt-Tiny** (nie ConvNeXt V2, ma licencję CC BY-NC). Wagi wstępnie wytrenowane muszą mieć licencję Apache-2.0, MIT lub BSD; konkretny tag wag jest sprawdzany przed użyciem.
- **Wejście 224 px**, preprocessing z `grzybobranie_ml.preprocess` (P1) według wpisu manifestu; dla wag `timm` normalizacja `mean_std` zgodna z konfiguracją wag.
- **AMP** (preferowane bf16 na RTX 5070; fp16 z `GradScaler` jako zapas), `channels_last`.
- **RTX 5070 Laptop, 8 GB VRAM:** start od batcha 32, zwiększanie tylko jeśli się mieści; w razie potrzeby gradient checkpointing lub akumulacja gradientu.
- Dane z shardów WebDataset (`pack_shards`), kolejność etykiet z `labels.json`.
- Wybór między kandydatami: ten sam budżet epok, jedno ziarno, ocena na `val` (macro F1, tie-break: recall `high_stakes`). `test` tylko do końcowego raportu.
- Baseline służy jako punkt odniesienia jakości; model do aplikacji i jego eksport LiteRT są przedmiotem osobnych etapów (P2/P3).

## 5. Założenia do potwierdzenia

1. Grupa to spójna składowa po `author_id` **lub** `observation_id` (anty-wyciek, jak prototyp `split.py`). Alternatywa: para `(author_id, observation_id)`, mniej restrykcyjna i daje równiejsze liczności, ale ten sam autor mógłby być w wielu splitach.
2. Progi 55/40 liczone są w tej wersji **w obu wariantach** (łącznie po filtrach oraz w samym `train`), raport pokazuje oba; **blokujący** jest wariant `train` (konserwatywny).
3. Cele 15% liczone w obserwacjach, nie w zdjęciach.
4. Minimum 5 obserwacji także w `train`.

## 6. Pytania otwarte

**Dla Wojciecha / Architekta:**
- Progi 55/40: czy dotyczą liczby zdjęć po filtrach (jak `pokrycie-67.csv`), czy tylko zbioru train? Przy wariancie `train` część rzadkich gatunków (np. 55 zdjęć łącznie) może wypaść z głowicy.
- Definicja grupy (założenie 1) i maksymalny dopuszczalny udział jednej składowej.
- Baseline EfficientNetV2-S/ConvNeXt-Tiny to model referencyjny; jak ma się do kandydata mobilnego (np. `mobilenetv4_conv_small` w README) i czy chcemy destylację?
- Co z planowanym `data/png_cache/{224,256}`: usunąć z README czy zostawić do parytetu?
- Przechowywanie kopii roboczych a art. 26³ ust. 3: jak długo trzymamy dataset i shardy po zakończeniu treningu?

**Dla agenta danych:** lista w opisie PR.
