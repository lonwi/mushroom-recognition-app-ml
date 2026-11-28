# Kontrakt danych: agent danych ↔ pipeline ML

> **Status:** szkic 1.0.1-draft z 2026-11-28, po decyzjach Wojciecha i odpowiedziach agenta danych (wersja pakietu `grzybobranie-ml` pozostaje v0.1.0).
> **Strony:** agent danych (właściciel prywatnego datasetu HF `lonwi/grzybobranie-zdjecia`) i pipeline ML z tego repozytorium.
> **Przepis zdjęć:** `jpeg_q85_max768_v1` (`encode()` w `hf-768/scripts/common.py` po stronie danych).
> **Uwaga prawna:** sekcja 8 opisuje decyzje projektowe i nie jest poradą prawną. Przed zmianą modelu użycia (np. komercjalizacja) trzeba ją przeglądnąć z prawnikiem.

Reguły ogólne projektu (progi OOD, zakazane źródła, backbone'y) są w `README.md`, `AGENTS.md` i `docs/soul.md`. Ten plik opisuje wyłącznie **granicę danych**: co dostarcza agent danych i co z tym robi ML.

## 1. Podział odpowiedzialności

| Etap | Właściciel | Wynik |
|---|---|---|
| Pozyskanie, filtr źródeł, `jpeg_q85_max768_v1`, manifest | agent danych | prywatny dataset HF + `manifest_768.parquet` |
| `prepare_data` (normalizacja licencji, rozłączność z ood-v1, deduplikacja dokładna, podział, raporty) | ML | `data/splits/*` + raporty |
| `pack_shards` (WebDataset) | ML | `data/shards/{train,val,test}/` na ext4 |
| Trening, kalibracja OOD, eksport | ML | `artifacts/*` |

Agent danych **nie** buduje shardów. ML **nie** zmienia bajtów zdjęć i **nie** wylicza ścieżek z wzorca — zawsze korzysta z kolumny `path` manifestu.

## 2. Zdjęcia (`jpeg_q85_max768_v1`)

- Każde zdjęcie to **pojedynczy plik JPEG**, przepis `jpeg_q85_max768_v1`.
- **Bez powiększania.** `LANCZOS thumbnail` jest stosowany tylko gdy `max(w, h) > 768`, więc po przetworzeniu zawsze `max(width, height) ≤ 768`.
- Kodowanie: PIL, jakość **q85**, `optimize`, podpróbkowanie **4:2:0**, JPEG **baseline** (nie progresywny). Kanał alfa jest spłaszczany na białe tło. 16 bitów jest skalowane do 8 bitów.
- **Orientacja jest wypalona w pikselach, EXIF usunięty** (`has_exif` to błąd walidatora po stronie danych). Profil ICC jest konwertowany do sRGB i **nie jest osadzany** — ML zakłada sRGB. **Znane ryzyko:** gdy konwersja ICC się nie powiedzie, piksele zostają bez konwersji i **nie ma flagi** w manifeście.
- Skoro orientacja jest już wypalona, `exif_transpose` w preprocessingu P1 na tych plikach jest nieszkodliwym no-opem.
- **Bez ponownej kompresji w dół strumienia.** ML nigdy nie zapisuje zdjęć ponownie jako JPEG. Augmentacje i preprocessing działają na zdekodowanych pikselach. Do shardów trafiają **te same bajty** pliku z datasetu.
- **ood-v1 jest przetwarzany tym samym przepisem** i leży w tym samym datasecie; wiersze ood-v1 mają `set == "ood-v1"` (ML wylicza `is_ood`).
- `sha256` to skrót **pliku 768** (dokładnie tych bajtów), `original_sha256` to skrót pliku źródłowego sprzed resize. `original_width` / `original_height` to wymiary **po zastosowaniu orientacji EXIF**, przed zmniejszeniem.

## 3. Manifest

Kanoniczny plik to **`manifest_768.parquet`**. CSV jest kopią (te same kolumny i typy). `prepare_data` czyta w pierwszej kolejności Parquet. **Jeden manifest obejmuje wszystkie zbiory**, rozróżniane kolumną `set`.

Ścieżki w HF **nie są jednolite**. Zawsze należy używać kolumny `path` i **nigdy** nie wyliczać ścieżek z wzorca katalogów.

| Kolumna | Typ | Wymagana | Opis |
|---|---|---|---|
| `path` | string | tak | ścieżka względem korzenia datasetu HF, separator `/`; jedyne źródło lokalizacji pliku |
| `sha256` | string (64 hex, małe litery) | tak | skrót pliku 768 |
| `original_sha256` | string (64 hex, małe litery) | tak | skrót pliku źródłowego; unikalny w manifeście |
| `species` | string | tak* | slug gatunku, taki jak kolumna `class` w `configs/pokrycie-67.csv` |
| `observation_id` | string | nie | identyfikator obserwacji, z prefiksem źródła; **może być pusty** (zob. niżej) |
| `author_id` | string | tak | stabilny identyfikator autora, z prefiksem źródła |
| `license` | string | tak | **surowa** etykieta z źródła (obecnie 28 wartości, np. `CC BY-NC`, `CC BY-NC 4.0`, `CC BY-SA 3.0 DE`, `CC0 1.0`, `Public domain`, `pd-self`) |
| `license_url` | string | nie, ale wiarygodniejsza przy wersji | URL licencji; ML używa go przy normalizacji do SPDX, gdy surowy napis nie ma wersji |
| `source` | string | tak | dostawca, np. `inaturalist`, `wikimedia_commons`, `mushroom_observer`, `svampeatlas`, `artsobservasjoner`, `arter_dk`, `observation_org`, `ala` |
| `width`, `height` | int | tak | wymiary po `jpeg_q85_max768_v1`, px |
| `original_width`, `original_height` | int | tak | wymiary po orientacji EXIF, przed zmniejszeniem |
| `set` | string | tak | podzbiór; znana wartość `ood-v1`. ML wylicza `is_ood = (set == "ood-v1")` |
| `attribution`, `source_url` | string | nie | pomocnicze, do raportów i atrybucji |

Nie ma kolumny `tdm_optout` (zob. sekcja 8 i pytania otwarte).

Wymagania i wyjątki:

- \*`species` jest wymagany dla wierszy treningowych. Dla `set == "ood-v1"` może być pusty lub pomocniczy i nie wchodzi do treningu.
- `pd-self` / `pd_self` występuje **tylko jako wartość licencji**, nie jako źródło. W manifeście nie ma wierszy źródłowych `pd_self` ani `local_drop`.
- **Puste `observation_id`:** w bieżących danych 648 wierszy (624 `artsobservasjoner`, 18 `arter_dk`, 5 `observation_org`, 1 `ala`). ML **nie** wpisuje w to miejsce sztucznego identyfikatora. Do **grupowania** (tylko) wylicza zapasowy token `anon:<source>:<original_sha256[:16]>`. To **nie** jest prawdziwe `observation_id` i nie trafia do `<key>.json` w shardach jako identyfikator obserwacji.
- `max(width, height) ≤ 768` oraz cechy JPEG przepisu (q85, baseline, 4:2:0) są weryfikowane w `prepare_data` (krok 1).

## 4. Przypięta rewizja HF

Konfiguracja ML (`hf_dataset_revision` w planowanym `configs/data.json`) przechowuje **pełny 40-znakowy SHA commita**. Nazwy gałęzi i `main` są **niedozwolone**. `prepare_data`:

1. pobiera dokładnie tę rewizję na ext4 (nigdy `/mnt/c`),
2. zapisuje w raportach rewizję i SHA-256 manifestu,
3. odmawia pracy, gdy lokalna kopia nie zgadza się z przypiętą rewizją.

**Nie przypinać teraz rewizji `338e87b7…`.** Trwa wgrywanie zestawu ~106 tys. zdjęć (ok. 65 gatunków); agent danych przekaże nowy pełny SHA, dopiero wtedy wpis trafi do konfiguracji. Do tego czasu `hf_dataset_revision` pozostaje puste, a `prepare_data` nie startuje produkcyjnego przebiegu.

Nowa wersja danych = nowy commit na HF + nowy wpis w konfiguracji ML (zmiana widoczna w review).

## 5. `prepare_data` (ML)

Kolejność kroków (każdy zapisuje licznik do raportu):

1. **Walidacja manifestu:** wymagane kolumny, typy, unikalność `original_sha256` i `path`, `width`/`height` w granicach, weryfikacja `sha256` wszystkich plików (strumieniowo) oraz próbkowo cech JPEG przepisu (q ≈ 85, baseline, 4:2:0, dłuższy bok ≤ 768). Ścieżki wyłącznie z kolumny `path`.
2. **Polityka licencji:** surowe etykiety są normalizowane do SPDX (lub stałych projektowych). Przy ustalaniu **wersji** pierwszeństwo ma `license_url`. Dozwolone po normalizacji: `CC0`, `CC-BY`, `CC-BY-NC`, `CC-BY-SA` (wszystkie wersje) oraz `public-domain` i `pd-self`. Odrzucane: DF20/DF24, obrazy generowane, rysunki/ryciny, `ND`, brak licencji. Brak wersji w surowym napisie **nie** odrzuca wiersza, jeśli typ jest dozwolony — trafia do raportu jako `license_version_unknown`. Nie ma kolumny `tdm_optout`; brak zastrzeżenia traktuje się jako **nieznany** (nie jako „brak opt-out”). Odrzucone wiersze trafiają do raportu z powodem.
3. **Rozłączność z ood-v1:** wiersze treningowe, których `original_sha256` (a dodatkowo niepuste `observation_id`) występuje w `set == "ood-v1"`, są usuwane. Raport pokazuje liczbę usuniętych i **zero** wspólnych kluczy po filtrze.
4. **Deduplikacja dokładna** po `original_sha256` i po `sha256` pliku 768 (zostaje wiersz o najmniejszym `path`). Near-duplicates: **brak pHash i innych danych podobieństwa** po stronie datasetu; ML ich nie szuka i może tylko odnotować w raporcie, że kontrola near-dup jest niedostępna.
5. **Ograniczenie do klas docelowych** (kolumna `class` w `configs/pokrycie-67.csv`).
6. **Podział train/val/test** zgodnie z decyzją R-01 (`docs/decisions/2026-11-28-split-and-training-decisions.md`). Puste `observation_id` grupuje się przez token zapasowy z sekcji 3, nie przez pusty string.
7. **Raporty** w `data/splits/`: `split_report.json` i `split_report.md` (liczności, zdjęcia, unikalne obserwacje i autorzy na gatunek i split, kontrola wycieku, progi 55/40 łącznie po filtrach i osobno w `train`, lista odrzuceń, `license_version_unknown`), `splits.json` (przypisania), `labels.json` (kolejność klas). Raporty nigdy nie zawierają soli.

`prepare_data` kończy się kodem niezerowym i raportem `FAILED`, gdy którakolwiek kontrola jakości danych nie przechodzi. Nie ma cichych obejść.

## 6. `pack_shards` (ML, po `prepare_data`)

Wejście: `splits.json`, `labels.json`, lokalna kopia datasetu (pliki wskazywane kolumną `path`). Wyjście: shardy WebDataset w `data/shards/{train,val,test}/` na **ext4** (nie `/mnt/c`).

- **Bajty zdjęć bez zmian:** wpis `.jpg` to dokładnie plik z datasetu (weryfikacja `sha256` przed zapisem).
- **Klucz próbki:** `__key__ = original_sha256` (64 znaki hex, bez kropek).
- **Pliki próbki:**
  - `<key>.jpg` bajty JPEG,
  - `<key>.json` UTF-8: `species`, `observation_id` (oryginalna wartość, także pusta), `license` (surowa i znormalizowana), `sha256`, `original_sha256` (celowo **bez** `author_id` i **bez** tokenu zapasowego grupowania),
  - `<key>.cls` indeks klasy jako ASCII (np. `17`). Indeks wynika z **kolejności etykiet głowicy modelu**, czyli z `labels.json` (domyślnie sortowanie alfabetyczne po slugu spośród klas w głowicy). Ta sama lista trafia do `model-manifest.json` jako `classes`.
- **Osobne zbiory:** `train-%06d.tar`, `val-%06d.tar`, `test-%06d.tar`. Shard train ≈ **0,5 GB** (zamykany po przekroczeniu 512 MiB), shardy val i test mniejsze (cel ≈ 128 MiB).
- **Deterministyczna kolejność:** klucze sortowane po `original_sha256`, następnie tasowane przez `random.Random(seed)` z jawnym `seed` zapisanym w `shards.json`. Wpisy w tarze mają stałe metadane (`mtime=0`, `uid=gid=0`, stała kolejność `.cls`, `.jpg`, `.json`), więc **te same dane + ten sam seed = te same bajty shardów**.
- **Indeks:** `shards.json` (nazwy, liczba próbek, rozmiar, SHA256 każdego sharda, seed, rewizja HF, SHA256 manifestu) oraz `SHA256SUMS`.

## 7. Minimalna liczba zdjęć na gatunek: progi 55 / 40

Decyzja Wojciecha z 28.11.2026:

- **≥ 55** zdjęć **w całym zbiorze po filtrach** (`prepare_data`, kroki 1–5) dla gatunków `high_stakes` / toksycznych (kolumna `high_stakes` w `configs/pokrycie-67.csv`),
- **≥ 40** dla pozostałych.
- Liczba zdjęć w samym splicie `train` jest **raportowana osobno** i nie jest kryterium blokującym.

Gatunek poniżej progu (liczonego łącznie po filtrach) trafia do `uncoveredSpecies` w `model-manifest.json`, nie do głowicy. To progi liczności, nie progi bramki energii/OOD. `prepare_data` **sama nie zmienia listy klas**: gdy po filtrach któryś gatunek nie przechodzi progu, raport wskazuje go i zatrzymuje pipeline do decyzji człowieka.

## 8. Licencje i prawo

**Decyzja właściciela (Wojciech Bubółka, 28.11.2026):** aplikacja jest **niekomercyjna**.

- Zdjęcia z licencją **BY-NC** są używane do trenowania na podstawie **wyjątku o eksploracji tekstów i danych (TDM)**: art. 26³ ustawy o prawie autorskim i prawach pokrewnych (wdrożenie art. 4 dyrektywy 2019/790 o prawie autorskim na jednolitym rynku cyfrowym), **o ile uprawniony nie dokonał zastrzeżenia w formacie przeznaczonym do odczytu maszynowego** (art. 26³ ust. 2). Licencja NC jest ponadto spójna z niekomercyjnym charakterem aplikacji (do potwierdzenia w przeglądzie prawnym).
- **Kontrola opt-out TDM nie jest dziś wykonywana** i w manifeście nie ma kolumny `tdm_optout`. Brak zastrzeżenia traktuje się jako **nieznany**, nie jako potwierdzony brak opt-out. Dodanie sprawdzania i kolumny wymaga osobnej decyzji Wojciecha (pytanie otwarte).
- **Powierzchnia aplikacji:** UI pokazuje wyłącznie zdjęcia **CC0 / CC BY**. Licencje **NC, SA, public-domain i pd-self są wyłącznie treningowe** — te zdjęcia nie trafiają do Atlasu ani innych widoków użytkownika.
- **Prywatność i brak redystrybucji:** dataset HF `lonwi/grzybobranie-zdjecia` pozostaje **prywatny**, shardy i kopie robocze nie opuszczają maszyny treningowej i prywatnego HF, zdjęcia nie są publikowane ani udostępniane osobom trzecim. Wytrenowany model nie zawiera zdjęć. Atrybucja (autor, licencja, źródło) pozostaje w manifeście.
- **Przechowywanie (art. 26³ ust. 3, decyzja z 28.11.2026):** kopie robocze (HF, `data/`, shardy) są trzymane **przez czas rozwoju modelu** i **usuwane wraz z zamknięciem projektu**. Służą wyłącznie treningowi, kalibracji i odtwarzalności i nie są używane do żadnych innych celów.
- **Zmiana modelu użycia** (np. komercjalizacja aplikacji, publiczny dataset, udostępnienie modelu innym) wymaga ponownego przeglądu tej sekcji i licencji zdjęć, w tym wierszy BY-NC i BY-SA.

*Stan na 2026-11-28, na podstawie tekstu art. 26³ (Dz.U. 2024 poz. 1254; t.j. Dz.U. 2025 poz. 24). To nie jest porada prawna; interpretacja wymaga weryfikacji przez prawnika, a ostateczną wykładnię art. 4 dyrektywy 2019/790 ustala TSUE.*

## 9. Zmiany kontraktu

Zmiana kolumn, formatu, parametrów zdjęć lub zasad shardowania wymaga: aktualizacji tego pliku, nowej przypiętej rewizji HF (jeśli dotyczy danych) i wpisu w `docs/decisions/`.
