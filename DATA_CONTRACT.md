# Kontrakt danych: agent danych ↔ pipeline ML

> **Status:** szkic 1.0.0-draft z 2026-11-28 (wersja pakietu `grzybobranie-ml` pozostaje v0.1.0).
> **Strony:** agent danych (właściciel prywatnego datasetu HF `lonwi/grzybobranie-zdjecia`) i pipeline ML z tego repozytorium.
> **Uwaga prawna:** sekcja 8 opisuje decyzje projektowe i nie jest poradą prawną. Przed zmianą modelu użycia (np. komercjalizacja) trzeba ją przeglądnąć z prawnikiem.

Reguły ogólne projektu (progi OOD, zakazane źródła, backbone'y) są w `README.md`, `AGENTS.md` i `docs/soul.md`. Ten plik opisuje wyłącznie **granicę danych**: co dostarcza agent danych i co z tym robi ML.

## 1. Podział odpowiedzialności

| Etap | Właściciel | Wynik |
|---|---|---|
| Pozyskanie, filtr źródeł, resize, JPEG q85, manifest | agent danych | prywatny dataset HF + manifest |
| `prepare_data` (polityka licencji, rozłączność z ood-v1, deduplikacja, podział, raporty) | ML | `data/splits/*` + raporty |
| `pack_shards` (WebDataset) | ML | `data/shards/{train,val,test}/` na ext4 |
| Trening, kalibracja OOD, eksport | ML | `artifacts/*` |

Agent danych **nie** buduje shardów. ML **nie** zmienia bajtów zdjęć.

## 2. Zdjęcia

- Każde zdjęcie to **pojedynczy plik JPEG**: dłuższy bok = **768 px** (bez powiększania mniejszych; patrz uwaga niżej), jakość **q85**, kolorowy RGB.
- **Bez ponownej kompresji w dół strumienia.** ML nigdy nie zapisuje zdjęć ponownie jako JPEG. Augmentacje i preprocessing działają na zdekodowanych pikselach. Do shardów trafiają **te same bajty** pliku z datasetu.
- **ood-v1 jest przetwarzany tak samo** (768 px, q85) i leży w tym samym datasecie; wiersze ood-v1 są oznaczone w manifeście (`is_ood`).
- `sha256` w manifeście to skrót **pliku po resize** (dokładnie tych bajtów), `original_sha256` to skrót pliku źródłowego sprzed resize. Pierwszy służy do weryfikacji integralności, drugi do deduplikacji, rozłączności z ood-v1 i jako stały klucz próbki.

## 3. Manifest

Format: **Parquet** (preferowany) albo CSV UTF-8. Nazwy kolumn i typy są takie same w obu. Dokładnie jeden manifest obejmuje także wiersze ood-v1 (z flagą). ML czyta oba formaty.

| Kolumna | Typ | Wymagana | Opis |
|---|---|---|---|
| `file_path` | string | tak | ścieżka względem korzenia datasetu HF, separator `/` |
| `sha256` | string (64 hex, małe litery) | tak | skrót pliku po resize |
| `original_sha256` | string (64 hex, małe litery) | tak | skrót pliku źródłowego; unikalny w manifeście |
| `species` | string | tak | slug gatunku, taki jak kolumna `class` w `configs/pokrycie-67.csv` |
| `observation_id` | string | tak | identyfikator obserwacji, z prefiksem źródła (np. `inat:12345`) |
| `author_id` | string | tak | stabilny identyfikator autora, z prefiksem źródła |
| `license` | string | tak | licencja **z wersją**, ustalone wartości: `CC0 1.0`, `CC BY 4.0`, `CC BY-NC 4.0`, `CC BY-SA 4.0`, `pd_self` (dla starszych wersji CC odpowiednio np. `CC BY 3.0`) |
| `source` | string | tak | kod źródła (np. `inat`, `mo`, `gbif`, `local_drop`) |
| `width` | int | tak | szerokość po resize, px |
| `height` | int | tak | wysokość po resize, px |
| `is_ood` | bool | tak | `true` dla wierszy ood-v1 |
| `tdm_optout` | bool lub pusta | nie (do potwierdzenia) | `true` jeśli znaleziono maszynowo odczytywalne zastrzeżenie (sekcja 8); pusta = nie sprawdzano |
| `license_url`, `attribution`, `source_url` | string | nie | pomocnicze, do raportów i atrybucji |

Wymagania wartości:
- `author_id` i `observation_id` nie mogą być puste. Dla źródeł bez autora lub obserwacji agent danych nadaje **syntetyczny, unikalny** identyfikator (np. `anon:<source>:<original_sha256[:16]>`), tak aby brak danych **nie sklejał** niepowiązanych zdjęć w jedną grupę podziału.
- `max(width, height) <= 768` i q85 są weryfikowane w `prepare_data` (krok 1).
- Wiersze ood-v1 nie mają wymaganego `species` z listy 67 klas; dla nich `species` może być pustym stringiem lub etykietą pomocniczą i nie wchodzi do treningu.

## 4. Przypięta rewizja HF

Konfiguracja ML przechowuje **pełny 40-znakowy SHA commita** datasetu (`hf_dataset_revision`, planowany plik `configs/data.json`). Nazwy gałęzi i `main` są **niedozwolone**. `prepare_data`:

1. pobiera dokładnie tę rewizję na ext4 (nigdy `/mnt/c`),
2. zapisuje w raportach rewizję i SHA256 manifestu,
3. odmawia pracy, gdy lokalna kopia nie zgadza się z przypiętą rewizją.

Nowa wersja danych = nowy commit na HF + nowy wpis w konfiguracji ML (zmiana widoczna w review).

## 5. `prepare_data` (ML)

Kolejność kroków (każdy zapisuje licznik do raportu):

1. **Walidacja manifestu:** wymagane kolumny, typy, unikalność `original_sha256` i `file_path`, wartości `license` z listy dozwolonych, `width`/`height` w granicach, weryfikacja `sha256` wszystkich plików (strumieniowo) oraz próbkowo rozmiaru i tablic kwantyzacji JPEG (szacunek q ≈ 85, dłuższy bok ≤ 768).
2. **Polityka licencji:** dozwolone tylko `CC0`, `CC BY`, `CC BY-NC`, `CC BY-SA` (z wersją) i `pd_self`. Odrzucane: DF20/DF24, obrazy generowane, rysunki/ryciny, `ND`, brak licencji, `tdm_optout = true`. Odrzucone wiersze trafiają do raportu z powodem.
3. **Rozłączność z ood-v1:** wiersze treningowe, których `original_sha256` (a dodatkowo `observation_id`) występuje w ood-v1, są usuwane. Raport pokazuje liczbę usuniętych i **zero** wspólnych kluczy po filtrze.
4. **Deduplikacja:** dokładne duplikaty po `original_sha256` i po `sha256` (zostaje wiersz o najmniejszym `file_path`). Podobne obrazy (near-duplicates) są tylko **raportowane**, o ile manifest dostarcza taką informację.
5. **Ograniczenie do klas docelowych** (kolumna `class` w `configs/pokrycie-67.csv`).
6. **Podział train/val/test** zgodnie z decyzją R-01 (`docs/decisions/2026-11-28-split-and-training-decisions.md`).
7. **Raporty** w `data/splits/`: `split_report.json` i `split_report.md` (liczności, zdjęcia, unikalne obserwacje i autorzy na gatunek i split, kontrola wycieku, progi 55/40, lista odrzuceń), `splits.json` (przypisania), `labels.json` (kolejność klas). Raporty nigdy nie zawierają soli.

`prepare_data` kończy się kodem niezerowym i raportem `FAILED`, gdy którakolwiek kontrola jakości danych nie przechodzi. Nie ma cichych obejść.

## 6. `pack_shards` (ML, po `prepare_data`)

Wejście: `splits.json`, `labels.json`, lokalna kopia datasetu. Wyjście: shardy WebDataset w `data/shards/{train,val,test}/` na **ext4** (nie `/mnt/c`).

- **Bajty zdjęć bez zmian:** wpis `.jpg` to dokładnie plik z datasetu (weryfikacja `sha256` przed zapisem).
- **Klucz próbki:** `__key__ = original_sha256` (64 znaki hex, bez kropek).
- **Pliki próbki:**
  - `<key>.jpg` bajty JPEG,
  - `<key>.json` UTF-8: `species`, `observation_id`, `license`, `sha256`, `original_sha256` (celowo **bez** `author_id`: nie jest potrzebny w treningu),
  - `<key>.cls` indeks klasy jako ASCII (np. `17`). Indeks wynika z **kolejności etykiet głowicy modelu**, czyli z `labels.json` (domyślnie sortowanie alfabetyczne po slugu spośród klas w głowicy). Ta sama lista trafia do `model-manifest.json` jako `classes`.
- **Osobne zbiory:** `train-%06d.tar`, `val-%06d.tar`, `test-%06d.tar`. Shard train ≈ **0,5 GB** (zamykany po przekroczeniu 512 MiB), shardy val i test mniejsze (cel ≈ 128 MiB).
- **Deterministyczna kolejność:** klucze sortowane po `original_sha256`, następnie tasowane przez `random.Random(seed)` z jawnym `seed` zapisanym w `shards.json`. Wpisy w tarze mają stałe metadane (`mtime=0`, `uid=gid=0`, stała kolejność `.cls`, `.jpg`, `.json`), więc **te same dane + ten sam seed = te same bajty shardów**.
- **Indeks:** `shards.json` (nazwy, liczba próbek, rozmiar, SHA256 każdego sharda, seed, rewizja HF, SHA256 manifestu) oraz `SHA256SUMS`.

## 7. Minimalna liczba zdjęć na gatunek: progi 55 / 40

- **≥ 55** zdjęć treningowych dla gatunków `high_stakes` / toksycznych (kolumna `high_stakes` w `configs/pokrycie-67.csv`),
- **≥ 40** dla pozostałych.

Gatunek poniżej progu trafia do `uncoveredSpecies` w `model-manifest.json`, nie do głowicy. To progi liczności, nie progi bramki energii/OOD. `prepare_data` **sama nie zmienia listy klas**: gdy po filtrach któryś gatunek nie przechodzi progu, raport wskazuje go i zatrzymuje pipeline do decyzji człowieka. Zob. otwarte pytanie o definicję (łącznie po filtrach czy tylko split train) w dokumencie decyzji.

## 8. Licencje i prawo

**Decyzja właściciela (Wojciech Bubółka, 28.11.2026):** aplikacja jest **niekomercyjna**.

- Zdjęcia z licencją **BY-NC** są używane do trenowania na podstawie **wyjątku o eksploracji tekstów i danych (TDM)**: art. 26³ ustawy o prawie autorskim i prawach pokrewnych (wdrożenie art. 4 dyrektywy 2019/790 o prawie autorskim na jednolitym rynku cyfrowym), **o ile uprawniony nie dokonał zastrzeżenia w formacie przeznaczonym do odczytu maszynowego** (art. 26³ ust. 2). Licencja NC jest ponadto spójna z niekomercyjnym charakterem aplikacji (do potwierdzenia w przeglądzie prawnym).
- Jeśli agent danych wykryje maszynowo odczytywalny opt-out dla zdjęcia, ustawia `tdm_optout = true`, a `prepare_data` takie wiersze wyklucza.
- **Prywatność i brak redystrybucji:** dataset HF `lonwi/grzybobranie-zdjecia` pozostaje **prywatny**, shardy i kopie robocze nie opuszczają maszyny treningowej i prywatnego HF, zdjęcia nie są publikowane ani udostępniane osobom trzecim. Wytrenowany model nie zawiera zdjęć. Atrybucja (autor, licencja, źródło) pozostaje w manifeście.
- **Przechowywanie:** art. 26³ ust. 3 pozwala przechowywać zwielokrotnione utwory „wyłącznie w celu eksploracji tekstów i danych, tak długo, jak jest to konieczne”. Kopie robocze (HF, `data/`, shardy) służą wyłącznie treningowi, kalibracji i odtwarzalności modelu i nie są używane do żadnych innych celów.
- **Zmiana modelu użycia** (np. komercjalizacja aplikacji, publiczny dataset, udostępnienie modelu innym) wymaga ponownego przeglądu tej sekcji i licencji zdjęć, w tym wierszy BY-NC i BY-SA.

*Stan na 2026-11-28, na podstawie tekstu art. 26³ (Dz.U. 2024 poz. 1254; t.j. Dz.U. 2025 poz. 24). To nie jest porada prawna; interpretacja wymaga weryfikacji przez prawnika, a ostateczną wykładnię art. 4 dyrektywy 2019/790 ustala TSUE.*

## 9. Zmiany kontraktu

Zmiana kolumn, formatu, parametrów zdjęć lub zasad shardowania wymaga: aktualizacji tego pliku, nowej przypiętej rewizji HF (jeśli dotyczy danych) i wpisu w `docs/decisions/`.
