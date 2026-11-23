# Decyzja architektoniczna: Normalizacja zbioru do 768 px i integracja klucza `original_sha256`

**Status:** Zaakceptowana / W toku realizacji  
**Data:** 2026-11-23  
**Właściciel:** Trenowanie modelu Grzybobranie AI / Zbieranie informacji o grzybach  
**Powiązane dokumenty:** `DATA_CONTRACT.md`, `docs/data_audit.md`, `docs/prompty.md` (P1, P3, P5)

---

## 1. Kontekst i problem

Prywatne repozytorium danych Hugging Face `lonwi/grzybobranie-zdjecia` osiągnęło limit bezpłatnej przestrzeni dyskowej **100 GB** (`used_storage = 100,005,916,916 B`). Serwery HF zablokowały operacje pobierania (download) i wysyłania (upload), zwracając kod `HTTP 403: Private repository storage limit reached`.

Audyt repozytorium wykazał:
1. 184 395 zdjęć w pełnej rozdzielczości zajmuje 92,37 GB (średnio ~501 kB / zdjęcie).
2. 246 osieroconych obiektów LFS z historii starych manifestów zajmuje 7,57 GB.
3. Przed nami do pobrania pozostaje jeszcze ~107 158 wierszy z paczki `species-65`, co przy starych rozmiarach wymagałoby kolejnych ~56 GB, daleko przekraczając limity konta.

---

## 2. Receptura normalizacji (Normalize-768)

Bot zdjęciowy wdraża standardową procedurę kompresji i standaryzacji zdjęć:
* Orientacja EXIF zaaplikowana w pikselach.
* Przestrzeń barw: konwersja profilu ICC do standardowego **sRGB**.
* Kanał alfa: spłaszczony na białym tle (RGB).
* Skalowanie: **dłuższy bok $\le 768$ px** (filtr Lanczos, brak upscalingu jeśli zdjęcie jest mniejsze).
* Format: JPEG jakość **85** z optymalizacją Huffmana (`optimize=True`).
* Metadane: usunięte bloki EXIF, ICC i XMP.

### Skutek objętościowy:
* Średni rozmiar zdjęcia: **~110–130 kB** (redukcja o ok. 75%).
* Cały zbiór 184 tys. zdjęć skurczy się z 92,4 GB do **~22,5 GB**.
* Po dojściu pozostałych zdjęć z `species-65` cały zbiór zajmie **~35 GB** (czyli zaledwie ~35% limitu konta 100 GB).

---

## 3. Zmiana kontraktu metadanych i obsługa w pipeline ML

Aby zapewnić całkowitą spójność z podziałami danych (P1) oraz kontrolą rozłączności ze zbiorem OOD v1:

1. **Nowa kolumna `original_sha256`:**
   - Przechowuje sumę kontrolną SHA-256 oryginalnego, nieskompresowanego pliku źródłowego.
   - Stanowi **niezmienny klucz referencyjny** powiązany z wpisami w `data/ood-v1/ood_ids.txt` oraz wygenerowanymi splitami `data/splits/*.csv`.

2. **Kolumna `sha256`:**
   - Staje się sumą kontrolną znormalizowanego pliku JPEG 768 px.

3. **Dodatkowe kolumny:**
   - `original_bytes`, `original_width`, `original_height`, `recipe` (np. `jpeg-q85-max768-srgb-noexif-v1`).

---

## 4. Zasada działania po stronie Grzybobranie ML

Wszystkie moduły pipeline'u ML (`prepare_data`, `cache_manager`, kontrola wycieku `leak_report.json`, weryfikacja OOD) stosują zasadę podwójnego sprawdzania:
```python
# Klucz do weryfikacji tożsamości i rozłączności z OOD:
check_hash = row.get("original_sha256") or row.get("sha256")
```
Gwarantuje to:
- 100% kompatybilność wsteczną z wygenerowanymi już splitami `data/splits/train.csv`, `val.csv`, `test.csv`.
- Pełną szczelność kontroli wycieków OOD v1 niezależnie od tego, czy plik jest w wersji oryginalnej, czy znormalizowanej 768 px.
