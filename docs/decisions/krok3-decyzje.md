# Krok 3 — Decyzje (wspólnie podjęte z Wojciechem, 2026-11-10)

## 1. Głowica (Head)
- **Do 67 wyjść** gatunków mini+ (docelowo 67). Brak klas zbiorczych (`not_a_mushroom`, `unknown_mushroom` nie wchodzą do głowicy).
- Progi minimalnej liczby zdjęć treningowych, aby gatunek został w głowicy: ≥ 55 dla `high_stakes` / toksycznych, ≥ 40 dla pozostałych (to nie są progi bramki energii/OOD).

## 2. Aplikacja
- Bezpłatna, bez zakupów w aplikacji. Zdjęcia treningowe CC BY-NC wolno używać do treningu (w UI tylko CC0 i CC BY).
- Wagi bazowe: **Apache-2.0 / MIT / BSD** (MobileNetV3/V4, EfficientNet-Lite itd.). Wagi CC BY-NC (ConvNeXt V2, Apple "other") są wyłączone domyślnie.
- Działanie na telefonie przez `react-native-fast-tflite` 3.0.1 (LiteRT 2.1.3), na CPU jako ścieżka produkcyjna.

## 3. Środowisko obliczeniowe
- Trening lokalny na laptopie **RTX 5070 Laptop 8 GB VRAM**, Windows 11 + WSL2 (Ubuntu).
- PyTorch **Stable 2.15.0 + CUDA 13.2** (indeks `cu132`), sm_120 (Blackwell).
- Instalacja wewnątrz WSL2 (koła Linux), nie w Pythonie Windows.

## 4. Nazwa bota
- **"Trenowanie modelu Grzybobranie AI"**.

## 5. Rozmiar wejścia
- Sporządzenie rozdzielczości 224 vs 256. Decyzja zatwierdzona przez Wojciecha na podstawie pomiaru wydajności i dokładności na **Xiaomi Mi 11 5G**.

## 6. Gatunki mini+ bez zdjęć
- Gatunki bez wystarczających zdjęć są **poza głowicą**, nie jako wyjścia z `in_model: false`.
- Głowica zawiera tylko gatunki z danymi spośród 67. Pozostałe z listy 67 są w `uncoveredSpecies` (główna funkcja łagodnego degradowania aplikacji).

## 7. DANGEROUS_PAIRS
- Bot proponuje listę z `karty_sobowtory.json` (3443 linii, pary niebezpieczne DANGEROUS_GENERA/DANGEROUS_PAIRS).
- Listę zatwierdził Wojciech.

## 8. Artefakty modelu
- Artyfakty (manifest, `.tflite`) trzymane w **prywatnym repozytorium HF**: `HF_MODEL_REPO` (propozycja nazwy).
- Dane zdjęciowe w `HF_DATASET_REPO`.

## 9. Zakaz wag DF20/DF24
- Zakaz obejmuje również wagi wytrenowane/dotrenowane na tych zbiorach (np. checkpointy `*ft_df20*`). Nie używamy ani zdjęć, ani wag dotrenowanych.

## Dodatkowe decyzje (Faza 0 — Q12–Q20, decyzje-f0.md)
- Trening poza repozytorium: kontrakt prototypu zamrożony w `ml/OUTDATED.md` (fixture'y, `ship_gates.py`, moduły, test_contract.py).
- Zmiana ścieżek instalacji do `src/services/*`, wołania `fetch_gbif` i `prepare_data` wypadła z testu.
- Liczby 54 wyjścia, `not_a_mushroom`, `unknown_mushroom`, `class_cap=2500` oznaczone jako przestarzałe.
- Manifest v1: do 67 gatunkowych wyjść, bez klas zbiorczych, gatunki bez danych w `uncoveredSpecies`.
- Limit 3 zdjęcia / obserwacja już zastosowany w źródłach; tu nie ścinamy ponownie.
- OOD filtr: metadane lokalnie w `data/ood-v1/ood_ids.txt` oraz dane w prywatnym HF `data/ood-v1/`, z kopią SHA-256 na OneDrive.
- Pomoc trasa w Fazie 0 to `help` (`src/app/(tabs)/help.tsx`); numery alarmowe nie są w tej wersji.

## Podsumowanie sumy kontrolne (pokrycie-67)
| Źródło | Klas | Wiersze źródłowe | OOD odfiltrowane | total po filtrze | ui_safe |
|---|---|---|---|---|---|
| v3 | 51 | 74 040 | 0 | 74 040 | 28 043 |
| new16 | 14 | 13 863 | 0 | 13 601 | 1 993 |
| new16+addendum | 2 | 31 (addendum) | 0 | 293 | 22 |
| **razem** | **67** | **87 934** | **0** | **87 934** | **30 058** |

Rozkład licencji (po filtrze = źródło): CC0 6 795, CC BY 23 263, CC BY-NC 40 887, CC BY-SA 16 761, pd-self 228.

## Gatunki najcieńsze (informacyjnie)
- `amanita_verna`: 67 wierszy, próg 55, high-stakes=true, ui_safe=8.
- `lepiota_brunneoincarnata`: 226, próg 55, high-stakes=true, ui_safe=14.
- `inocybe_erubescens`: 180, próg 55, high-stakes=true, ui_safe=24.
- `entoloma_sinuatum`: 391, próg 55, high-stakes=true, ui_safe=37.
- `russula_virescens`: 699, próg 55, high-stakes=true, ui_safe=57.

## SHA-256
- `pokrycie-67.csv`: `48f67e65fea2a4d0a1c21a85fb3d4103c887a885a4c16fcb454aa9d52459f7ea`
- `karty_sobowtory.json`: `2127ecc0d0e8ca88b468019fcb77758d2e70d2d2c92fae7dd20febbf2886ef15`

## Walidator
- `python3 scripts/validate_pokrycie-67.py` — 67 wierszy, zbiór klas = 51 `_v3` + 16 new16, sumy licencji zgodne.

## Status decyzji
- Decyzje z 2026-11-10 są wiążące (soul.md §2.1).
- Niektóre pozycje otwarte (pomiar 224/256 na Xiaomi, gatunki bez zdjęć poza głowicą, DANGEROUS_PAIRS akceptacja Wojciecha, osobne repo modeli, zakaz wag DF20/DF24) wymagają dalszej uzgodnienia.
