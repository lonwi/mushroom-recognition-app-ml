# Raport ze Spike'a F0 / Spike B — Dekoder zdjęć i preprocessing na urządzeniu referencyjnym 📱🔬

**Data uruchomienia:** 2026-11-22T18:24:47.314Z  
**Urządzenie referencyjne:** Xiaomi Mi 11 Lite 5G  
**Środowisko:** Android / React Native / natywny dekoder + TypeScript resampling  
**Powiązane etapy:** P2 (architektura), P6/P7 (preprocessing i parytet), P8 (ship gates / budżety)

---

## 1. Konfiguracja testu Spike B

* **Seria pomiarowa:** 10 żądanych, 10 ukończonych (sukces: 10, nieudane: 0).
* **Próg krótszego boku:** 448 px (downscale dużych zdjęć).
* **Wejście modelu z manifestu:** $256 \times 256$ px.
* **Normalizacja:** `scale_offset` (odwzorowanie na $[-1, 1]$).
* **Architektura potoku:**
  - Dekodowanie obrazu: **natywne** z podpróbkowaniem sprzętowym.
  - Resampling i normalizacja: **TypeScript** (JavaScript thread / typed arrays).
  - *Uwaga:* Czas ściany obejmuje wyłącznie dekodowanie i preprocessing. Czas inferencji modelu LiteRT nie wchodzi w ten pomiar.

---

## 2. Pomiary czasowe (Latency)

| Przebieg | Dekodowanie (ms) | Preprocessing (ms) | Czas ściany (ms) |
|:---:|:---:|:---:|:---:|
| 1 | 410.8 | 811.3 | 1468.0 |
| 2 | 375.5 | 788.0 | 1353.3 |
| 3 | 382.6 | 799.8 | 1429.8 |
| 4 | 380.5 | 806.8 | 1451.6 |
| 5 | 377.8 | 801.7 | 1440.8 |
| 6 | 389.1 | 794.9 | 1426.0 |
| 7 | 399.2 | 796.7 | 1448.2 |
| 8 | 386.9 | 792.2 | 1386.3 |
| 9 | 385.3 | 792.4 | 1437.9 |
| 10 | 375.9 | 801.3 | 1430.6 |

### Statystyka zagregowana ($n=10$):
* **Czas ściany całkowity:**
  - Średnia: **1427.2 ms**
  - Mediana: **1434.3 ms**
  - Min / Max: 1353.3 ms / 1468.0 ms
  - **p95: 1468.0 ms**
* **Suma dekodowania natywnego:**
  - Średnia: **386.4 ms**
  - Mediana: 384.0 ms
  - Min / Max: 375.5 ms / 410.8 ms
  - **p95: 410.8 ms**
* **Suma preprocessingu (TS):**
  - Średnia: **798.5 ms**
  - Mediana: 798.2 ms
  - Min / Max: 788.0 ms / 811.3 ms
  - **p95: 811.3 ms**

---

## 3. Pomiary pamięciowe (RAM & Heaps)

* **Szczyt PSS (Proportional Set Size):** **210 678 kB (205.7 MB)**
* **Szczyt sterty natywnej (Native Heap):** **107 840 kB (105.3 MB)**
* **Szczyt sterty Javy (Java Heap):** **29 166 kB (28.5 MB)**
* **Błędy OOM (Out-of-Memory):** **0 (nie zgłoszono)**

---

## 4. Kryteria brzegowe do bramek wydań (Ship Gates P8)

Na podstawie wyników Spike B formalizuje się następujące reguły ewaluacji budżetów sprzętowych:

1. **Urządzenie referencyjne: Xiaomi Mi 11 Lite 5G**
   - **Szczyt PSS:** $\le 400$ MB na trzy zdjęcia.  
     *(Zmierzone: 205.7 MB — kryterium spełnione z zapasem 194.3 MB).*
   - **Budżet sesji 3 zdjęć:** $\le 5$ s p95 czasu ściany.  
     *(Do budżetu wlicza się p95 czasu ściany dekodowania + preprocessingu. Inferencja modelu LiteRT na poziomie 1–3 ms dodaje pomijalny narzut).*
   - **Budżet pojedynczego skanu:** $\le 2$ s p95.  
     *(Zmierzone p95 ściany dla 1 zdjęcia: 1468.0 ms — mieści się w limicie 2.0 s).*

2. **Urządzenie budżetowe (Worst-case): Huawei P smart 2019 (3 GB RAM)**
   - **Brak OOM i brak ubicia procesu:** test musi zakończyć się bez awarii i bez zamknięcia aplikacji przez systemowego zabójcę pamięci (Linux Low Memory Killer / LMK).
   - **Czas:** brak twardego progu czasowego (priorytetem jest stabilność procesu).
   - *Uwaga operacyjna:* Jeżeli aplikacja nagle zniknie w tle, ubicie przez system należy odnotować jako błąd krytyczny `PROCESS_KILLED_BY_LMK`.
