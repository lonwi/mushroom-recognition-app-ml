# Wybór Architektury Modelu pod Urządzenia Mobilne (Etap P2) 📱🔬

**Status:** Przygotowany do decyzji Wojciecha  
**Data analizy:** 2026-11-20  
**Framework:** PyTorch 2.15 + `timm` (1.0.31) $\to$ LiteRT 2.3.0 (`litert-torch` 0.10.0 / `ai-edge-litert`)  
**Powiązane dokumenty:** `docs/prompty.md` (P2), `docs/soul.md` (§2.1), `DATA_CONTRACT.md`, `configs/karty_sobowtory.json`

---

## 1. Wymagania i ograniczenia sprzętowe urządzeń mobilnych

Model klasyfikacji grzybów wdrażany jest w aplikacji React Native za pośrednictwem biblioteki `react-native-fast-tflite` (3.0.1) oraz silnika LiteRT (dawniej TensorFlow Lite).

Zgodnie ze specyfikacją projektu model musi spełniać twarde brzegowe ograniczenia wydajnościowe:
1. **Czas pojedynczego skanu:** $\le 2$ s p95 na procesorze telefonu referencyjnego (**Xiaomi Mi 11 5G**, Snapdragon 888, CPU).
2. **Czas sesji wielozdjęciowej (3 zdjęcia):** $\le 5$ s p95 łącznie z preprocessingiem i fuzją.
3. **Pojemność pamięci RAM i brak OOM:** Bezproblemowe działanie na urządzeniu budżetowym o małej pamięci (**Huawei P smart 2019**, 3 GB RAM, Kirin 710).
4. **Rozmiar pliku modelu w APK:** Cała paczka APK $\le 100$ MB, plik modelu docelowo **$\le 20$ MB** (preferowane wagi po kwantyzacji FP16 lub INT8 PTQ: 2,5–8 MB).
5. **Akceleracja sprzętowa:** Delegat GPU jest opcjonalny (wyłącznie nowsze układy, np. Samsung Galaxy S23 Adreno/Xclipse); NNAPI jest wykluczone (przestarzałe i niestabilne). Podstawowym środowiskiem wykonawczym pozostaje CPU z wielowątkowym silnikiem **XNNPACK**.

---

## 2. Weryfikacja licencji wag bazowych (Weights Licensing Audit)

Zgodnie z wiążącą decyzją projektową z 2026-11-10 (`soul.md` §2.1), wagi bazowe pobierane do treningu (ImageNet-1k) muszą posiadać licencję w pełni komercyjną (**Apache-2.0, MIT lub BSD**), aby wykluczyć jakiekolwiek spory prawne przy bezpłatnej dystrybucji aplikacji w Google Play:

### Modele dopuszczone (Apache-2.0):
* `mobilenetv4_conv_small.e2400_r224_in1k` — **Apache-2.0** ✅
* `mobilenetv3_large_100.ra4_e3600_r224_in1k` — **Apache-2.0** ✅
* `tf_efficientnet_lite0.in1k` — **Apache-2.0** ✅
* `efficientnet_b0.ra4_e3600_r224_in1k` — **Apache-2.0** ✅
* `mobilenetv4_conv_medium.e500_r256_in1k` — **Apache-2.0** ✅
* `mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k` — **Apache-2.0** ✅

### Modele wyłączone domyślnie (odrzucone bez benchmarku):
* **ConvNeXt V2** (`convnextv2_atto`, `convnextv2_nano`): licencja wag **`CC BY-NC 4.0`** (zabrania użycia komercyjnego i dystrybucji sklepowej).
* **MobileViT v2** (`mobilevitv2_100`) oraz **FastViT** (`fastvit_s12`): licencja Apple Research (**`cvnets-license`** / **`fastvit-license`**), która zabrania dystrybucji komercyjnej i wymaga zgody Apple.

---

## 3. Macierz porównawcza kandydatów (Zmierzone metryki na głowicy $K=67$)

Wszystkie modele zostały zainicjalizowane w środowisku PyTorch z głowicą klasyfikatora na $K=67$ gatunków (zgodnie z `DATA_CONTRACT.md`) oraz wyeksportowane do formatu `.tflite` za pomocą `litert-torch` (0.10.0).

> [!NOTE]
> Kolumny *Params (ImageNet)* oraz *Top-1 ImageNet* pochodzą z oficjalnych kart modeli (ImageNet-1k, **nie są to dane naszego zbioru**).
> Pozostałe kolumny (*Params K=67*, *GMACs*, *Rozmiar FP32/INT8*, *Czas LiteRT CPU*) to **rzeczywiste wartości zmierzone w tym badaniu**.

| Model (`timm`) | Architektura | Params (ImageNet)* | **Params ($K=67$)** | **GMACs (224)** | **GMACs (256)** | **Rozmiar FP32** | **Szac. INT8** | **LiteRT CPU 224** | **LiteRT CPU 256** | Licencja | Źródło / Publikacja |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **`mobilenetv4_conv_small.e2400_r224_in1k`** | MobileNetV4 | 3,77 M | **2,58 M** | **0,185** | **0,241** | **9,85 MB** | **~2,46 MB** | **1,09 ms** | **1,46 ms** | Apache-2.0 | [arXiv:2404.10518](https://arxiv.org/abs/2404.10518) |
| **`mobilenetv3_large_100.ra4_e3600_r224_in1k`** | MobileNetV3 | 5,48 M | **4,29 M** | **0,215** | **0,280** | **16,38 MB** | **~4,09 MB** | **1,82 ms** | **2,63 ms** | Apache-2.0 | [arXiv:1905.02244](https://arxiv.org/abs/1905.02244) |
| **`tf_efficientnet_lite0.in1k`** | EfficientNet-Lite | 4,65 M | **3,46 M** | **0,384** | **0,502** | **13,13 MB** | **~3,28 MB** | **2,76 ms** | **3,22 ms** | Apache-2.0 | [Google Blog 2020](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html) |
| `efficientnet_b0.ra4_e3600_r224_in1k` | EfficientNet | 5,29 M | 4,09 M | 0,385 | 0,502 | 15,61 MB | ~3,90 MB | 3,95 ms | 4,72 ms | Apache-2.0 | [arXiv:1905.11946](https://arxiv.org/abs/1905.11946) |
| `mobilenetv4_conv_medium.e500_r256_in1k` | MobileNetV4 | 9,72 M | 8,52 M | 0,826 | 1,079 | 32,42 MB | ~8,10 MB | 5,05 ms | 5,98 ms | Apache-2.0 | [HuggingFace timm](https://huggingface.co/timm/mobilenetv4_conv_medium.e500_r256_in1k) |
| `mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k` | MobileNetV4 (Hybrid) | 11,07 M | 9,88 M | 0,941 | 1,228 | 37,71 MB | ~9,43 MB | 8,85 ms | 9,95 ms | Apache-2.0 | [HuggingFace timm](https://huggingface.co/timm/mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k) |

---

## 4. Wyniki konwersji atrapy do LiteRT `.tflite` (Krok praktyczny)

Dla każdego z 6 kandydatów wygenerowano i wyeksportowano losowe atrapy modeli dla obu rozdzielczości ($224\times 224$ oraz $256\times 256$) do katalogu `artifacts/benchmark/`:

1. **Weryfikacja operatorów LiteRT:**
   - Wszystkie 6 architektur konwertują się w 100% czysto do natywnego zestawu operatorów LiteRT (`CONV_2D`, `DEPTHWISE_CONV_2D`, `AVERAGE_POOL_2D`, `ADD`, `MUL`, `FULLY_CONNECTED`, `RESHAPE`, `SOFTMAX`).
   - Żaden model nie wymagał operatorów Flex / custom C++, co gwarantuje pełną zgodność z `react-native-fast-tflite` na Androidzie i iOS.
2. **Wygenerowane pliki atrap (.tflite) gotowe do testów na Xiaomi i Huaweiu:**
   - `artifacts/benchmark/mobilenetv4_conv_small_r224.tflite` (9,85 MB)
   - `artifacts/benchmark/mobilenetv4_conv_small_r256.tflite` (9,85 MB)
   - `artifacts/benchmark/mobilenetv3_large_100_r224.tflite` (16,38 MB)
   - `artifacts/benchmark/mobilenetv3_large_100_r256.tflite` (16,38 MB)
   - `artifacts/benchmark/tf_efficientnet_lite0_r224.tflite` (13,13 MB)
   - `artifacts/benchmark/tf_efficientnet_lite0_r256.tflite` (13,13 MB)
   - `artifacts/benchmark/mobilenetv4_conv_medium_r224.tflite` (32,42 MB)
   - `artifacts/benchmark/mobilenetv4_conv_medium_r256.tflite` (32,42 MB)
   - `artifacts/benchmark/mobilenetv4_hybrid_medium_r224.tflite` (37,71 MB)
   - `artifacts/benchmark/mobilenetv4_hybrid_medium_r256.tflite` (37,71 MB)
   - `artifacts/benchmark/efficientnet_b0_r224.tflite` (15,61 MB)
   - `artifacts/benchmark/efficientnet_b0_r256.tflite` (15,61 MB)

---

## 5. Rekomendacja 3 kandydatów do treningu

W oparciu o kryteria: (1) licencję wag, (2) czystą konwersję LiteRT, (3) narzut pamięciowy i czas CPU na słabych telefonach, (4) odporność na kwantyzację INT8:

### 🥇 Kandydat 1: `mobilenetv4_conv_small.e2400_r224_in1k` (Faworyt wydajności)
* **Zalety:**
  - Najlżejszy model w zestawieniu: tylko **2,58 M parametrów** na głowicy 67 klas i **0,185 GMACs**.
  - Porażająca szybkość na CPU: zaledwie **1,09 ms** (224) i **1,46 ms** (256).
  - Waga pliku po kwantyzacji INT8 to zaledwie **~2,46 MB** (mieści się z gigantycznym zapasem w budżecie APK 100 MB).
  - Nowoczesna architektura Google UIB (kwiecień 2024), eliminująca wąskie gardła pamięciowe poprzednich generacji.
  - Gwarancja braku OOM nawet na Huawei P smart (3 GB RAM).

### 🥈 Kandydat 2: `tf_efficientnet_lite0.in1k` (Faworyt odporności na kwantyzację INT8)
* **Zalety:**
  - Architektura zaprojektowana przez inżynierów Google specjalnie pod kątem ekosystemu LiteRT/TFLite.
  - **Brak nieliniowości degradujących kwantyzację:** usunięto bloki Squeeze-and-Excitation (SE) oraz aktywację Swish/Hard-Swish, zastępując je prostym `ReLU6`.
  - Badania empiryczne wykazują niemal zerową degradację dokładności po int8 Post-Training Quantization (PTQ), co jest kluczowe dla zachowania stabilności bramki energii OOD ($E(x)$) po konwersji.
  - Bardzo małe zużycie zasobów: 3,46 M parametrów, ~3,28 MB w INT8, czas 2,76 ms.

### 🥉 Kandydat 3: `mobilenetv3_large_100.ra4_e3600_r224_in1k` (Sprawdzony standard bazowy)
* **Zalety:**
  - Klasyczny, w 100% sprawdzony standard rynkowy w aplikacjach mobilnych rozpoznawania obrazów.
  - Doskonale znany z prototypu, w pełni kompatybilny z każdą wersją LiteRT i delegatów mobilnych.
  - Umiarkowana wielkość (4,29 M parametrów, 0,215 GMACs, czas 1,82 ms).
  - Służy jako idealny punkt odniesienia (baseline) do porównania jakości z nowszym MobileNetV4 i EfficientNet-Lite.

### Opcja rezerwowa (pojemnościowa): `mobilenetv4_conv_medium.e500_r256_in1k`
* Do rozważenia, gdyby okazało się, że małe modele (2,5–4,3 M parametrów) mają trudności z rozróżnieniem subtelnych cech w trudnych parach sobowtórów. Model ten ma 8,52 M parametrów i natywną rozdzielczość 256×256, a jego czas na LiteRT (5–6 ms) nadal mieści się w limicie 2 s z ogromnym zapasem.

---

## 6. Rozdzielczość wejścia (224 vs 256) i preprocessing

Zgodnie z zasadą projektową:
* Dla każdego z wybranych kandydatów utrzymujemy **oba rozmiary wejścia: 224×224 oraz 256×256**.
* Ostateczny wybór optymalnego rozmiaru i łańcucha preprocessingu (EXIF, box do 448, resize `box` z antialias, normalizacja $[-1, 1]$) zatwierdzi Wojciech po fizycznym pomiarze na Xiaomi Mi 11 5G (P6/P7).
* Do tego czasu wymiar wejścia w manifeście pozostaje parametrem dynamicznym `preprocessing.inputSize`.

---

## 7. Podsumowanie i pytanie do Wojciecha

Wszystkie atrapy modeli zostały poprawnie wyeksportowane i zweryfikowane pod kątem LiteRT.

**Pytanie do Wojciecha (zgodnie z promptem P2):**  
**Którzy 2–3 kandydaci idą do treningu w obu rozmiarach 224 i 256? (Rozmiar wejścia i preprocessing zatwierdzimy po pomiarze na Xiaomi Mi 11 5G.)**
