# Prompty dla bota „Trenowanie modelu Grzybobranie AI”

> Każdy prompt wklejasz jako osobną wiadomość, w podanej kolejności. Każdy kończy się pytaniem do Wojciecha, a bot nie przechodzi dalej bez odpowiedzi.
> Wspólne źródła: `soul.md` (w tym §2.1 decyzje z 2026-11-10 i §4 środowisko WSL2), `model-manifest.schema.json`, `docs/decisions/krok3-decyzje.md` (U1–U8, D-07…D-13), `krok4-plan.md` §7 (M0–M5), prototyp `ml/` (`split.py`, `preprocess.py`, `recognition_math.py`, `export_tflite.py`, `ship_gates.py`, `score_delegate_dump.py`), OOD `data/ood-v1/`.
> Zamknięte (2026-11-10, soul.md §2.1): do 67 wyjść (docelowo 67), bez klas zbiorczych, gatunki bez danych poza głowicą; aplikacja bezpłatna bez IAP; trening na lokalnym RTX 5070 Laptop 8 GB w WSL2; 224 vs 256 i preprocessing zatwierdza Wojciech po pomiarze na Xiaomi Mi 11 5G; DANGEROUS_PAIRS z karty_sobowtory.json po akceptacji; artefakty w osobnym prywatnym repo HF (propozycja: HF_MODEL_REPO); zakaz DF20/DF24 także dla wag; nazwa bota jak w tytule.

---

## P0. Rozruch, środowisko i inwentaryzacja (jednorazowo)

```
Przeczytaj soul.md (całość, zwłaszcza §2.1 i §4), model-manifest.schema.json, docs/decisions/krok3-decyzje.md (U1–U8, D-07–D-13) i krok4-plan.md §7.

Decyzje już podjęte (2026-11-10), nie pytaj o nie ponownie:
- do 67 wyjść gatunków mini+ (docelowo 67) plus próg energy/OOD; bez not_a_mushroom i unknown_mushroom; gatunki bez danych poza głowicą (uncoveredSpecies), nie in_model: false;
- aplikacja bezpłatna, bez zakupów w aplikacji; wagi bazowe domyślnie tylko Apache-2.0/MIT/BSD;
- trening na lokalnym GPU (RTX 5070 Laptop 8 GB, 32 GB RAM, Windows 11 + WSL2);
- 224 vs 256 i preprocessing: oba warianty aż do pomiaru na Xiaomi Mi 11 5G i akceptacji Wojciecha;
- zakaz DF20/DF24 obejmuje też wagi (*ft_df20* itd.);
- artefakty w osobnym prywatnym repo HF (propozycja: HF_MODEL_REPO);
- nazwa bota: „Trenowanie modelu Grzybobranie AI”.

Najpierw środowisko w WSL2 (soul.md §4): sterownik NVIDIA tylko w Windows, w WSL bez sterownika Linux; PyTorch ze wsparciem sm_120. Oficjalne stable na 2026-11-10 to 2.15.0, koło CUDA 13.2:
  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
(źródło: https://pytorch.org/get-started/locally/; Linux + CUDA 13.2 w selektorze to też „pip3 install torch torchvision” z PyPI). Python ≥ 3.11. Zweryfikuj torch.__version__, torch.cuda.is_available(), nazwę GPU, torch.cuda.get_arch_list() (musi być sm_120) i krótkie mnożenie na CUDA. Zbiór i cache tylko na ext4 WSL, nie na /mnt/c. Trackio loguj lokalnie. HF Jobs nie uruchamiaj.

Potem przejrzyj prototyp <app-repo>/ml/ (albo github.com/<your-org>/<repo>, katalog ml/) i opisz w tabeli:
- które moduły nadają się do ponownego użycia (split.py, dedup.py, preprocess.py, recognition_math.py, ship_gates.py, score_delegate_dump.py, export_tflite.py, license_policy.py),
- co trzeba zmienić pod mini+ 67 klas bez klas zbiorczych.

Jawno zaznacz, że prototyp jest nieaktualny wobec tej decyzji i wymaga przebudowy, nie tylko „dostosowania liczb”:
- ml/DATA_CONTRACT.md (ok. l. 98) i ml/AGENTS.md opisują 54 wyjścia (52 gatunki + unknown_mushroom + not_a_mushroom) oraz MobileNetV3-Small w Keras — nieaktualne; nowy kontrakt to do 67 wyjść (docelowo 67), bez klas zbiorczych;
- ml/README.md i labels.json: fetch class_cap 2500 dla każdej z klas zbiorczych; sampling.py ma AGGREGATE_CLASS_IDS;
- prepare/fetch (prepare_data.py, fetch_gbif.py) budują te klasy — tę logikę usuwamy, nie przenosimy. Wcześniejsza wzmianka o limicie 3693: do weryfikacji w prepare_data.py; nawet jeśli się znajdzie, nie zostaje w planie;
- DANGEROUS_GENERA w prototypie ma tylko 4 rodzaje (do uzupełnienia w manifeście: co najmniej Amanita, Galerina, Gyromitra, Inosperma, Cortinarius, Lepiota);
- recognition_math.py / ship_gates.py / manifest.py nadal znają klasy zbiorcze — trzeba je zastąpić bramką energy/OOD.

Sprawdź dostęp: hf whoami, odczyt HF_DATASET_REPO (bez zapisu), obecność data/ood-v1/ i lokalnego data/ood-v1/ood_ids.txt, GitHub (odczyt), OneDrive (odczyt).
Nie trenuj i nie pobieraj dużych plików. Zakończ listą ryzyk i pytaniem: „Które moduły przenosimy i w jakiej kolejności przerabiamy prepare/fetch, DATA_CONTRACT.md i recognition_math.py na kontrakt do 67 wyjść (docelowo 67) bez klas zbiorczych, z gatunkami bez danych poza głowicą?”
```

## P1. Audyt danych i podział train/val/test bez wycieku

```
Cel: zbiór mini+ (lista 67 gatunków; głowica = tylko klasy z wystarczającymi danymi) z podziałem bez wycieku. Bez klas zbiorczych i bez logiki fetch cap 2500 / limitów not_a_mushroom i unknown_mushroom.
1. Wczytaj najnowszą paczkę od bota „Zbieranie informacji o grzybach” (np. _v3) z HF_DATASET_REPO. Podaj hash plików CSV. Liczby wierszy bierz z plików, nie z pamięci.
2. Audyt dla każdej z 67 klas mini+: liczba zdjęć, obserwacji i autorów; rozkład licencji; odsetek BY-NC/BY-SA/pd_self; maks. zdjęć na obserwację; źródła. Odrzuć DF20/DF24 (także wagi *ft_df20*) i każdy wiersz bez licencji z dozwolonej listy (CC0, CC BY, BY-NC, BY-SA, pd_self). Zgłoś klasy z <100 zdjęciami lub <20 obserwacjami. Klasy bez wystarczających zdjęć **nie wchodzą do głowicy** — trafiają na listę uncoveredSpecies (Atlas je pokazuje, skaner ich nie zwraca). Nie twórz wyjść z in_model: false.
3. Deduplikacja: dokładna (SHA-256) i perceptualna (pHash/dHash, próg do ustalenia) w obrębie klasy i między klasami. Zdjęcia z różnych klas z tym samym hashem to podejrzane etykiety.
4. Podział 70/15/15 grupowany jak w ml/split.py: spójne składowe po autorze, obserwacji (iNat/MO/GBIF) i zdjęciu, z kluczem HMAC. Stratyfikuj po klasie, na ile pozwalają grupy.
5. Kontrole wycieku: zero wspólnych grup, URL-i, hashy i near-duplikatów między splitami; rozłączność z OOD v1 po URL, ID obserwacji i hashu na podstawie data/ood-v1/ood_ids.txt (kopia: HF HF_DATASET_REPO, data/ood-v1/); brak rycin i obrazów generowanych (D-17). Nie ma źródeł „klas zbiorczych” do odfiltrowania — tych klas nie budujemy.
6. Policz, ile obserwacji w val/test ma ≥2 i ≥3 zdjęcia. Te dane są potrzebne do kalibracji fuzji (D9 §8.7).
Wynik: data_audit.md + splits/*.csv + leak_report.json, wszystko z SHA-256, na ext4 WSL. Nie wysyłaj nic do innych botów.
Zakończ pytaniem: „Akceptujesz podział i listę klas do dozbierania? Mam przygotować prośbę do bota zdjęciowego?”
```

## P2. Wybór architektury pod telefon

```
Cel: wybrać 2–3 kandydatów do treningu. Zmieść się w: skan ≤2 s p95 (Xiaomi Mi 11, CPU), sesja 3 zdjęć ≤5 s p95, APK ≤100 MB łącznie z resztą aplikacji (model najlepiej ≤20 MB), działanie bez OOM na Huawei P smart 2019 (3 GB RAM). Delegat GPU jest opcjonalny (S23), NNAPI nie wchodzi w grę.

Licencja wag (decyzja 2026-11-10): aplikacja jest bezpłatna, bez IAP, ale dystrybucja przez Google Play może być sporna jako użytek komercyjny. Do porównania i treningu bierz tylko wagi Apache-2.0 / MIT / BSD. ConvNeXt V2 (CC BY-NC 4.0) i MobileViT/FastViT (licencja Apple „other”) są wyłączone domyślnie — możesz je krótko wymienić jako odrzucone, bez benchmarku, chyba że Wojciech odblokuje wyjątek.

Porównaj co najmniej te rodziny (źródła i dane z model card poniżej; sprawdź, czy są aktualne):

| Model (timm) | Params | GMACs | Rozdz. | Licencja wag | Źródło |
|---|---|---|---|---|---|
| mobilenetv3_large_100.ra4_e3600_r224_in1k | 5,5 M | 0,2 | 224 | Apache-2.0 | https://arxiv.org/abs/1905.02244 · https://huggingface.co/timm/mobilenetv3_large_100.ra4_e3600_r224_in1k |
| mobilenetv4_conv_small.e2400_r224_in1k | 3,8 M | 0,2 | 224 | Apache-2.0 | https://arxiv.org/abs/2404.10518 · https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k |
| mobilenetv4_conv_medium.e500_r256_in1k | 9,7 M | 1,1 | 256 | Apache-2.0 | https://huggingface.co/timm/mobilenetv4_conv_medium.e500_r256_in1k |
| mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k | 11,1 M | 1,3 | 256 | Apache-2.0 | https://huggingface.co/timm/mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k |
| tf_efficientnet_lite0.in1k | 4,7 M | 0,4 | 224 | Apache-2.0 | https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html · https://huggingface.co/timm/tf_efficientnet_lite0.in1k |
| efficientnet_b0.ra4_e3600_r224_in1k | 5,3 M | 0,4 | 224 | Apache-2.0 | https://arxiv.org/abs/1905.11946 · https://huggingface.co/timm/efficientnet_b0.ra4_e3600_r224_in1k |

Wyłączone domyślnie (nie trenuj): convnextv2_atto / convnextv2_nano.fcmae_ft_in22k_in1k (CC BY-NC 4.0, https://huggingface.co/timm/convnextv2_nano.fcmae_ft_in22k_in1k); mobilevitv2_100.cvnets_in1k / mobilevit_s oraz FastViT (licencja Apple „other”).

Kryteria (w tej kolejności): (1) licencja wag zgodna z pulą Apache-2.0/MIT/BSD, (2) konwersja do .tflite bez operatorów spoza wbudowanego zestawu LiteRT (sprawdź w LiteRT 2.1.3 / fast-tflite 3.0.1), (3) zmierzony czas CPU i pamięć, (4) dokładność po fine-tuningu na naszym val, (5) odporność na int8 (zmiana energii i zgodności top-1).
Rozmiar wejścia 224 vs 256 i łańcuch preprocessingu nie są zamykane w tym kroku. Dla każdego kandydata przygotuj **oba** rozmiary (jeśli architektura ma natywne 224 i 256, użyj obu; inaczej przeskaluj ten sam pipeline). Baseline preprocessingu to prototyp (EXIF, box do 448, potem NxN, `/127.5 − 1`) — też podlega ocenie. Ostateczną parę (N + preprocessing) zatwierdza Wojciech po pomiarze wydajności i dokładności na Xiaomi Mi 11 5G (P6/P7). Do tego czasu N w manifeście zostaje parametrem, nie stałą M0.
Krok praktyczny: wyeksportuj każdego kandydata z losowymi wagami do .tflite w 224 i 256 (litert-torch, https://github.com/google-ai-edge/litert-torch) i zmierz benchmark_model / ai-edge-litert na boxie albo w WSL. Poproś Architekta (przez Wojciecha) o wstępny pomiar atrap o obu kształtach na Xiaomi i Huaweiu; pełne porównanie wytrenowanych modeli jest w P7.
Nie podawaj dokładności na naszych danych, dopóki jej nie zmierzysz. Wartości ImageNet z model card oznacz jako „ImageNet, nie nasze dane”. Params/GMACs z tabeli potwierdź w model card w dniu użycia.
Zakończ pytaniem: „Którzy 2–3 kandydaci idą do treningu w obu rozmiarach 224 i 256? (Rozmiar wejścia i preprocessing zatwierdzimy po pomiarze na Xiaomi Mi 11 5G.)”
```

## P3. Trening i fine-tuning

```
Cel: wytrenować wybranych kandydatów na splitach z P1. Najpierw model pośredni (klasy z wystarczającymi danymi), potem docelowo mini+ z K ≤ 67 wyjść (docelowo 67) bez klas zbiorczych. Gatunków bez danych nie wstawiaj do głowicy.
- Framework: timm + PyTorch (skill huggingface-vision-trainer, przepisy adaptuj do maszyny lokalnej). Główna ścieżka: laptop RTX 5070 Laptop 8 GB VRAM, 32 GB RAM, WSL2 (soul.md §4). HF Jobs tylko jako zapas i tylko po moim „tak” na konkretne zadanie i koszt. Przed startem podaj szacunek czasu GPU na laptopie (nie zmyślaj; jeśli nie mierzyłeś, napisz „szacunek / do weryfikacji”).
- Przed runem: mixed precision (bf16 jeśli wspierane, inaczej fp16 + GradScaler); batch i gradient accumulation tak, żeby zmieścić się w 8 GB osobno dla 224 i 256 (wartości startowe z soul.md §4 są do weryfikacji); DataLoader vs 32 GB RAM; dane na ext4, nie /mnt/c; checkpoint + resume; Trackio lokalnie. Do akceptacji Wojciecha trenuj **oba** rozmiary.
- Preprocessing treningowy: baseline z prototypu (EXIF, box do 448, potem NxN, uint8 z zaokrągleniem połówek w górę, `/127.5 − 1`), spójny we wszystkich runach 224 i 256, dopóki pomiar na Xiaomi go nie zmieni. Augmentacje tylko przed tym krokiem lub zgodne z nim. Pochodzą z prawdziwych warunków: kadr, jasność, rozmycie, kompresja JPEG, obrót.
- Etapy: (a) zamrożony backbone, sama głowica; (b) pełny fine-tuning z niższym LR, cosine, warmup, label smoothing ≤0,1 (zanotuj, bo wpływa na kalibrację), EMA wag, class-balanced sampling lub loss z wagami dla słabych klas.
- Głowica: tyle wyjść, ile klas z danymi (K ≤ 67, docelowo 67), bez rejectClasses, bez not_a_mushroom/unknown_mushroom i bez wyjść in_model: false. OOD jest poza softmaxem, bramką energii. Listę gatunków mini+ poza głowicą zapisz od razu (pójdzie do uncoveredSpecies).
- Logi w Trackio (lokalnie): loss, top-1/top-5 macro i per klasa, recall klas wysokiego ryzyka, macierz pomyłek dla par niebezpiecznych.
- Wybór checkpointu tylko na val. Test zostaje nietknięty do P7.
Wynik: checkpoint + config + lokalny Trackio + SHA-256 + tabela val.
Zakończ pytaniem: „Który checkpoint idzie do kalibracji? Czy któraś klasa wymaga dozbierania danych przed następnym runem?”
```

## P4. Kalibracja temperaturą i ECE

```
Cel: wypełnić sekcję calibration manifestu.
- Temperature scaling (Guo i in. 2017, https://arxiv.org/abs/1706.04599): dopasuj T na val (NLL, LBFGS), zgłoś T.
- ECE (15 binów, równa szerokość) i MCE przed i po kalibracji na val i na teście, z przedziałem bootstrap 95%. Do tego reliability diagram (PNG) i ECE osobno dla 8 klas śmiertelnych.
- Policz kalibrację na wyjściu .tflite, a nie tylko w PyTorch. Kwantyzacja może ją zmienić (patrz P6, wróć tutaj po eksporcie).
- Zaproponuj granice przedziałów słownych dla D-10 (wysoka/średnia/niska zgodność) i próg ECE, przy którym aplikacja pokazuje procent zaokrąglony do 5 pp.
Jeśli ECE na teście przekracza proponowany próg, ustaw calibrated: false i napisz dlaczego.
Zakończ pytaniem: „Akceptujesz T, próg ECE i granice przedziałów?”
```

## P5. Bramka OOD energy i progi fuzji

```
Cel: wypełnić ood.* i ood.fusion["2"], ["3"].
- Energia: E(x) = −T · logsumexp(logits / T) (Liu i in. 2020, https://arxiv.org/abs/2010.03759). Kolejność reguł jak w D9 §8.6, bez klas zbiorczych: (1) E > energyThreshold → rejected/out_of_distribution, (2) top-1 < minTop1Prob → rejected/unclear, (3) w pozostałych przypadkach candidates + dangerous_genus/low_confidence.
- Dane: część kalibracyjna data/ood-v1/ (lokalne metadane: data/ood-v1/; HF: HF_DATASET_REPO, data/ood-v1/) do wyboru progów, część testowa tylko do raportu. Najpierw potwierdź rozłączność z treningiem (P1, krok 5, plik ood_ids.txt). Liczby zdjęć i taksonów bierz z tych artefaktów, nie z pamięci.
- Raport: AUROC, FPR@95%TPR, odsetek zachowanych zdjęć znanych klas (≥0,95 jak w D9 §8.7) i wynik osobno dla groźnych sobowtórów spoza modelu (Lepiota subincarnata, L. castanea, Clitocybe rivulosa, Tricholoma pardinum…). Policz też „confident edible” na OOD: grzyb spoza modelu rozpoznany z wysoką pewnością jako gatunek jadalny.
- Fuzja (D9 §8.7): p_v = softmax(z_v/T), H_v = −Σ p ln p, c_v = 1 − H_v/ln K, w_v = c_v/Σc, z̄ = Σ w_v z_v. K = liczba wyjść głowicy (≤ 67, docelowo 67). Progi dla m=2 i m=3 osobno, na grupach zdjęć tej samej obserwacji. Bramki fuzji takie same jak dla jednego zdjęcia oraz brak spadku macro top-1 względem jednego zdjęcia. Bramka OOD działa na każdym zdjęciu PRZED fuzją.
- Wygeneruj fixture'y decision_cases.json i fusion_cases.json (logity → oczekiwana decyzja) dla testów P2 w aplikacji.
Jeśli nie ma danych do progu dla danego m, nie wpisuj go. Aplikacja wtedy ukryje fuzję.
Zakończ pytaniem: „Akceptujesz progi i kompromis między odrzuceniami a przepuszczeniem OOD?”
```

## P6. Eksport do .tflite i kwantyzacja

```
Cel: pliki .tflite działające w react-native-fast-tflite 3.0.1 (LiteRT 2.1.3, CPU, delegates: []), osobno dla wejścia 224 i 256, aż Wojciech zatwierdzi rozmiar po pomiarze na Xiaomi Mi 11 5G.
- Konwersja: litert-torch (https://github.com/google-ai-edge/litert-torch, https://developers.google.com/edge/litert/conversion/pytorch/overview). Wejście NHWC float32 [1,N,N,3] (opakuj model w permute, jeśli trzeba), wyjście surowe logity [1,K] (bez softmaxu w grafie, bo energia potrzebuje logitów; K = liczba wyjść głowicy, ≤ 67, docelowo 67). Zapisz wersję konwertera i LiteRT.
- Dla każdego rozmiaru N ∈ {224, 256} przygotuj warianty kwantyzacji: fp32, fp16 (domyślna ścieżka z prototypu), dynamic-range, int8. Do kalibracji int8 użyj tylko zdjęć z train (jak export_tflite.py), nigdy val/test/OOD.
- Dla każdego wariantu (N × kwantyzacja) podaj: rozmiar pliku, listę operatorów (bez Flex/SELECT_TF_OPS), zgodność top-1 z float32 na teście (≥0,99 całość i ≥0,99 klasy wysokiego ryzyka), zmianę energii i ECE, czas na boxie lub w WSL.
- Do testu na telefonach wyślij te warianty kwantyzacji, które przechodzą bramki, w **obu** rozmiarach 224 i 256. Jeśli kwantyzacja nie przechodzi, zostaw fp16/fp32 dla danego N. Progu nie obniżaj. N w manifeście zamykamy dopiero po P7 i akceptacji Wojciecha.
- Potwierdź, że pliki ładują się w ai-edge-litert w wersji zgodnej z 2.1.3 (lub najbliższej dostępnej; zapisz, którą sprawdzałeś).
Zakończ pytaniem: „Które warianty kwantyzacji (dla 224 i dla 256) wysyłamy do pomiaru na Xiaomi Mi 11 5G?”
```

## P7. Test parytetu z aplikacją (P1–P3)

```
Cel: model na telefonie widzi to samo co w Pythonie, a Wojciech może zatwierdzić rozmiar wejścia i preprocessing na podstawie pomiaru na Xiaomi Mi 11 5G.
- P1 preprocessing: wygeneruj fixture'y (zdjęcie → tensor uint8 i float) **dla 224 i dla 256** na co najmniej 3 zdjęciach, w tym jednym z obrotem EXIF i jednym szerszym niż 448 px. Baseline to łańcuch prototypu (EXIF, box 448, NxN, `/127.5 − 1`). Format zgodny z ml/fixtures/ z prototypu.
- P2 decyzja i fuzja: decision_cases.json i fusion_cases.json z P5 (osobno, jeśli logity zależą od N).
- P3 na urządzeniu, **osobno dla 224 i 256**: lista „złotych zdjęć” (≥200, z klasami wysokiego ryzyka). Architekt uruchamia autotest w buildzie deweloperskim na Xiaomi Mi 11 5G (CPU; to pomiar decydujący), Huaweiu (CPU, smoke) i S23 (CPU i GPU), a zrzut logitów oceniasz jak score_delegate_dump.py: zgodność decyzji ≥0,99. Dla S23 GPU zapisz raport w delegates z sha256 modelu.
- Dla każdego N i telefonu podaj czasy p50/p95 dla 1 i 3 zdjęć, szczyt pamięci (dane od Architekta, nie szacunki) oraz dokładność / zgodność na złotych zdjęciach. Zestaw 224 vs 256 (i ewentualne uwagi do preprocessingu prototypu) w jednej tabeli.
Jeśli P3 nie przechodzi, popraw preprocessing lub dekoder i skalibruj od nowa. Progu nie ruszaj.
Zakończ pytaniem: „Wysłać paczkę fixture'ów (224 i 256) i listę złotych zdjęć do Architekta? Po pomiarze na Xiaomi: który rozmiar wejścia i jaki preprocessing zatwierdzasz do manifestu?” (wysyłka i zamknięcie N tylko po zgodzie).
```

## P8. Ewaluacja ship gates i karta modelu

```
Cel: werdykt D-13 dla kandydata (M4/M5) na teście i ood-test, uruchomiony jednorazowo na zamrożonym checkpoincie i pliku .tflite.
Bramki (fail closed, brak metryki = niezaliczona):
1. confident_toxic_as_edible = 0 we wszystkich klasach trujących,
2. wskaźnik pomyłek w każdej parze niebezpiecznej ≤ 0,05 (DANGEROUS_PAIRS: bot proponuje listę z karty_sobowtory.json, Wojciech zatwierdza; bez akceptacji bramki nie zamykaj),
3. recall klas wysokiego ryzyka ≥ 0,85 przy n ≥ 8 (jak w ship_gates.py) i Wilson dolna granica w raporcie,
4. zgodność TFLite vs float ≥ 0,99 (całość i wysokie ryzyko), P3 ≥ 0,99 na Xiaomi,
5. OOD: zachowanie znanych ≥ 0,95, open-set confident edible ≤ 0,02 (górna granica 95% ≤ 0,05),
6. bramki fuzji z P5,
7. budżety: ≤2 s / ≤5 s p95 Xiaomi, Huawei bez awarii, rozmiar modelu w budżecie APK.
Karta modelu (docs/model-card.md; D10 l. 215, 296; Mitchell i in. 2019, https://arxiv.org/abs/1810.03993): przeznaczenie i ograniczenia („podpowiedź, nie ocena jadalności”), dane i licencje (w tym BY-NC tylko w treningu), podział, metryki per klasa z n_test, osobna tabela 8 klas śmiertelnych i słabych klas, kalibracja, OOD, walidacja z podziałem na miejsce, fotografa i czas, znane awarie, wersje, hashe, licencja wag bazowych.
Wynik: ship_gates_report.json + model-card.md. Zostaw pola na podpis Wojciecha i wynik zamkniętej bety puste.
Zakończ pytaniem: „Czy podpisujesz kandydata jako recenzent, czy wracamy do treningu? Niezaliczone bramki: …”
```

## P9. Wydanie manifestu

```
Cel: paczka wydania dla Architekta.
- Zbuduj model-manifest.json wg model-manifest.schema.json v1: schemaVersion, modelVersion (semver; pośrednie z -test lub channel "test"), sha256 i rozmiar .tflite, preprocessing (N i łańcuch już zatwierdzone po P7), classes (tylko gatunki w głowicy, wszystkie in_model: true; K ≤ 67, docelowo 67), uncoveredSpecies (pozostałe gatunki mini+ poza głowicą; K + |uncoveredSpecies| = 67), calibration, ood (+fusion), fusionWeights, dangerousGenera (co najmniej Amanita, Galerina, Gyromitra, Inosperma, Cortinarius, Lepiota), perClassMetrics, shipGates, delegates, minAtlasVersion, provenance (w tym modelRepo i modelRevision).
- Pola rejectClasses nie dodawaj. W tym projekcie jest nieużywane (decyzja 2026-11-10): odrzucanie wyłącznie progami OOD. Nie umieszczaj niepokrytych gatunków w classes z in_model: false — tylko uncoveredSpecies. Schemat nadal ma opcjonalne rejectClasses i wymaga in_model przy każdym wpisie classes (w tym projekcie zawsze true). Wydanie z klasami zbiorczymi albo martwymi wyjściami jest błędem.
- Waliduj JSON Schema 2020-12 (python -m jsonschema albo check-jsonschema). Sprawdź reguły degradacji: brak calibrated → ukryte procenty; brak fusion[m] → brak fuzji; brak energyThreshold → skaner niedostępny; zły sha256 lub minAtlasVersion → skaner niedostępny; gatunki z uncoveredSpecies: Atlas je pokazuje, skaner ich nie zwraca.
- Changelog względem poprzedniej wersji: klasy, uncoveredSpecies, metryki, progi, preprocessing. Zmiana preprocessingu lub kształtu wejścia/wyjścia = major.
- Upload do osobnego prywatnego repo HF z artefaktami modelu (propozycja nazwy: HF_MODEL_REPO; nie do folderu w HF_DATASET_REPO). Wydanie = tag + commit, wpisane w provenance.modelRevision, kopia SHA-256 na OneDrive, tylko po zgodzie.
Nie oznaczaj channel: "production" bez shipGates.passed = true i podpisu Wojciecha.
Zakończ pytaniem: „Wydać tę wersję do repo modelu (tag + commit) i przekazać Architektowi?”
```
