# Skille, szablony i connectory dla bota „Trenowanie modelu Grzybobranie AI”

> Stan katalogu sprawdzony 2026-11-10, zaktualizowany tego samego dnia po decyzjach o lokalnym GPU i nazwie bota. Skille Hugging Face pochodzą z pluginu **Hugging Face (id 735)**, który jest już zainstalowany na koncie Wojciecha (1 connector, 18 skilli).

## 1. Skille z katalogu Grok Bot

### Podstawowe (włącz od razu)
| Skill | Do czego w tym projekcie |
|---|---|
| `huggingface-vision-trainer` | Główny skill treningu: fine-tuning klasyfikatorów timm (MobileNetV3/V4, EfficientNet-Lite) na **lokalnym GPU w WSL2**. Przepisy z HF Jobs traktuj jako odniesienie (sprzęt, Trackio, zapis na Hub), ale nie uruchamiaj jobów jako ścieżki podstawowej (P3, soul.md §4). |
| `huggingface-trackio` | Logowanie metryk runów **lokalnie** (loss, top-1 per klasa, recall klas wysokiego ryzyka) i alerty diagnostyczne, żeby każda liczba w raporcie miała źródło. Publikacja Space / HF tylko po zgodzie. |
| `hf-cli` | Pobieranie z prywatnego `HF_DATASET_REPO` (w tym `data/ood-v1/`) na ext4 WSL (nie `/mnt/c`), sprawdzanie uprawnień (`hf auth whoami`), upload artefaktów do **osobnego prywatnego repo modelu** (propozycja: `HF_MODEL_REPO`) z tagiem i commitem, tylko po zgodzie. `hf jobs` tylko jako zapas i tylko po osobnej zgodzie na zadanie i koszt. |
| `huggingface-datasets` | Dataset Viewer API: metadane splitów, filtrowanie wierszy, statystyki i URL-e parquet przy audycie danych (P1) bez pobierania całego zbioru. |
| `huggingface-papers` | Czytanie paperów cytowanych w promptach (MobileNetV4, EfficientNet, temperature scaling, energy OOD) i linków do kodu i wag. |
| `huggingface-best` | Szybkie zestawienie modeli i benchmarków przy wyborze kandydatów (P2), wyłącznie spośród wag Apache-2.0/MIT/BSD. Rankingi nie mierzą czasu na telefonie, więc to tylko punkt startu. |

### Pomocnicze (gdy będą potrzebne)
| Skill | Do czego |
|---|---|
| `huggingface-tool-builder` | Skrypty łączące wywołania HF API, np. cykliczny raport stanu zbioru albo porównanie wersji paczek CSV. |
| `huggingface-local-models` | **Pasuje słabo.** Skill dotyczy głównie llama.cpp/GGUF. Trening tu to PyTorch + timm na RTX 5070 w WSL2; środowiska nie buduj według tego skilla. |
| `huggingface-gradio` / `huggingface-spaces` | Opcjonalny prywatny Space do ręcznego przeglądu pomyłek modelu (zdjęcie → top-3 → energia). Domyślnie Space musi być prywatny. |
| `huggingface-community-evals` | **Pasuje słabo.** Dotyczy ewaluacji LLM (inspect-ai, lighteval). Ewaluację klasyfikatora i `ship_gates` lepiej robić własnymi skryptami z `ml/`. |
| `office-xlsx` / `xlsx` | Tabele audytu danych i metryk per klasa w arkuszu do przeglądu Wojciecha. |
| `code-changes` + `source-control` | Zmiany w repo (`ml/`, fixture'y parytetu, `docs/model-card.md`) przez PR na GitHub. |
| `routines` | Przypomnienie o kamieniach milowych M0–M5 albo o wznowieniu przerwanego treningu na laptopie (checkpoint/resume). Nie jako watchdog jobów HF — te nie są ścieżką podstawową. |
| `skill-authoring` | Zapisanie powtarzalnych procedur (np. „eksport + bramki + manifest” albo „setup WSL2 + sm_120”) jako własnych skilli bota. |
| `huggingface-llm-trainer` | **Nie potrzebny.** To trening LLM (TRL, Unsloth). Wymieniam tylko dlatego, że pojawia się w wyszukiwaniu. |

### Brakujące w katalogu
Nie znalazłem skilli do eksportu LiteRT/TFLite, kwantyzacji, kalibracji ani OOD. Te etapy pokrywają prompty P4–P6 i kod z prototypu `ml/` (`export_tflite.py`, `ship_gates.py`, `score_delegate_dump.py`), a do konwersji z PyTorch służy biblioteka `litert-torch` (https://github.com/google-ai-edge/litert-torch). Po pierwszym udanym eksporcie warto zapisać go jako własny skill (`skill-authoring`). Nie ma też gotowego skilla „PyTorch + CUDA na WSL2 / Blackwell”; instrukcja jest w `soul.md` §4.

## 2. Szablony botów (SearchBotTemplates)

**Nie udało się sprawdzić.** Narzędzie `SearchBotTemplates` nie było dostępne w sesji, w której przygotowano ten pakiet. W katalogu narzędzi wykonawcy nie było narzędzia do szablonów. Warto sprawdzić to przy tworzeniu bota w aplikacji albo poprosić głównego asystenta.

W katalogu pluginów (`SearchPlugins`, zapytanie „bot template machine learning trainer”) nie ma szablonu bota do trenowania modeli. Najbliższe wyniki to pluginy narzędziowe: Hugging Face (zainstalowany), DataRobot, AWS SageMaker i Roboflow. Żaden nie jest potrzebny, bo HF pokrywa odczyt danych i wag, a trening idzie lokalnie.

## 3. Sugerowane connectory

| Connector | Stan na koncie Wojciecha (2026-11-10) | Po co botowi |
|---|---|---|
| **Hugging Face** (`Huggingface-skills`, plugin 735) | połączony | Dostęp do prywatnego `HF_DATASET_REPO` (dane, `data/ood-v1/`) oraz osobnego prywatnego repo modelu (propozycja: `HF_MODEL_REPO`; wydania z tagami i commitami), wagi bazowe timm, Trackio. HF Jobs nie jest domyślne. Do zapisu na Hub potrzebny jest token z odpowiednimi uprawnieniami, który ustawia Wojciech. Bot nie zakłada ani nie odczytuje tokenów. |
| **GitHub** (`cursor-github`, wbudowany) | połączony | Odczyt `<your-org>/<repo>` (`ml/`), PR-y z fixture'ami parytetu i kartą modelu. Dostęp do zapisu dawaj tylko na repo projektu. |
| **OneDrive** | połączony | Kopie SHA-256 zbiorów i artefaktów (zgodnie z ustaleniem dla `data/ood-v1/`). Domyślnie tylko odczyt i dopisywanie, bez usuwania. |
| Gmail, Playwright, Chrome DevTools | połączone | Niepotrzebne do treningu, więc nie dawałbym ich temu botowi. |

Uwaga praktyczna: box jest wspólny dla botów Wojciecha, więc nowy bot zobaczy pliki w tym repozytorium i w powiązanym repo aplikacji (prototyp `<app-repo>/ml/`, `krok3-decyzje.md`, `krok4-plan.md`, `data/ood-v1/` i ten pakiet). Własnych notatek innych botów nie widzi. Trening na laptopie Wojciecha (WSL2) to osobny filesystem — zbiór trzeba tam świadomie skopiować lub pobrać przez `hf`, na ext4, nie przez `/mnt/c`.
