# Raport Audytu Danych i Podziału Train/Val/Test (Etap P1) 🍄📊

**Status:** Zrealizowany i zweryfikowany  
**Data wykonania:** 2026-11-20  
**Sól HMAC:** `split_salt` (fingerprint: `88dcce3d`)  
**Podział danych:** 70% Train / 15% Val / 15% Test (ze stratyfikacją R-01)  

---

## 1. Podsumowanie wykonawcze

- **Łączna liczba weryfikowanych rekordów:** **87,934**
- **Liczba gatunków w konfiguracji mini+:** **67**
- **Liczba klas wchodzących do głowicy:** **67** (wszystkie spełniają progi $\ge 55$ / $\ge 40$)
- **Gatunki zakwalifikowane do `uncoveredSpecies`:** **0** (brak gatunków zdegradowanych)
- **Zdjęcia dopuszczone do UI aplikacji (`ui_safe`):** **30,058** (34.18%)
- **Rozłączność ze zbiorem OOD v1:** **100% czysta** (0 wycieków / 0 nałożeń)
- **Deduplikacja:** **0 duplikatów URL-i** / **0 duplikatów ID zdjęć**

---

## 2. Globalny rozkład licencji w korpusie

| Licencja | Użycie | Liczba zdjęć | Udział procentowy | Status prawny |
|---|---|---|---|---|
| `cc-by-nc-4.0` | Tylko Trening | 40,887 | 46.50% | Dozwolone (Free App) |
| `cc-by-4.0` | Trening + UI | 21,955 | 24.97% | Dozwolone (Pełne) |
| `cc-by-sa-3.0` | Tylko Trening | 11,795 | 13.41% | Dozwolone (Pełne) |
| `cc0-1.0` | Trening + UI | 6,795 | 7.73% | Dozwolone (Pełne) |
| `cc-by-sa-4.0` | Tylko Trening | 3,492 | 3.97% | Dozwolone (Pełne) |
| `cc-by-sa-2.0` | Tylko Trening | 1,393 | 1.58% | Dozwolone (Pełne) |
| `cc-by-2.0` | Trening + UI | 803 | 0.91% | Dozwolone (Pełne) |
| `cc-by-3.0` | Trening + UI | 496 | 0.56% | Dozwolone (Pełne) |
| `pd_self` | Tylko Trening | 228 | 0.26% | Dozwolone (Pełne) |
| `cc-by-sa-2.5` | Tylko Trening | 79 | 0.09% | Dozwolone (Pełne) |
| `cc-by-2.5` | Trening + UI | 9 | 0.01% | Dozwolone (Pełne) |
| `cc-by-sa-1.0` | Tylko Trening | 2 | 0.00% | Dozwolone (Pełne) |

---

## 3. Sumy kontrolne wygenerowanych zbiorów danych (`data/splits/`)

| Podzbiór | Plik | Liczba zdjęć | Udział | Suma SHA-256 |
|---|---|---|---|---|
| **Train** | `train.csv` | 61,418 | 69.85% | `09a09bace369c3dc4b63d4d31bb095fa6db23c2ea3a67fe8106b986c16a64afb` |
| **Val** | `val.csv` | 13,234 | 15.05% | `9fc70ad57e45e642c6fb4c073cf9d3719aa86b79b69567a1233811f17b83e475` |
| **Test** | `test.csv` | 13,282 | 15.10% | `59fbe0ece39997987bbb60dca778f97e4b57e30a661c1b33ff87bf33cf341422` |

---

## 4. Statystyki obserwacji wielozdjęciowych pod kątem fuzji (P5)

Metryki obserwacji zawierających $\ge 2$ oraz $\ge 3$ zdjęcia w zbiorach ewaluacyjnych:

| Zbiór | Unikalne obserwacje | Obserwacje $\ge 2$ zdjęć | Obserwacje $\ge 3$ zdjęć |
|---|---|---|---|
| **Walidacja (`val`)** | 8,614 | 1,781 | 862 |
| **Test (`test`)** | 9,048 | 1,888 | 946 |

---

## 5. Szczegółowy audyt 67 gatunków (Pokrycie, Grupy, Podział)

| Gatunek (slug) | Nazwa polska | Źródło | Zdjęć | Obs. | Aut. | Max/Obs | Train | Val | Test | `high_stakes` | Status głowicy |
|---|---|---|---|---|---|---|---|---|---|:---:|:---:|
| `amanita_verna` | muchomor wiosenny | new16+addendum | 67 | 31 | 27 | 3 | 45 | 11 | 11 | ⚠️ Tak | ✅ W głowicy |
| `inocybe_erubescens` | strzępiak ceglasty | v3 | 180 | 78 | 63 | 3 | 123 | 30 | 27 | ⚠️ Tak | ✅ W głowicy |
| `lepiota_brunneoincarnata` | czubajeczka brązowoczerwonawa (pl.wikipedia) | new16+addendum | 226 | 71 | 68 | 3 | 158 | 34 | 34 | ⚠️ Tak | ✅ W głowicy |
| `entoloma_sinuatum` | wieruszka zatokowata | v3 | 391 | 184 | 134 | 3 | 271 | 61 | 59 | ⚠️ Tak | ✅ W głowicy |
| `armillaria_mellea` | opieńka miodowa | v3 | 564 | 447 | 345 | 3 | 394 | 85 | 85 | Nie | ✅ W głowicy |
| `russula_emetica` | gołąbek wymiotny | v3 | 699 | 328 | 278 | 3 | 489 | 105 | 105 | ⚠️ Tak | ✅ W głowicy |
| `agaricus_moelleri` | pieczarka szarołuskowa (pl.wikipedia) | new16 | 782 | 299 | 260 | 3 | 548 | 117 | 117 | ⚠️ Tak | ✅ W głowicy |
| `amanita_virosa` | muchomor jadowity | v3 | 784 | 671 | 490 | 3 | 539 | 118 | 127 | ⚠️ Tak | ✅ W głowicy |
| `morchella_esculenta` | smardz jadalny | v3 | 789 | 489 | 393 | 3 | 553 | 118 | 118 | Nie | ✅ W głowicy |
| `clitocybe_phyllophila` | lejkówka liściowa | new16 | 812 | 151 | 226 | 3 | 568 | 122 | 122 | ⚠️ Tak | ✅ W głowicy |
| `cortinarius_orellanus` | zasłonak rudy | v3 | 856 | 394 | 271 | 3 | 597 | 130 | 129 | ⚠️ Tak | ✅ W głowicy |
| `rubroboletus_satanas` | borowik szatański | v3 | 867 | 542 | 408 | 3 | 605 | 131 | 131 | ⚠️ Tak | ✅ W głowicy |
| `russula_virescens` | gołąbek zielonawy | v3 | 988 | 792 | 549 | 3 | 692 | 148 | 148 | Nie | ✅ W głowicy |
| `amanita_regalis` | muchomor królewski | new16 | 1000 | 588 | 407 | 3 | 698 | 151 | 151 | ⚠️ Tak | ✅ W głowicy |
| `armillaria_ostoyae` | opieńka ciemna | new16 | 1000 | 551 | 398 | 3 | 700 | 150 | 150 | Nie | ✅ W głowicy |
| `caloboletus_calopus` | borowik żółtopory | new16 | 1000 | 541 | 408 | 3 | 699 | 151 | 150 | Nie | ✅ W głowicy |
| `leucocoprinus_leucothites` | czubnik pospolity (pl.wikipedia) | new16 | 1000 | 455 | 357 | 3 | 698 | 152 | 150 | Nie | ✅ W głowicy |
| `marasmius_oreades` | twardzioszek przydrożny | new16 | 1000 | 511 | 405 | 3 | 694 | 155 | 151 | Nie | ✅ W głowicy |
| `russula_vesca` | gołąbek wyborny (w indeksie: cukrówka) | new16 | 1000 | 529 | 317 | 3 | 700 | 150 | 150 | Nie | ✅ W głowicy |
| `tricholoma_portentosum` | gąska niekształtna | new16 | 1000 | 288 | 316 | 3 | 693 | 150 | 157 | Nie | ✅ W głowicy |
| `omphalotus_olearius` | kielichowiec pomarańczowy (pl.wikipedia) | new16 | 1001 | 543 | 415 | 3 | 700 | 150 | 151 | ⚠️ Tak | ✅ W głowicy |
| `paragyromitra_infula` | piestrzenica infułowata (pl.wikipedia) | new16 | 1001 | 579 | 374 | 3 | 699 | 152 | 150 | ⚠️ Tak | ✅ W głowicy |
| `tricholoma_sulphureum` | gąska siarkowa | new16 | 1001 | 509 | 343 | 3 | 701 | 150 | 150 | ⚠️ Tak | ✅ W głowicy |
| `agaricus_sylvaticus` | pieczarka lśniąca | new16 | 1002 | 236 | 331 | 3 | 702 | 150 | 150 | Nie | ✅ W głowicy |
| `amanita_porphyria` | muchomor porfirowy | new16 | 1002 | 462 | 304 | 3 | 699 | 151 | 152 | Nie | ✅ W głowicy |
| `tricholoma_equestre` | gąska zielonka | v3 | 1105 | 779 | 550 | 3 | 765 | 174 | 166 | ⚠️ Tak | ✅ W głowicy |
| `cantharellus_cibarius` | pieprznik jadalny | v3 | 1160 | 1093 | 711 | 3 | 812 | 174 | 174 | Nie | ✅ W głowicy |
| `leccinum_aurantiacum` | koźlarz czerwony | v3 | 1203 | 891 | 675 | 3 | 843 | 180 | 180 | Nie | ✅ W głowicy |
| `inocybe_geophylla` | strzępiak ziemistoblaszkowy (pl.wikipedia) | v3 | 1212 | 325 | 337 | 3 | 848 | 182 | 182 | ⚠️ Tak | ✅ W głowicy |
| `suillus_variegatus` | maślak pstry | v3 | 1224 | 884 | 596 | 3 | 841 | 184 | 199 | Nie | ✅ W głowicy |
| `gyroporus_castaneus` | piaskowiec kasztanowaty | v3 | 1240 | 876 | 588 | 3 | 868 | 186 | 186 | Nie | ✅ W głowicy |
| `suillus_bovinus` | maślak sitarz | v3 | 1295 | 960 | 608 | 3 | 903 | 194 | 198 | Nie | ✅ W głowicy |
| `xerocomus_subtomentosus` | podgrzybek zajączek | v3 | 1296 | 875 | 604 | 3 | 908 | 194 | 194 | Nie | ✅ W głowicy |
| `amanita_excelsa` | muchomor twardawy | v3 | 1304 | 1012 | 622 | 3 | 912 | 196 | 196 | Nie | ✅ W głowicy |
| `lactarius_torminosus` | mleczaj wełnianka | v3 | 1307 | 949 | 633 | 3 | 912 | 196 | 199 | ⚠️ Tak | ✅ W głowicy |
| `calocybe_gambosa` | gęśnica wiosenna | v3 | 1317 | 939 | 618 | 3 | 920 | 198 | 199 | Nie | ✅ W głowicy |
| `lactarius_deliciosus` | mleczaj rydz | v3 | 1329 | 1162 | 776 | 3 | 931 | 199 | 199 | Nie | ✅ W głowicy |
| `neoboletus_luridiformis` | borowik ceglastopory | v3 | 1340 | 594 | 606 | 3 | 937 | 201 | 202 | Nie | ✅ W głowicy |
| `sparassis_crispa` | siedzuń sosnowy (szmaciak gałęzisty) | v3 | 1356 | 1107 | 792 | 3 | 950 | 203 | 203 | Nie | ✅ W głowicy |
| `kuehneromyces_mutabilis` | łuszczak zmienny | v3 | 1400 | 1232 | 804 | 3 | 980 | 210 | 210 | Nie | ✅ W głowicy |
| `amanita_citrina` | muchomor cytrynowy | v3 | 1428 | 1218 | 842 | 3 | 1000 | 214 | 214 | Nie | ✅ W głowicy |
| `hydnum_repandum` | kolczak obłączasty | v3 | 1453 | 974 | 710 | 3 | 1017 | 218 | 218 | Nie | ✅ W głowicy |
| `suillellus_luridus` | borowik ponury | v3 | 1472 | 1095 | 756 | 3 | 1030 | 221 | 221 | ⚠️ Tak | ✅ W głowicy |
| `agaricus_campestris` | pieczarka polna (łąkowa) | v3 | 1481 | 982 | 801 | 3 | 1037 | 222 | 222 | Nie | ✅ W głowicy |
| `gyromitra_esculenta` | piestrzenica kasztanowata | v3 | 1488 | 1176 | 830 | 3 | 1041 | 224 | 223 | ⚠️ Tak | ✅ W głowicy |
| `amanita_pantherina` | muchomor plamisty | v3 | 1497 | 1237 | 821 | 3 | 1042 | 228 | 227 | ⚠️ Tak | ✅ W głowicy |
| `suillus_luteus` | maślak zwyczajny | v3 | 1544 | 1319 | 809 | 3 | 1075 | 232 | 237 | Nie | ✅ W głowicy |
| `xerocomellus_chrysenteron` | podgrzybek złotawy | v3 | 1555 | 1028 | 739 | 3 | 1089 | 233 | 233 | Nie | ✅ W głowicy |
| `chlorophyllum_rhacodes` | czubajka czerwieniejąca | v3 | 1557 | 988 | 797 | 3 | 1089 | 234 | 234 | Nie | ✅ W głowicy |
| `suillus_grevillei` | maślak żółty | v3 | 1559 | 1300 | 830 | 3 | 1091 | 234 | 234 | Nie | ✅ W głowicy |
| `cortinarius_caperatus` | płachetka zwyczajna | v3 | 1560 | 1016 | 668 | 3 | 1077 | 234 | 249 | Nie | ✅ W głowicy |
| `hypholoma_capnoides` | maślanka łagodna (pl.wikipedia) | v3 | 1561 | 1038 | 701 | 3 | 1093 | 234 | 234 | Nie | ✅ W głowicy |
| `agaricus_xanthodermus` | pieczarka karbolowa | v3 | 1572 | 1256 | 738 | 3 | 1092 | 244 | 236 | ⚠️ Tak | ✅ W głowicy |
| `leccinum_scabrum` | koźlarz babka | v3 | 1595 | 1288 | 842 | 3 | 1098 | 239 | 258 | Nie | ✅ W głowicy |
| `imleria_badia` | podgrzybek brunatny | v3 | 1609 | 1278 | 824 | 3 | 1127 | 241 | 241 | Nie | ✅ W głowicy |
| `paxillus_involutus` | krowiak podwinięty | v3 | 1679 | 1326 | 815 | 3 | 1171 | 253 | 255 | ⚠️ Tak | ✅ W głowicy |
| `tylopilus_felleus` | goryczak żółciowy | v3 | 1749 | 1250 | 808 | 3 | 1224 | 263 | 262 | Nie | ✅ W głowicy |
| `hygrophoropsis_aurantiaca` | lisówka pomarańczowa | v3 | 1764 | 1403 | 929 | 3 | 1234 | 265 | 265 | Nie | ✅ W głowicy |
| `amanita_rubescens` | muchomor czerwieniejący | v3 | 1778 | 1285 | 968 | 3 | 1244 | 267 | 267 | Nie | ✅ W głowicy |
| `scleroderma_citrinum` | tęgoskór pospolity (cytrynowy) | v3 | 1880 | 1400 | 1000 | 3 | 1316 | 282 | 282 | ⚠️ Tak | ✅ W głowicy |
| `macrolepiota_procera` | czubajka kania | v3 | 2000 | 1264 | 1139 | 3 | 1400 | 300 | 300 | Nie | ✅ W głowicy |
| `amanita_phalloides` | muchomor zielonawy (sromotnikowy) | v3 | 2058 | 1471 | 955 | 3 | 1440 | 309 | 309 | ⚠️ Tak | ✅ W głowicy |
| `galerina_marginata` | hełmówka jadowita (obrzeżona) | v3 | 2232 | 1611 | 897 | 3 | 1562 | 335 | 335 | ⚠️ Tak | ✅ W głowicy |
| `boletus_edulis` | borowik szlachetny | v3 | 2275 | 1428 | 1118 | 3 | 1593 | 341 | 341 | Nie | ✅ W głowicy |
| `lycoperdon_perlatum` | purchawka chropowata | v3 | 2350 | 1627 | 1267 | 3 | 1646 | 352 | 352 | Nie | ✅ W głowicy |
| `hypholoma_fasciculare` | maślanka wiązkowa | v3 | 2445 | 1554 | 1150 | 3 | 1711 | 367 | 367 | ⚠️ Tak | ✅ W głowicy |
| `amanita_muscaria` | muchomor czerwony | v3 | 3693 | 1428 | 1884 | 3 | 2584 | 555 | 554 | ⚠️ Tak | ✅ W głowicy |

---

## 6. Gatunki o małej liczności próby i realizacja decyzji R-01

Dla gatunków krytycznych (`high_stakes`), w szczególności ***Amanita verna*** (67 zdjęć), algorytm podziału deterministycznego zagwarantował twarde minimum w zbiorze testowym:
- ***Amanita verna* w teście:** **11 zdjęć** ($n \ge 8$ spełnione!).
- ***Amanita verna* w walidacji:** **11 zdjęć**.
- ***Amanita verna* w treningu:** **45 zdjęć**.

Wszystkie 67 gatunków posiada co najmniej 31 unikalnych obserwacji terenowych (minimum: *Amanita verna* — 31 obs., 27 autorów). Żaden gatunek nie został zdegradowany do `uncoveredSpecies`.

---

## 7. Podsumowanie kontroli szczelności i wycieków

- **Brak wycieku spójnych składowych:** Żadna obserwacja ani autor nie występuje jednocześnie w więcej niż jednym podzbiorze (`train`/`val`/`test`).
- **Rozłączność ze zbiorem OOD v1:** 100% rozłączności względem `data/ood-v1/ood_ids.txt`.
- **Czystość licencyjna:** Wykluczono całkowicie zbiory DF20/DF24, licencje ND oraz nieautentyczne grafiki.
