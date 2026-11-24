# Технічні матеріали для магістерської дисертації

Зібрано з репозиторію `asr_ssl` станом на 24.11.2026. Призначення — сировина для розділів 2 і 3 МД (див. `docs/diploma/plan.md`): що саме зроблено, якою моделлю, на яких даних і з якими параметрами, які результати й готові графіки.

Правила користування:

1. Кожне число має джерело (файл або скрипт). Перед вставкою в МД звіряти з `evaluation/results.csv`.
2. Моделі, навчені в `bf16-true`, з витоком dev у train або з FLEURS / Ukrainian Dialects у train, як основні результати не подавати (`docs/thesis/thesis_plan.md`, розд. 1 і 4; `todo.md`, п. 1). У таблицях вони позначені як застарілі.
3. Шляхи до графіків — відносно кореня репозиторію.

## Зміст

1. Загальна архітектура конвеєра
2. Нерозмічений набір VOA: метадані
3. Фільтрація нерозміченого аудіо за мовою
4. Сегментація, псевдомітки, LLM-суддя (VAD, розмітка вчителем, суддя, суддя на людських мітках, діалектна лексика)
5. Навчання вчителя і студентів (параметри, ланцюжки, усі запуски, вибір кількості епох, інфраструктура)
6. Оцінювання, результати, витоки даних (нормалізація й метрики, таблиця всіх прогонів, аналіз помилок, leak-clean, датасети, аудит коду)
7. Зведений каталог готових графіків (7.7 — галерея всіх графіків запусків)
8. Головні застереження для тексту МД

Скрипти разових обчислень, на які посилаються розділи 4 і 6, лежать у `docs/diploma/materials_scripts/` (запускати з кореня репозиторію через `env/bin/python`).

## 1. Загальна архітектура конвеєра

### 1.1 Етапи

Усі етапи запускаються з `main.py` (див. docstring), налаштування — `config.yaml` (для vast.ai `config.vast.yaml` розширює його через `extends`; для другої GPU — `config_parallel.yaml`, `--config 1`). Команда `python main.py pipeline` виконує етапи секції `pipeline` по черзі; кожен має режим `run`, `skip` або `skip_if_present` (`config.yaml:201`).

| № | Етап | Вхід → вихід | Код | Команда |
|---|---|---|---|---|
| 1 | Метадані й мова Whisper | VOA (HF `robinhad/VOA-ukr`) → `datasets/voa-ukr/voa_ukr_meta.json` | `scripts/monitor/meta_collector.py` | окремий скрипт |
| 2 | Фільтр мови (`pipeline.language`) | логіти MMS + мова Whisper → `ukrainian.csv` | `scripts/monitor/predict_languages.py`, `split_language_meta.py` | `main.py pipeline` |
| 3 | Сегментація VAD (`pipeline.timestamps`) | `ukrainian.csv` → `ukrainian_30s.csv` (сегменти ≤ 30 с) | `pseudo-labelling/vad-chunking/` | `main.py pipeline` |
| 4 | Псевдомітки (`label`) | сегменти → `labels.csv` | `pseudo-labelling/label.py` | `main.py label` |
| 5 | Оцінка міток LLM-суддею (`clean`) | `labels.csv` → `label_marks.csv` (0–5) | `pseudo-labelling/llm-as-a-judge/judge.py`, `decisions.py` | `main.py judge` |
| 6 | Вчитель | CV train → чекпоінт | `fine-tune/fine-tune.py` | `main.py fine-tune` |
| 7 | Студент / ланцюжок кроків | псевдомітки з `accepted_mark` (+ CV) → чекпоінт | `fine-tune/fine-tune-student.py`, `multi_step.py` | `main.py student`, `main.py multi-step` |
| 8 | Оцінювання | чекпоінт → `evaluation/<run>/`, рядок у `results.csv` | `evaluation/evaluate.py` | `main.py evaluate` |

Допоміжні: `main.py cache-voa` (одноразове декодування VOA-сегментів в int16 16 кГц, щоб студент не запускав ffmpeg), `main.py plot` (криві навчання), `main.py marks` (розподіл оцінок), MLflow для відстеження експериментів.

### 1.2 Секції конфігурації (`config.yaml`)

| Секція | Що задає |
|---|---|
| `paths` | корені даних, чекпоінтів, оцінок |
| `model` | базова модель (`openai/whisper-medium`), назва запуску, мова (`uk`) |
| `gpus`, `checkpoint_every`, `remove_mid_epoch_checkpoints` | ресурси й чекпоінти (кожні 25 % епохи) |
| `mlflow` | відстеження експериментів (`output/mlflow/mlflow.db`) |
| `batch_size`, `training` | гіперпараметри навчання вчителя (див. розд. 5) |
| `data` | train / dev / test набори |
| `pseudo_labelling` | вчитель для псевдоміток, файли метаданих і сегментів |
| `student` | навчання студента, `accepted_mark` |
| `multi_step` | ланцюжки кроків навчання |
| `voa_prep` | файли підготовки VOA і графіків |
| `pipeline` | режими етапів, `mms_model`, кількість процесів, параметри судді |

Рис. «Схема конвеєра» для МД треба намалювати за цією таблицею (готового немає).

## 2. Нерозмічений набір VOA: метадані

### 2.1 Призначення

Сирий нерозмічений корпус, з якого будується вибірка напівкерованого навчання. Джерело — Hugging Face `robinhad/VOA-ukr` (новинні програми Української служби «Голосу Америки»), спліт `train`. Список файлів і метадані зібрано скриптом `scripts/monitor/meta_collector.py` у `datasets/voa-ukr/voa_ukr_meta.json`.

### 2.2 Характеристики (обчислено з `datasets/voa-ukr/voa_ukr_meta.json`)

| Показник | Значення |
|---|---|
| Файлів | 20 507 |
| Загальна тривалість | 2 463,1 год |
| Розмічених (`is_labelled`) | 0 (поле транскрипції — «-») |
| Частота дискретизації | 44 100 Гц у всіх файлах |
| Канали | 2 (20 504 файли), 1 (3 файли) |
| Кодек | MP3 у всіх файлах |
| Тривалість файлу, хв | середня 7,21; медіана 3,89; 90-й перцентиль 19,89; 99-й — 29,99; максимум 479,95 |
| Частка файлів, довших за 30 с | 99,14 % |

Висновки для методу:

1. Усі файли треба передискретизувати 44,1 → 16 кГц (вимога Whisper) і звести до моно.
2. 99 % файлів довші за 30-секундне вікно Whisper, тому потрібна сегментація (VAD, див. розділ про сегментацію).

### 2.3 Графіки (`analysis/voa-meta-plots/`, скрипт `scripts/monitor/meta_stats.py`)

Увага: `language.png` з цієї ж серії побудовано на частковій версії метаданих (2 613 файлів), тож решта графіків `meta_stats.py` теж може бути застарілою. Перед вставкою в МД перебудувати: `python scripts/monitor/meta_stats.py` (пише в `voa_prep.meta_figures` = `output/eda`).

| Файл | Що показує |
|---|---|
| `duration_hist.png` | Гістограма тривалостей файлів |
| `duration_hist_p99.png` | Те саме, обрізано за 99-м перцентилем (без довгого хвоста) |
| `duration_cumulative.png` | Кумулятивна тривалість: скільки годин дають файли до певної довжини |
| `duration_by_language.png` | Box plot тривалості (хв) за мовою Whisper: українські файли найдовші (медіана ≈ 4,8 хв), англійські короткі (≈ 1,3 хв) — це переважно короткі синхрони/вставки |
| `sample_rate.png`, `num_channels.png`, `codec.png`, `split.png`, `is_labelled.png` | Стовпчикові діаграми однорідності формату (одне значення на весь корпус) |
| `whisper_length.png` | Розподіл довжини (символів) транскрипції Whisper за перші 120 с |
| `language.png` | **Застарілий:** кругова діаграма мов Whisper лише на 2 613 файлах (uk 2 307, en 254, ru 23…), тобто побудована на ранній частковій версії метаданих; для повного корпусу див. табл. 3.2 |

![analysis/voa-meta-plots/duration_hist.png](../../analysis/voa-meta-plots/duration_hist.png)

![analysis/voa-meta-plots/duration_hist_p99.png](../../analysis/voa-meta-plots/duration_hist_p99.png)

![analysis/voa-meta-plots/duration_cumulative.png](../../analysis/voa-meta-plots/duration_cumulative.png)

![analysis/voa-meta-plots/whisper_length.png](../../analysis/voa-meta-plots/whisper_length.png)

![analysis/voa-meta-plots/sample_rate.png](../../analysis/voa-meta-plots/sample_rate.png)

![analysis/voa-meta-plots/num_channels.png](../../analysis/voa-meta-plots/num_channels.png)

![analysis/voa-meta-plots/codec.png](../../analysis/voa-meta-plots/codec.png)


## 3. Фільтрація нерозміченого аудіо за мовою

### 3.1 Призначення

Залишити в корпусі лише українське мовлення. Під час ручного прослуховування виявлено: записи повністю іншими мовами (синхрони англійською, російською), закадровий переклад поверх мови оригіналу, музику без мовлення. Такі записи дали б вчителю завідомо хибні псевдомітки.

### 3.2 Принцип роботи

Метод — **узгодження двох незалежних класифікаторів мови**: файл вважається українським, лише якщо обидва визначили українську.

1. **Класифікатор 1 — Whisper.** `scripts/monitor/meta_collector.py`: faster-whisper `medium`, декодується перших 120 с аудіо (ffmpeg → 16 кГц моно). Мова визначається вбудованим механізмом Whisper за першими 4 вікнами по 30 с (`language_detection_segments = round(120 / 30)`), результат — код ISO 639-1 і ймовірність.
2. **Класифікатор 2 — MMS-LID.** `scripts/monitor/predict_languages.py`: `facebook/mms-lid-256` (спеціалізована модель ідентифікації мови з проєкту Massively Multilingual Speech, 256 мов). Вхід — перші 300 с аудіо. Вихід — логіти 256 мов на файл (`datasets/voa-ukr/mms-language-confidences.csv`); прогнозована мова — argmax, впевненість — softmax-ймовірність.
3. **Рішення.** `scripts/monitor/split_language_meta.py`: `language == "uk"` (Whisper) **і** `argmax == "ukr"` (MMS) → `datasets/voa-ukr/ukrainian.csv`; решта → `datasets/voa-ukr/other_languages.csv`.
4. **Ручна перевірка розбіжностей.** `scripts/local/listen_disagreements.py` по черзі відтворює файли, де мітки моделей розходяться, і показує обидві мітки.
5. **Порівняння міток.** `scripts/monitor/language_analysis.py`: коди Whisper (ISO 639-1) відображаються в ISO 639-3 і порівнюються з MMS; будує графіки.

У конвеєрі це крок `pipeline.language` (`pipeline.py:230`, `run_language`): спершу MMS, потім розділення. Метадані Whisper збираються заздалегідь.

### 3.3 Моделі й параметри

| Параметр | Whisper | MMS-LID | Джерело |
|---|---|---|---|
| Модель | faster-whisper `medium` (CTranslate2) | `facebook/mms-lid-256` | `meta_collector.py:38`, `config.yaml:211` |
| Скільки аудіо слухає | перші 120 с | перші 300 с | `meta_collector.py:39`, `predict_languages.py:36` |
| Частота | 16 кГц | 16 кГц | `meta_collector.py:194` |
| Точність | float16 на GPU (int8 на CPU) | fp16 autocast | `meta_collector.py:225`, `config.yaml:213` |
| Батч | 1 файл | до 16 файлів; обмеження за `найдовший × кількість` секунд | `config.yaml:212`, `predict_languages.py:158` |
| Інше | `beam_size=5`, `vad_filter=True`, `condition_on_previous_text=False` (транскрипт пишеться в метадані) | — | `meta_collector.py:236` |
| Правило рішення | `uk` | `ukr` (argmax логітів) | `split_language_meta.py:31` |

Розбіжність з текстом практики: у звіті з практики сказано, що Whisper слухає перші 5 хв; у коді — 120 с для Whisper, 300 с (5 хв) для MMS.

### 3.4 Результати (обчислено з `voa_ukr_meta.json` і `mms-language-confidences.csv`)

**Таблиця 3.1 — Узгодженість класифікаторів щодо української мови**

| Категорія | Файлів | % файлів | Годин | % годин |
|---|---:|---:|---:|---:|
| Обидві моделі — українська | 17 834 | 86,97 | 2 313,36 | 93,92 |
| Лише Whisper — українська | 32 | 0,16 | 3,27 | 0,13 |
| Лише MMS — українська | 57 | 0,28 | 5,08 | 0,21 |
| Жодна — не українська | 2 584 | 12,60 | 141,40 | 5,74 |
| **Разом** | **20 507** | 100 | **2 463,10** | 100 |

1. Залишено 17 834 файли, **2 313,36 год** (`ukrainian.csv`); відкинуто 2 673 файли, 149,74 год (`other_languages.csv`).
2. Неоднозначних файлів (лише одна модель) — 89, 8,35 год (0,34 % годин): правило «обидві» майже нічого не коштує в обсязі.
3. Повний збіг мітки мови (будь-якої, після відображення кодів) — 19 756 з 20 507 файлів (96,3 %), розбіжність — 751 (`language_agreement.png`).
4. Медіанна ймовірність Whisper: 0,991 для «обидві uk», 0,954 — «лише Whisper», 0,762 — «лише MMS». Тобто там, де MMS чує українську, а Whisper ні, Whisper помітно менш упевнений.

**Таблиця 3.2 — Найчастіші мови за кожною моделлю (файли)**

| Whisper | Файлів | MMS | Файлів |
|---|---:|---|---:|
| uk | 17 866 | ukr | 17 891 |
| en | 1 604 | eng | 1 025 |
| ru | 837 | rus | 799 |
| no | 39 | bci (бауле) | 161 |
| cy | 35 | war (варай) | 116 |
| it | 18 | nno | 88 |
| de | 17 | nob | 48 |
| pl | 16 | cym | 36 |

**Таблиця 3.3 — Матриця збігів (рядки — Whisper, стовпці — MMS, файли)**

| Whisper \ MMS | ukr | eng | rus | bci | war | nno | інші |
|---|---:|---:|---:|---:|---:|---:|---:|
| uk | 17 834 | 3 | 9 | 0 | 0 | 0 | 20 |
| en | 32 | 998 | 5 | 161 | 115 | 81 | 212 |
| ru | 20 | 5 | 784 | 0 | 0 | 5 | 23 |
| cy | 0 | 12 | 0 | 0 | 0 | 0 | 23 |
| no | 1 | 1 | 0 | 0 | 0 | 0 | 37 |
| it | 0 | 0 | 0 | 0 | 0 | 0 | 18 |
| інші | 4 | 6 | 1 | 0 | 1 | 2 | 94 |

Спостереження з матриці:

1. Найбільше розбіжностей у рядку `en`: 606 файлів, які Whisper назвав англійськими, MMS розніс по екзотичних мовах (бауле, варай, норвезька). Це узгоджується з ручним спостереженням: **на музиці без мовлення Whisper зазвичай повертає англійську, а MMS — випадкову мову**.
2. Валлійська (`cy`), норвезька (`no`), італійська у Whisper — також переважно музика й шум, а не справжні записи цими мовами.
3. Російська визначається узгоджено (784 файли).

### 3.5 Типові випадки розбіжностей (ручне прослуховування, звіт з практики, розд. 3.2)

1. Дві мови в записі: Whisper визначає мову початку, MMS — ту, що триваліша.
2. Акцент: MMS може обрати мову за акцентом (англійська зі скандинавським акцентом → скандинавська).
3. Суржик: MMS зазвичай визначає правильно (українська), Whisper — мову вкраплень.
4. Музика без мовлення: Whisper — англійська, MMS — випадково.

### 3.6 Графіки

| Файл | Що показує |
|---|---|
| `analysis/voa-meta-plots/language_agreement.png` | Стовпчики «agree / differ»: 19 756 проти 751 файлу (збіг мітки мови MMS і Whisper) |
| `analysis/voa-meta-plots/language_pie.png` | Кругова діаграма мов за MMS: ukr 17 891, eng 1 025, rus 799, bci 161, war 116, інші 515 |
| `analysis/voa-meta-plots/confidence.png` | Розподіл softmax-впевненості MMS за кошиками: переважна більшість файлів ≥ 0,99 |
| `analysis/voa-meta-plots/duration_by_language.png` | Тривалість файлів за мовою Whisper |
| `docs/thesis/practic.pdf`, рис. 3.1–3.2 | Ті самі дані у звіті з практики (збіжність і розподіл мов) |

![analysis/voa-meta-plots/language_agreement.png](../../analysis/voa-meta-plots/language_agreement.png)

![analysis/voa-meta-plots/language_pie.png](../../analysis/voa-meta-plots/language_pie.png)

![analysis/voa-meta-plots/confidence.png](../../analysis/voa-meta-plots/confidence.png)

![analysis/voa-meta-plots/duration_by_language.png](../../analysis/voa-meta-plots/duration_by_language.png)


Для МД бракує одного графіка: таблиця 3.1 у вигляді стовпчикової діаграми (4 категорії, файли й години). Його можна побудувати з тих самих двох файлів.

### 3.7 Те саме на розмічених датасетах

`analysis/language/detect_languages.py` проганяє ту саму логіку (Whisper 120 с з VAD-фільтром + MMS 300 с + правило «обидві») на кліпах `common_voice`, `fleurs`, `opentts`, `tg_voices`, `ukr_dialects`, `cv10_test` і будує `report.md` і `plots/` (`agreement.png`, `other_whisper.png`, `other_mms.png`, `confidence.png`, `duration.png`). **Локально результатів немає** (`predictions.csv` і `report.md` відсутні) — запускається на сервері з GPU, якщо потрібен у МД.

### 3.8 Відтворення

```bash
python scripts/monitor/meta_collector.py          # метадані + мова Whisper → voa_ukr_meta.json
python scripts/monitor/predict_languages.py --fp16 # логіти MMS → mms-language-confidences.csv
python scripts/monitor/split_language_meta.py      # ukrainian.csv / other_languages.csv
python scripts/monitor/language_analysis.py        # графіки language_*.png, confidence.png
python scripts/monitor/meta_stats.py               # графіки метаданих
python scripts/local/listen_disagreements.py       # прослуховування розбіжностей
python main.py pipeline                            # крок pipeline.language у складі конвеєра
```

## 4. Сегментація, псевдомітки, LLM-суддя


Усі шляхи відносні до кореня репозиторію `asr_ssl`. Числа, позначені «обчислено», отримано скриптами з `docs/diploma/materials_scripts/` (python з `env/bin/python`, модуль `csv`) з указаних файлів; решта взята безпосередньо з файлів.


### 4.1 VAD-сегментація аудіо VOA на шматки ≤ 30 с

#### Призначення

Whisper приймає не більше 30 с аудіо. Новинні кліпи VOA тривають у середньому 7,8 хв, тому кожен кліп довший за 30 с ріжеться на шматки ≤ 30 с, причому розріз ставиться на межі мовлення (за Silero VAD), щоб не розрізати слово.

#### Принцип роботи (кроками)

1. Вхідний список кліпів — `datasets/voa-ukr/ukrainian.csv` (результат етапу визначення мови; кліпи, визнані українськими).
2. Кліпи з `duration_seconds ≤ 30` копіюються без змін, з порожнім полем `timestamp` (`chunk.py`, `write_output`, `chunk_row(row, None)`).
3. Для довших кліпів `chunk_orchestrator.py` запускає окремий процес `chunk.py --shard <parquet>` на кожен з 195 parquet-шардів VOA, не більше `--max-processes` одночасно (у пайплайні `pipeline.chunk_processes: 6`).
4. `chunk.py` читає з шарду байти mp3, декодує їх через `ffmpeg` у моно float32 16 кГц (`-f f32le -ac 1 -ar 16000`).
5. Silero VAD (`get_speech_timestamps(..., sampling_rate=16000, return_seconds=True)`) повертає сегменти мовлення; усі їхні початки й кінці зводяться в один відсортований список «меж».
6. Жадібний алгоритм `chunk_spans`: курсор = 0; поки `курсор + 30 < тривалість`, у вікні `(курсор, курсор+30)` береться найпізніша межа мовлення; якщо меж немає — розріз рівно на `курсор + 30`. Наступне вікно починається з розрізу. Хвіст (≤ 30 с) — останній шматок.
7. Прогрес кожного шарду дописується у `datasets/voa-ukr/vad-progress/<shard>.jsonl` (JSON-рядок `{"path", "spans"}`), тож перерваний запуск продовжується.
8. Після завершення всіх шардів оркестратор зливає прогрес і пише `datasets/voa-ukr/ukrainian_30s.csv`: на кожен шматок рядок з `ID = <ID кліпу>_<NNN>`, `duration_seconds = end − start`, `timestamp = "start-end"` (3 знаки після коми), порожнім `whisper_transcription`.
9. При псевдорозмітці (`preprocess.py`) аудіо повторно декодується ffmpeg з частотою `pseudo_labelling.sample_rate`, вирізається за `timestamp` і додатково обрізається до `data.max_audio_seconds` (30 с) (`slice_audio`).
10. Мінімальної тривалості шматка й фільтрації коротких шматків у коді немає.

#### Модель і параметри

| Параметр | Значення | Джерело |
|---|---|---|
| Модель VAD | Silero VAD, `load_silero_vad()` | `pseudo-labelling/vad-chunking/chunk.py:257` |
| Пакет | `silero-vad` (версія не зафіксована) | `requirements.txt:22` |
| Поріг імовірності мовлення, мін. тривалість мовлення/тиші | не задано в коді — значення за замовчуванням бібліотеки; конкретні числа: не знайдено (пакет локально не встановлено) | `chunk.py:114-119` |
| Максимальна довжина шматка | 30,0 с | `chunk.py:43` (`MAX_SECONDS`) |
| Частота дискретизації для VAD | 16000 Гц | `chunk.py:44` |
| Ресемплінг | `ffmpeg -nostdin -v error -i pipe:0 -f f32le -ac 1 -ar 16000 pipe:1` (моно, float32) | `chunk.py:127-147` |
| Правило розрізу | найпізніша межа мовлення у вікні 30 с, інакше рівно 30 с | `chunk.py:88-106` |
| Мінімальна тривалість шматка | не застосовується | `chunk.py` (відсутня перевірка) |
| Потоків torch на процес | 1 | `chunk.py:256` |
| Паралельних процесів | 6 | `config.yaml:210` (`pipeline.chunk_processes`) |
| Формат ID шматка | `<ID>_{index:03d}` | `chunk.py:183` |
| Частота при розмітці | 16000 Гц | `config.yaml:103` |
| Обрізання при розмітці | 30,0 с | `config.yaml:87`, `preprocess.py:153-163` |
| Вихідний кеш для студента | `.npy`, моно int16, 16 кГц, обрізка 30 с, шлях `<voa_cache>/<ID[:2]>/<ID>.npy`, метадані `cache.json` | `fine-tune/voa_slices.py:1-8, 80-94`; `fine-tune/cache_voa.py:1-22` |

#### Дані (вхід, вихід, обсяги)

Обчислено з `datasets/voa-ukr/ukrainian.csv` та `datasets/voa-ukr/ukrainian_30s.csv`:

| Показник | Значення |
|---|---|
| Усього кліпів у метаданих VOA (`voa_ukr_meta.json`, `n_files`) | 20 507 (2 463,1 год) |
| Українських кліпів (вхід сегментації) | 17 834 |
| Сумарна тривалість українських кліпів | 2 313,36 год |
| Кліпів > 30 с (ріжуться VAD) | 17 724 (2 312,59 год) |
| Кліпів ≤ 30 с (без змін) | 110 (0,77 год) |
| Тривалість кліпу: середня / медіана / мін / макс | 467,0 с / 254,0 с / 4,04 с / 20 582 с |
| Формат джерела | mp3, 44 100 Гц, 17 832 стерео + 2 моно |
| Рядків (шматків) у `ukrainian_30s.csv` | 351 904 |
| Сумарна тривалість шматків | 2 313,38 год |
| Шматків з VAD-розрізом / цілих кліпів | 351 794 / 110 |
| Шматків на один розрізаний кліп: середнє / медіана / макс | 19,8 / 11 / 837 |
| Тривалість шматка: середня / медіана | 23,67 с / 26,3 с |
| Перцентилі тривалості шматка 1 % / 5 % / 95 % / 99 % | 1,21 / 7,90 / 30,0 / 30,0 с |
| Шматків коротших за 1 с / 2 с | 2 975 / 4 620 |
| Шматків рівно 30 с (вимушений розріз без межі мовлення) | 70 724 (20,1 % розрізаних шматків) |
| Індекс шардів `output/pseudo-labelling/voa_shard_index.json` | 195 шардів, 20 507 шляхів аудіо |

Файли прогресу `datasets/voa-ukr/vad-progress/` локально відсутні (не знайдено).

#### Результати

Таблиця вище. Головне: сегментація зберігає всі 2 313 год (втрат немає), але приблизно кожен п'ятий шматок розрізано «наосліп» рівно на 30 с, бо всередині вікна VAD не знайшов межі (безперервне мовлення без пауз або фонова музика).

#### Графіки

Графіки метаданих VOA (не мовні; побудовані `scripts/monitor/meta_stats.py` із часткового чекпойнта `voa_ukr_meta.json` — приблизно 2 600 кліпів і ~230 год, а не повних 20 507 кліпів):

1. `analysis/voa-meta-plots/duration_hist.png` — гістограма тривалості кліпів у хвилинах (60 бінів).
2. `analysis/voa-meta-plots/duration_hist_p99.png` — та сама гістограма до 99-го перцентиля (31,0 хв); основна маса кліпів 1–8 хв, пік близько 1 хв.
3. `analysis/voa-meta-plots/duration_cumulative.png` — накопичена тривалість (год) за кліпами, відсортованими за тривалістю; крива доходить до ~230 год на ~2 600 кліпах.
4. `analysis/voa-meta-plots/whisper_length.png` — гістограма довжини транскрипції Whisper на кліп (символи); більшість < 10 000, поодинокий викид ~104 000.
5. `analysis/voa-meta-plots/sample_rate.png` — усі кліпи 44 100 Гц.
6. `analysis/voa-meta-plots/num_channels.png` — усі кліпи двоканальні.
7. `analysis/voa-meta-plots/codec.png`, `split.png`, `is_labelled.png` — стовпчики кількості кліпів за кодеком, сплітом і ознакою розміченості (у повному `ukrainian.csv`: усі mp3, усі `train`, усі `is_labelled=False` — обчислено).


#### Скрипти для відтворення

1. `python pseudo-labelling/vad-chunking/chunk_orchestrator.py --max-processes 6`
2. Один шард вручну: `python pseudo-labelling/vad-chunking/chunk.py --shard <voa>/data/train-00000-of-00195.parquet`
3. У складі пайплайну: `python pipeline.py` (етап `pipeline.timestamps`).
4. Кеш декодованих шматків для студента: `python main.py cache-voa [--all] [--workers N]`.
5. Графіки метаданих: `python scripts/monitor/meta_stats.py`.


### 4.2 Генерація псевдоміток моделлю-вчителем

#### Призначення

Розпізнати всі 351 904 шматки VOA моделлю-вчителем (донавчений Whisper-medium) і записати текст як псевдомітку для навчання студента.

#### Принцип роботи (кроками)

1. Налаштування читаються з `config.yaml` (`pseudo-labelling/settings.py`, значень за замовчуванням у коді немає).
2. Вчитель завантажується з Lightning-чекпойнта `.ckpt`: створюється `openai/whisper-medium`, у нього підвантажуються ваги `state_dict` з префіксом `model.` (`label.py:87-110`).
3. `generation_config.language = "uk"`, `task = "transcribe"`, `forced_decoder_ids = None`; модель переводиться в bf16 на одному GPU (`label.py:184-191`).
4. Список шматків — `ukrainian_30s.csv`, відфільтрований за `ukrainian.csv` (`preprocess.load_segments`).
5. Для кожного шляху аудіо знаходиться шард (індекс кешується у `voa_shard_index.json`, перебудовується при зміні розміру/mtime шарду).
6. mp3 декодуються ffmpeg у пулі з `min(8, cpu_count)` потоків із випередженням 64 файли, поки GPU генерує (`label.py:36-39`).
7. Шматки збираються в батчі по `batch_size`; log-mel рахується на GPU; генерація жадібна (`num_beams=1`, `do_sample=False`).
8. Тексти декодуються з `skip_special_tokens=True` і дописуються в `labels.csv` (`ID,path,timestamp,text`). Порожній шматок (0 відліків) записується з порожнім текстом.

**Оновлення 24.11.2026.** `label.py` тепер зберігає ще й впевненість вчителя та обмежує зациклення. Наявні розмітки цих колонок не мають, для них треба перезапустити розмітку.

1. Нові колонки: `avg_logprob`, `confidence` = exp(avg_logprob), `min_logprob`, `num_tokens`, `compression_ratio`, `truncated`. Рахуються під час того самого жадібного декодування (logits processor `GreedyConfidence`), без другого проходу.
2. Ліміт токенів на шматок: min(440, ⌈секунди × 12⌉ + 16). Якщо шматок його вичерпав, примусово ставиться кінець тексту, а рядок позначається `truncated = 1`. На розмітці Т0 цей ліміт відрізав би всі 2 909 зациклень до 448 токенів.
3. Шматки сортуються за тривалістю у вікні з 4 батчів. Разом із лімітом це зменшує середню кількість кроків декодера на батч із 308 до ≈ 225 (оцінка за розміткою Т0).
9. Кожен запуск — окрема папка `output/pseudo-labelling/<slug вчителя>-NN`; `resume: true` продовжує незавершену (без файлу `complete`). Наприкінці пишеться `complete`.

#### Модель і параметри

| Параметр | Значення | Джерело |
|---|---|---|
| Базова архітектура | `openai/whisper-medium` | `config.yaml:26`; `run.json` усіх запусків |
| Вчитель, на мітках якого навчено оцінених студентів | `checkpoints/model-whisper-uk/model-whisper-uk-epoch00.ckpt` | `output/pseudo-labelling/model-whisper-uk-epoch00-01/run.json`; `evaluation/student_*/config.yaml` (`pseudo_labelling.teacher_model`) |
| Інший вчитель (повторна розмітка) | `checkpoints/model-whisper-m-fixed-leaks-5-epochs_01/model-whisper-m-fixed-leaks-5-epochs-epoch04.ckpt` | `output/pseudo-labelling/model-whisper-m-fixed-leaks-5-epochs-epoch04-0{1,2}/run.json` |
| Вчитель у поточному `config.yaml` | `model-whisper-m-fixed-leaks-10-epochs-epoch05.ckpt` (розмітки ним локально немає) | `config.yaml:99` |
| Мова / задача | `uk` / `transcribe` | `config.yaml:105-106`; `label.py:131-132, 184-186` |
| Пошук | жадібний: `num_beams=1`, `do_sample=False`, `use_cache=True` | `label.py:128-136` (так з першої версії, `git log -p`) |
| Розмір батчу у фактичних запусках | 64 | `run.json` усіх трьох запусків |
| Розмір батчу в поточному конфігу | 96 | `config.yaml:104` |
| Точність | bf16 (ваги й autocast) | `config.yaml:107`; `label.py:127, 190` |
| Частота дискретизації | 16000 Гц | `config.yaml:103` |
| Макс. довжина аудіо | 30,0 с | `config.yaml:87` |
| Макс. довжина виходу | не задано в `generate` (береться з `generation_config` моделі; число не знайдено в репозиторії); `data.max_label_tokens: 448` стосується навчання | `label.py:128-136`; `config.yaml:88` |
| Захист від галюцинацій (no_repeat_ngram, поріг compression ratio / logprob, temperature fallback) | відсутній; єдиний фільтр — подальший LLM-суддя | `label.py:113-137` |
| GPU | 1 (`nvidia1`) | `run.json` |
| Потоків декодування / випередження | `min(8, cpu)` / 64 файли | `label.py:37, 39` |

#### Дані (вхід, вихід, обсяги)

Обчислено з `output/pseudo-labelling/*/labels.csv` (тривалості — за `ID` з `ukrainian_30s.csv`):

| Запуск | Статус | Рядків | Унікальних ID | Порожніх текстів | Кліпів | Годин | Слів/шматок (сер. / мед. / макс.) | Шматків > 200 слів |
|---|---|---:|---:|---:|---:|---:|---|---:|
| `model-whisper-uk-epoch00-01` | завершено | 351 904 | 351 904 | 0 | 17 834 | 2 313,38 | 45,2 / 48 / 443 | 925 |
| `model-whisper-m-fixed-leaks-5-epochs-epoch04-01` | завершено (`complete` = `ok`) | 351 904 | 351 904 | 14 | 17 834 | 2 313,38 | 45,4 / 48 / 444 | 853 |
| `model-whisper-m-fixed-leaks-5-epochs-epoch04-02` | незавершено (немає `complete`) | 127 168 | 127 168 | 6 | 6 271 | 835,71 | 45,3 / 48 / 442 | 320 |

Шматки з > 200 словами на ≤ 30 с аудіо (норма ~48 слів) — імовірні повтори-галюцинації; максимальна довжина тексту 2 071 символ (whisper-uk) / 1 940 (5-epochs).

Орієнтовний час розмітки за mtime файлів (не логується): `model-whisper-uk-epoch00-01` — `run.json` 2026-10-12 14:19, `labels.csv` 2026-10-13 04:54, тобто близько 14,6 год на 2 313 год аудіо (≈ 160× реального часу). Для 5-epochs-01 оцінка неможлива через відновлення запуску.

#### Результати

Розмітку отримано для всіх 351 904 шматків (2 313,4 год). Якість псевдоміток окремо не вимірювалась (немає еталону) — її оцінює LLM-суддя (розділ 3).

#### Графіки

1. `output/pseudo-labelling/model-whisper-uk-epoch00-01/label_quality.png` — стовпчикова діаграма розподілу оцінок судді 0–5 для 351 872 псевдоміток вчителя whisper-uk: 0 — 4 771 (1,4 %), 1 — 4 996 (1,4 %), 2 — 36 238 (10,3 %), 3 — 88 910 (25,3 %), 4 — 111 917 (31,8 %), 5 — 105 040 (29,9 %). Будується `plot_marks.py`.

![output/pseudo-labelling/model-whisper-uk-epoch00-01/label_quality.png](../../output/pseudo-labelling/model-whisper-uk-epoch00-01/label_quality.png)


#### Скрипти для відтворення

1. `python pseudo-labelling/label.py` (параметри з `config.yaml`, секція `pseudo_labelling`).
2. Через пайплайн: `python pipeline.py` (етап `pipeline.label`).
3. Статистика вище: `env/bin/python docs/diploma/materials_scripts/lab.py` (обхід `labels.csv` модулем `csv`).


### 4.3 LLM-суддя якості псевдоміток

#### Призначення

Без еталонної транскрипції оцінити, чи є псевдомітка коректним українським текстом, і відібрати для студента шматки з оцінками зі списку `accepted_mark`. Суддя бачить лише текст, аудіо не слухає.

#### Принцип роботи (кроками)

1. `judge.py` читає `labels.csv` (за замовчуванням — найновіший для `pseudo_labelling.teacher_model`) і пропускає індекси, що вже є в `label_marks.csv` (продовження після зупинки; пошкоджені NUL-байтами рядки вважаються неоціненими).
2. Бекенд `llm` (за замовчуванням): батчі по 16 рядків `<index> <текст>` (текст зведено в один рядок) ідуть у OpenRouter Chat Completions; системне повідомлення = `prompt.txt` + `FORMAT_RULES`; `temperature = 0`.
3. Відповідь має бути рівно по рядку `<index> <оцінка>` на кожен індекс; інакше (зайвий текст, пропуск, повтор, оцінка > 5) батч відхиляється й повторюється.
4. Бекенд `decisions` (`decisions.py`): одна транскрипція на запит до OpenRouter Decisions API (альфа), одне питання типу `score` з шістьма рівнями з `prompt.txt`; `mark` = найімовірніший рівень, `score` = зважений за ймовірностями рівень, `confidence` = зосередженість імовірностей.
5. До `--workers` запитів одночасно; невдалий батч повторюється з паузою `min(8, 2^(спроба−1))` с, до `--retries` спроб, потім лишається на наступний запуск.
6. Прийняті оцінки дописуються в `label_marks.csv` (`index,ID,mark[,score,confidence]`).
7. Студент (`fine-tune/voa_slices.py:accepted_ids`) бере для кожного ID останню оцінку й лишає шматки з оцінкою з `student.accepted_mark` (у конфігу `[3, 4, 5]`; у multi_step кроки `[3,4,5]` і `[5]`); шматки з порожнім текстом відкидаються.
8. Режим `--test`: 100 випадкових рядків (seed 0), або 200, якщо якась оцінка не трапилась; результат у `judge_test-<модель>-<час>.csv` для ручної перевірки.

#### Промпт (дослівно, `pseudo-labelling/llm-as-a-judge/prompt.txt`)

```
Score each Ukrainian transcript for use as a speech-recognition label.
Work with Ukrainian only.

Mark from 0 to 5:
0 — nonsense, not a usable Ukrainian utterance
1 — mostly broken; only fragments can be understood
2 — understandable, with many lexical, grammatical, or morphological errors
3 — clear Ukrainian with several lexical, grammatical, or morphological errors
4 — good Ukrainian with only minor errors of those kinds
5 — a sound Ukrainian label with no lexical, grammatical, or morphological errors
```

Правила формату, що додаються до системного повідомлення в бекенді `llm` (`judge.py:65-71`, `FORMAT_RULES`, дослівно):

```
Reply with marks only.
The user message has one transcript per line: `<index> <transcript>`.
Return one line per transcript and no other text.
Each line must be `<index> <mark>`, with the same index and an integer mark from 0 to 5.
Use every given index once.
```

SHA-256 промпту: `46d5132211c4c1e950c194670612448ef39b7d7b14b415d3ca88495b9460ad97` (`analysis/label-judge/*/run.json`). Промпт не змінювався з першого коміту 2026-10-22 (`git log -p`).

#### Модель і параметри

| Параметр | Значення | Джерело |
|---|---|---|
| Чат-суддя в поточному конфігу | `google/gemma-4-26b-a4b-it` | `config.yaml:218` (`pipeline.judge_model`) |
| Decision-суддя (приклад/використаний в аналізі) | `typesafe/jev-1.14` | `pseudo-labelling/llm-as-a-judge/README.md:25,39`; `analysis/label-judge/typesafe_jev-1.14-02/run.json` |
| Модель, що оцінила псевдомітки `model-whisper-uk-epoch00-01` | не знайдено (у тодішніх конфігах `judge_model: ""`, модель задавалась через `.env`, якого в репозиторії немає; файл має заголовок `index,ID,mark` → бекенд `llm`) | `evaluation/student_*/config.yaml`; заголовок `label_marks.csv` |
| Модель, що оцінила `model-whisper-m-fixed-leaks-5-epochs-epoch04-01` | не знайдено; заголовок `index,ID,mark,score,confidence` і заповнені `score/confidence` → бекенд `decisions` | заголовок `label_marks.csv` |
| API (llm) | `https://openrouter.ai/api/v1/chat/completions` | `judge.py:64` |
| API (decisions) | `https://openrouter.ai/api/alpha/decisions` | `decisions.py:27` |
| Розмір батчу | 16 (llm), 1 (decisions) | `judge.py:59, 381` |
| Temperature | 0 | `judge.py:234` |
| Паралельних запитів | 4 | `config.yaml:214` (`pipeline.judge_workers`) |
| Спроб на батч | 3 | `config.yaml:215` (`pipeline.judge_retries`) |
| Пауза між спробами | `min(8, 2^(attempt−1))` с | `judge.py:280` |
| Таймаут запиту | 180 с (llm), 120 с (decisions) | `judge.py:250`; `decisions.py:125` |
| Seed тестової вибірки | 0; 100/200 рядків | `judge.py:60-62` |
| Прийнятні оцінки для студента | `[3, 4, 5]` | `config.yaml:126` |
| Вартість, токени | не знайдено (не логуються) | — |
| Час | не логується; за mtime: `label_marks.csv` whisper-uk — 2026-10-13 18:42 (після `labels.csv` 04:54), тобто ≤ ~14 год на 351 872 оцінки | mtime файлів |

#### Дані (вхід, вихід, обсяги)

1. Вхід: `labels.csv` (розділ 2).
2. Вихід: `output/pseudo-labelling/model-whisper-uk-epoch00-01/label_marks.csv` — 351 872 оцінки з 351 904 шматків (32 лишились неоціненими, обчислено); `output/pseudo-labelling/model-whisper-m-fixed-leaks-5-epochs-epoch04-01/label_marks.csv` — 236 457 оцінок (оцінювання не завершене, 67,2 % шматків). Дублікатів ID і пошкоджених рядків немає (обчислено).

#### Результати

Обчислено скриптом `marks.py` (поза репозиторієм): для кожного ID береться остання оцінка (як у `voa_slices.accepted_ids`); години = сума `duration_seconds` шматків з `ukrainian_30s.csv` за `ID`.

Розподіл оцінок, вчитель whisper-uk-epoch00 (на цих мітках навчено всіх оцінених студентів):

| Оцінка | Кількість | % | Годин |
|---:|---:|---:|---:|
| 0 | 4 771 | 1,36 | 15,0 |
| 1 | 4 996 | 1,42 | 24,7 |
| 2 | 36 238 | 10,30 | 241,8 |
| 3 | 88 910 | 25,27 | 622,4 |
| 4 | 111 917 | 31,81 | 776,3 |
| 5 | 105 040 | 29,85 | 633,1 |
| Разом | 351 872 | 100 | 2 313,2 |

Середня оцінка 3,74.

| Поріг | Шматків | % шматків | Годин | % годин |
|---|---:|---:|---:|---:|
| {2–5} | 342 105 | 97,22 | 2 273,5 | 98,3 |
| {3–5} | 305 867 | 86,93 | 2 031,7 | 87,8 |
| {4–5} | 216 957 | 61,66 | 1 409,3 | 60,9 |
| {5} | 105 040 | 29,85 | 633,1 | 27,4 |

Розподіл оцінок, вчитель whisper-m-fixed-leaks-5-epochs-epoch04 (запуск 01, оцінено 236 457 шматків, decision-бекенд):

| Оцінка | Кількість | % | Годин |
|---:|---:|---:|---:|
| 0 | 3 953 | 1,67 | 13,0 |
| 1 | 1 837 | 0,78 | 10,3 |
| 2 | 19 406 | 8,21 | 136,2 |
| 3 | 96 118 | 40,65 | 672,6 |
| 4 | 69 601 | 29,43 | 473,0 |
| 5 | 45 542 | 19,26 | 248,5 |
| Разом | 236 457 | 100 | 1 553,6 |

Середня оцінка 3,53. Пороги: {2–5} 230 667 (97,55 %, 1 530,3 год); {3–5} 211 261 (89,34 %, 1 394,1 год); {4–5} 115 143 (48,70 %, 721,5 год); {5} 45 542 (19,26 %, 248,5 год).

Порівняння двох розподілів некоректне як порівняння вчителів: різні судді (llm проти decisions) і різна повнота оцінювання.

#### Графіки

1. `output/pseudo-labelling/model-whisper-uk-epoch00-01/label_quality.png` — розподіл оцінок 0–5 (кількість і відсоток над кожним стовпчиком), 351 872 мітки; числа збігаються з таблицею вище.

#### Скрипти для відтворення

1. `python main.py judge --test` — пробна оцінка 100/200 рядків.
2. `python main.py judge` або `python pseudo-labelling/llm-as-a-judge/judge.py [--labels <labels.csv>] [--backend llm|decisions] [--workers 4] [--retries 3]`.
3. `python pseudo-labelling/llm-as-a-judge/judge.py --backend decisions --workers 16`.
4. Графік: `python main.py marks` або `python pseudo-labelling/llm-as-a-judge/plot_marks.py --marks <label_marks.csv>`.
5. У пайплайні — етап `pipeline.clean` (`python pipeline.py`); потрібні `OPENROUTER_API_KEY`, для `decisions` — `USE_DECISION_INSTEAD_OF_LLM=true` і `OPENROUTER_DECISION_MODEL`.


### 4.4 Суддя на людських мітках (калібрування судді)

#### Призначення

Перевірити, як той самий суддя з тим самим промптом оцінює ручні транскрипції розмічених датасетів: якщо він ставить високі оцінки еталонним міткам і помітно нижчі — діалектним, це характеризує його упередження та інтерпретацію порогів для псевдоміток.

#### Принцип роботи (кроками)

1. `analysis/label-judge/judge_labels.py` бере всі спліти (train/dev/test) усіх датасетів із `fine-tune/dataset_roles.py`, що є на диску; читає лише тексти.
2. Використовується той самий клас `Judge` з `judge.py` (той самий `prompt.txt`, батчі, повтори).
3. Кожен запуск — папка `analysis/label-judge/<модель>-NN/` з `sample.csv`, `marks.csv`, `run.json`, звітом і графіками.
4. Звіт рахує на dataset/split: середню оцінку з 95 % ДІ (`1,96·√(var/n)`, `judge_labels.py:255`), частки кожної оцінки, частку 4–5 і частку «поганих» міток 0–2.

#### Модель і параметри

| Параметр | Gemma | Jev |
|---|---|---|
| Модель | `google/gemma-4-26b-a4b-it` | `typesafe/jev-1.14` |
| Бекенд | `llm` (батч 16) | `decisions` (1 текст/запит) |
| Вибірка | усі транскрипції (`per_split: 0`), seed 0 | те саме |
| Початок / звіт | 2026-11-08 12:45:43 / 19:12:14 | 2026-11-08 19:14:03 / 19:26:57 |
| Оцінено | 161 798 з 161 814 | 161 814 з 161 814 |
| Джерело | `analysis/label-judge/google_gemma-4-26b-a4b-it-02/{run.json,report-google_gemma-4-26b-a4b-it.md}` | `analysis/label-judge/typesafe_jev-1.14-02/{run.json,report-typesafe_jev-1.14.md}` |

Папки `google_gemma-4-26b-a4b-it-01` (лише `sample.csv`, без оцінок) і `typesafe_jev-1.14-01` (96 оцінок) — перервані пробні запуски.

#### Дані

Розміри популяцій (`run.json`): common_voice train 62 460 / dev 10 955 / test 10 956; fleurs 2 810 / 325 / 750; opentts train 23 983; tg_voices train 12 069; ukr_dialects 27 520 / 3 347 / 3 435; cv10_test test 3 204. Разом 161 814.

#### Результати

Скопійовано зі звітів (середнє ± 95 % ДІ; частки у %):

| dataset/split | Gemma: середнє | Gemma: % 5 | Gemma: % 4–5 | Gemma: % 0–2 | Jev: середнє | Jev: % 5 | Jev: % 4–5 | Jev: % 0–2 | Jev: сер. score | Jev: сер. confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| common_voice/train | 4,69 ± 0,01 | 81,4 | 92,3 | 4,1 | 4,85 ± 0,01 | 94,7 | 95,8 | 2,2 | 4,42 | 0,66 |
| common_voice/dev | 4,72 ± 0,01 | 82,7 | 92,9 | 3,5 | 4,88 ± 0,01 | 95,4 | 96,4 | 1,7 | 4,45 | 0,68 |
| common_voice/test | 4,71 ± 0,01 | 82,3 | 93,0 | 3,7 | 4,87 ± 0,01 | 95,3 | 96,3 | 1,7 | 4,43 | 0,67 |
| fleurs/train | 4,74 ± 0,02 | 78,6 | 96,3 | 0,7 | 4,04 ± 0,03 | 28,4 | 75,6 | 0,5 | 3,81 | 0,56 |
| fleurs/dev | 4,79 ± 0,06 | 82,8 | 97,2 | 0,6 | 4,10 ± 0,08 | 32,6 | 77,5 | 0,0 | 3,84 | 0,57 |
| fleurs/test | 4,70 ± 0,04 | 75,6 | 95,7 | 0,9 | 4,02 ± 0,05 | 27,9 | 74,9 | 0,8 | 3,79 | 0,56 |
| opentts/train | 4,73 ± 0,01 | 82,4 | 93,7 | 2,7 | 4,90 ± 0,01 | 96,5 | 97,2 | 1,4 | 4,52 | 0,72 |
| tg_voices/train | 4,72 ± 0,01 | 81,8 | 93,3 | 2,9 | 4,90 ± 0,01 | 96,6 | 97,2 | 1,3 | 4,51 | 0,71 |
| ukr_dialects/train | 3,81 ± 0,01 | 38,3 | 60,6 | 17,1 | 3,81 ± 0,02 | 49,8 | 58,3 | 20,3 | 3,58 | 0,47 |
| ukr_dialects/dev | 3,83 ± 0,04 | 39,7 | 61,4 | 16,8 | 3,83 ± 0,05 | 50,2 | 59,3 | 19,9 | 3,58 | 0,47 |
| ukr_dialects/test | 3,76 ± 0,04 | 37,8 | 58,8 | 18,9 | 3,78 ± 0,05 | 48,6 | 57,4 | 20,9 | 3,56 | 0,47 |
| cv10_test/test | 4,54 ± 0,03 | 74,0 | 87,8 | 6,5 | 4,49 ± 0,04 | 74,9 | 82,0 | 5,0 | 4,02 | 0,51 |

Повні частки 0/1/2/3/4/5 — у звітах (наприклад, ukr_dialects/train: Gemma 0,1/1,1/15,9/22,3/22,4/38,3; Jev 2,3/2,0/16,0/21,4/8,5/49,8).

Узгодженість двох суддів (обчислено скриптом `agree.py` на 161 798 спільних індексах `marks.csv`; NUL-пошкоджені рядки пропущено):

| Метрика | Значення |
|---|---:|
| Точний збіг оцінки | 73,2 % |
| Розбіжність ≤ 1 бал | 90,1 % |
| Каппа Коена (невагова) | 0,297 |
| Квадратично зважена каппа | 0,468 |
| «Погані» (0–2): Gemma / Jev / обидва | 10 503 / 9 347 / 3 846 |

Висновки з даних:

1. Обидва судді ставлять еталонним міткам CV, OpenTTS, TG Voices середню оцінку 4,69–4,90, а ukr_dialects — 3,76–3,83; частка «поганих» міток у діалектах 17–21 % проти 1–4 % в інших. Тобто суддя карає діалектну лексику навіть у правильних ручних мітках.
2. FLEURS: Gemma 4,70–4,79, Jev лише 4,02–4,10 (оцінка 5 у 28–33 %) — найбільша розбіжність суддів; ймовірно, Jev штрафує нормалізований текст FLEURS без регістру й пунктуації (гіпотеза, не перевірялась).
3. Узгодженість суддів помірна (зважена каппа 0,47); «погані» мітки суддів перетинаються менш ніж наполовину.

#### Графіки

1. `analysis/label-judge/google_gemma-4-26b-a4b-it-02/plots/mark_distribution.png` — горизонтальні стовпчики 100 % на кожен dataset/split з n, кольорові сегменти оцінок 0–5 (Gemma); видно, що в ukr_dialects частка оцінки 5 лише 38–40 %, а в інших 74–83 %.
2. `analysis/label-judge/google_gemma-4-26b-a4b-it-02/plots/mean_mark.png` — середня оцінка з 95 % ДІ на dataset/split (Gemma).
3. `analysis/label-judge/google_gemma-4-26b-a4b-it-02/plots/poor_share.png` — частка міток з оцінками 0–2 на dataset/split (Gemma).
4. `analysis/label-judge/typesafe_jev-1.14-02/plots/mark_distribution.png` — те саме, що 1, для Jev.
5. `analysis/label-judge/typesafe_jev-1.14-02/plots/mean_mark.png` — середня оцінка Jev з 95 % ДІ, відсортована: tg_voices 4,90 … ukr_dialects/test 3,78; FLEURS 4,02–4,10.
6. `analysis/label-judge/typesafe_jev-1.14-02/plots/poor_share.png` — частка міток 0–2 у Jev: ukr_dialects 19,9–20,9 %, cv10_test 5,0 %, common_voice 1,7–2,2 %, opentts 1,4 %, tg_voices 1,3 %, fleurs 0,0–0,8 %.

![analysis/label-judge/google_gemma-4-26b-a4b-it-02/plots/mark_distribution.png](../../analysis/label-judge/google_gemma-4-26b-a4b-it-02/plots/mark_distribution.png)

![analysis/label-judge/google_gemma-4-26b-a4b-it-02/plots/mean_mark.png](../../analysis/label-judge/google_gemma-4-26b-a4b-it-02/plots/mean_mark.png)

![analysis/label-judge/google_gemma-4-26b-a4b-it-02/plots/poor_share.png](../../analysis/label-judge/google_gemma-4-26b-a4b-it-02/plots/poor_share.png)

![analysis/label-judge/typesafe_jev-1.14-02/plots/mark_distribution.png](../../analysis/label-judge/typesafe_jev-1.14-02/plots/mark_distribution.png)

![analysis/label-judge/typesafe_jev-1.14-02/plots/mean_mark.png](../../analysis/label-judge/typesafe_jev-1.14-02/plots/mean_mark.png)

![analysis/label-judge/typesafe_jev-1.14-02/plots/poor_share.png](../../analysis/label-judge/typesafe_jev-1.14-02/plots/poor_share.png)


#### Скрипти для відтворення

1. `python analysis/label-judge/judge_labels.py --model google/gemma-4-26b-a4b-it` (бекенд `llm`).
2. `python analysis/label-judge/judge_labels.py --backend decisions --model typesafe/jev-1.14 --workers 16`.
3. Пробний прогін: `python analysis/label-judge/judge_labels.py --backend decisions --per-split 200 --workers 16`.
4. Продовження / перебудова звіту: `--resume analysis/label-judge/<модель>-01`, `--report-only analysis/label-judge/<модель>-01`.


### 4.5 Унікальна лексика ukr_dialects (`analysis/unique-tokens/`)

#### Призначення

Знайти символи, слова (і опційно BPE-токени Whisper), які трапляються в транскрипціях ukr_dialects і не трапляються в жодному іншому розміченому датасеті; пояснити складність діалектного набору та зв'язок з оцінками судді.

#### Принцип роботи (кроками)

1. `scripts/monitor/unique_tokens.py` завантажує тексти цільового датасету (`--target ukr_dialects`, усі спліти) і всіх інших датасетів з `fine-tune/dataset_roles.py` (усі спліти).
2. Рівень символів: сирий текст посимвольно.
3. Рівень слів: нормалізація як при підрахунку WER (`lower`, `’`→`'`, видалення пунктуації `[^\w\s']+`, стиснення пробілів), розбиття за пробілами.
4. Рівень BPE-токенів Whisper — лише з `--whisper` (у репозиторії результату немає).
5. Для кожного рівня: елементи цілі, відсутні в еталоні, сортуються за частотою; у CSV пишуться `item,count,example` (приклад речення).

#### Модель і параметри

| Параметр | Значення | Джерело |
|---|---|---|
| Ціль | `ukr_dialects` | `scripts/monitor/unique_tokens.py:50` |
| Спліти | train, dev, test (ціль і еталон) | `unique_tokens.py:39, 51, 58` |
| Нормалізація | як `normalize_text` у `fine-tune/fine-tune.py` | `unique_tokens.py:36-45` |
| Вихід | `analysis/unique-tokens/{characters,words}.csv` | `unique_tokens.py:61, 115` |

#### Дані та результати

Обчислено з `analysis/unique-tokens/words.csv` і `characters.csv`:

| Показник | Значення |
|---|---:|
| Унікальних символів ukr_dialects | 0 (`characters.csv` містить лише заголовок) |
| Різних слів, яких немає в інших датасетах | 37 832 |
| Сумарна кількість їх входжень у ukr_dialects | 88 541 |
| З них слів, що трапились 1 раз | 25 999 |
| Слів з частотою ≥ 10 | 950 |
| Частка таких входжень від усіх слів ukr_dialects | не знайдено (скрипт виводить у консоль, у файл не пише) |

Топ-25 слів (кількість): йиго 1200, иванчік 642, йиму 639, него 531, из 525, тот 425, тоже 339, шош 313, шє 306, вна 271, будет 242, мині 231, нічо 226, иванчіка 217, пид 207, видразу 188, нев 187, мижи 186, шкиндя 171, вни 170, голов 165, людий 149, пудно 145, тимунь 143, уни 130. Примітки: «иванчік», «шкиндя» — імена персонажів одного тексту; «тот», «него», «будет», «тоже» — водночас русизми.

Зв'язок з оцінками судді на псевдомітках VOA (обчислено скриптом `dial.py`: 200 найчастіших слів із `words.csv`, нормалізація як вище; сегмент «діалектний», якщо містить хоч одне таке слово):

| Розмітка (суддя) | Сегменти | Кількість | Середня оцінка | % 4–5 | % 5 |
|---|---|---:|---:|---:|---:|
| whisper-uk-epoch00 (llm) | з діалектними словами | 1 214 | 3,40 | 46,2 | 16,4 |
| whisper-uk-epoch00 (llm) | без них | 350 658 | 3,74 | 61,7 | 29,9 |
| 5-epochs-epoch04-01 (decisions) | з діалектними словами | 1 593 | 3,42 | 41,3 | 10,5 |
| 5-epochs-epoch04-01 (decisions) | без них | 234 864 | 3,53 | 48,7 | 19,3 |

Цифри для whisper-uk збігаються з `docs/thesis/thesis_plan.md:117-122`. Частка «діалектних» сегментів мала (1 214 / 351 872 = 0,35 %).

#### Графіки

Не знайдено (скрипт графіків не будує).

#### Скрипти для відтворення

1. `python scripts/monitor/unique_tokens.py`
2. З BPE-токенами: `python scripts/monitor/unique_tokens.py --whisper`
3. Варіант test проти train: `python scripts/monitor/unique_tokens.py --target-splits test --reference-splits train --whisper`


### 4.6 Інше в `pseudo-labelling/` і `analysis/`

1. `analysis/voa-meta-plots/` — описано в розділі 1 (графіки метаданих). `confidence.png` і `duration_by_language.png` стосуються визначення мови й тут не розглядаються.
2. `fine-tune/cache_voa.py` + `fine-tune/voa_slices.py` — одноразове декодування прийнятих шматків у `.npy` int16 (16 кГц, ≤ 30 с), щоб студент не запускав ffmpeg на кожен mp3; `cache.json` фіксує `sample_rate`, `max_audio_seconds`, `dtype`, і кеш з іншими значеннями не використовується. Кількість кешованих шматків/годин локально: не знайдено (кеш лежить на сервері в `paths.voa_cache`).
3. Скрипти обчислень цього документа (`docs/diploma/materials_scripts/`, запускати з кореня репозиторію): `lab.py` (статистика `labels.csv`), `marks.py` (розподіл оцінок і години за порогами), `agree.py` (узгодженість Gemma/Jev), `dial.py` (діалектні слова проти оцінок).


### 4.7 Невідповідності та застереження

1. `config.yaml:99` вказує вчителя `model-whisper-m-fixed-leaks-10-epochs-epoch05.ckpt`, але розмітки цим вчителем у `output/pseudo-labelling/` немає; є лише whisper-uk-epoch00 (на ній навчено всіх студентів з `evaluation/`) і 5-epochs-epoch04.
2. `pseudo_labelling.batch_size` у конфігу 96, а всі фактичні запуски — 64 (`run.json`).
3. Модель судді для обох `label_marks.csv` на VOA ніде не записана (`run.json` псевдорозмітки її не містить; `judge_model` у тодішніх конфігах порожній). Для 5-epochs-01 використано decision-бекенд (є `score/confidence`), для whisper-uk — чат-бекенд.
4. Оцінювання 5-epochs-01 неповне (236 457 з 351 904); запуск 5-epochs-02 неповний (127 168 рядків, без `complete`).
5. `judge.py` не відфільтровує порожні тексти перед надсиланням судді (14 порожніх у 5-epochs-01); студент їх відкидає сам.
6. Немає мінімальної тривалості шматка: 2 975 шматків коротші за 1 с (мін. 0,005 с).
7. Немає захисту від галюцинацій при генерації; 853–925 шматків мають > 200 слів на ≤ 30 с.
8. Графіки `analysis/voa-meta-plots/` побудовані з часткового чекпойнта метаданих (~2 600 кліпів, ~230 год), а не з повних 20 507 кліпів (2 463 год).
9. Звіт Jev: увесь прогін 161 814 запитів по одному тексту між 19:14:03 і 19:26:57 (≈ 13 хв) — неправдоподібно швидко для 16 потоків; можливо, `started` у `run.json` перезаписано при `--resume`. Тривалість прогонів суддів не варто наводити як факт.
10. `analysis/label-judge/google_gemma-4-26b-a4b-it-02/marks.csv` містить блок NUL-байтів (наслідок обірваного запису); `judge.py` такі рядки пропускає.
11. Параметри Silero VAD (поріг, мінімальні тривалості мовлення/тиші, padding) не задані в коді, версія пакета не зафіксована — у тексті роботи слід писати «стандартні параметри Silero VAD».

## 5. Навчання вчителя і студентів


Усі шляхи відносні до кореня репозиторію `asr_ssl`. Числа взято з файлів, указаних у дужках, або обчислено скриптом Python за `metrics.csv` (pandas, `env/bin/python`). Якщо значення в репозиторії немає, написано «не знайдено».

Умовні позначення:
1. «Епоха k» нумерується з 1, а в назвах чекпоінтів і в `metrics.csv` епохи нумеруються з 0 (`epoch00` — кінець першої епохи; `meta.yaml` запуску на 10 епох: `checkpoint: …-epoch09.ckpt`, `epoch: 9`).
2. `step` у `metrics.csv` — це крок оптимізатора (з урахуванням накопичення градієнтів). Перевірка: 62 149 кліпів / (16 × 2) = 1 942 кроки на епоху, а кінець епохи 1 в `metrics.csv` — `step` 1941.
3. Dev WER/CER під час навчання рахуються лише на Common Voice dev (10 955 кліпів), якщо не сказано інше.


### 5.1 Донавчання вчителя / орієнтира на Common Voice (`fine-tune/fine-tune.py`)

#### 5.1.1 Призначення

1. Донавчити Whisper Medium (`openai/whisper-medium`) на розміченій українській мові, щоб отримати модель-вчителя для псевдорозмітки і контрольну модель «кероване навчання на CV» (орієнтир).
2. Той самий скрипт виконує кроки ланцюжка `multi_step`, у яких є лише розмічені набори (`fine-tune/multi_step.py:7-8`).

#### 5.1.2 Принцип роботи (кроками)

1. Читається конфіг `config.yaml` або файл з `ASR_CONFIG` (`fine-tune/settings.py:1-4`). GPU фіксується через `CUDA_VISIBLE_DEVICES` до ініціалізації CUDA (`fine-tune/fine-tune.py:37-49`).
2. Фіксується сід: `L.seed_everything(cfg.seed, workers=True)` і `torch.set_float32_matmul_precision("high")` (`fine-tune/fine-tune.py:389-390`). Кожен воркер DataLoader отримує похідний сід (`_seed_worker`, `fine-tune/fine-tune.py:231-234`).
3. Завантажуються спліти `data.train`, `data.dev`, `data.test` (`fine-tune/speech_data.py:443-450`). Набори однієї ролі конкатенуються в один спліт (`config.yaml:89`).
4. Train-частини Common Voice і ukr_dialects проходять обов'язковий фільтр витоків (`fine-tune/speech_data.py:427-433`, `fine-tune/leak_filter.py:53`). Він видаляє з train кліпи, акустичні або SHA-дублікати яких є в dev/test (див. 1.5).
5. Кожен кліп: ресемплінг до 16 кГц, обрізання до `max_audio_seconds` = 30 с, log-mel ознаки через `WhisperFeatureExtractor` (`fine-tune/fine-tune.py:194-206`).
6. Мітка = префікс `<|uk|><|transcribe|><|notimestamps|>` + токени тексту + EOS. Текст обрізається так, щоб уся послідовність не перевищувала `max_label_tokens` = 448 (`fine-tune/fine-tune.py:186-188`, `207-209`). Паддінг міток значенням −100, тож він не входить у втрату (`fine-tune/fine-tune.py:218-222`).
7. Початкові ваги: `training.init`, якщо задано (Lightning `.ckpt` або id HF), інакше `model.id` (`fine-tune/fine-tune.py:415-416`, `fine-tune/init_weights.py:9-32`).
8. Кодувальник заморожується (`requires_grad = False`) і завжди лишається в режимі `eval`. Декодувальник навчається з увімкненим dropout (`fine-tune/fine-tune.py:255-260`, `269-272`).
9. Перед навчанням рахується dev «до донавчання» (`baseline_eval: true`, `fine-tune/fine-tune.py:524-527`). Це точка `step 0` на графіках dev.
10. Навчання: втрата — крос-ентропія декодера Whisper (`model(...).loss`, `fine-tune/fine-tune.py:274-281`). AdamW лише по параметрах з `requires_grad` і лінійний розклад LR з розігрівом, крок розкладу — кожен крок оптимізатора (`fine-tune/fine-tune.py:352-363`).
11. Dev рахується кожні `val_every` частки епохи (`val_check_interval`, `fine-tune/fine-tune.py:509`). На dev один прохід кодувальника дає і втрату, і жадібне декодування (`num_beams=1`, `do_sample=False`) (`fine-tune/fine-tune.py:293-326`). WER/CER/MER/WIL/WIP і кількості замін, видалень і вставок рахує `jiwer` після нормалізації: нижній регістр, `’`→`'`, вилучення пунктуації, стискання пробілів (`fine-tune/fine-tune.py:95-119`).
12. Dev-спліт сортується від найдовших міток, щоб OOM проявився одразу (`fine-tune/fine-tune.py:434-435`). Train перемішується, неповний останній батч відкидається (`drop_last=shuffle`, `fine-tune/fine-tune.py:419-433`).
13. Чекпоінти пише `BatchFractionCheckpoint` (`fine-tune/checkpoints.py:163`): повний стан (ваги, AdamW, scheduler) на 25, 50 і 75 % епохи (`range(step, 100, step)`, `fine-tune/checkpoints.py:217`) і в кінці кожної епохи. Після завершення проміжні видаляються (`fine-tune/fine-tune.py:531-535`, `fine-tune/checkpoints.py:26-45`). Перерване навчання продовжується з `last.ckpt` (`training.resume: true`, `fine-tune/fine-tune.py:450-456`).
14. Після навчання пишуться `results.json`, `metrics.csv` (CSVLogger) і графіки `plots/*.png`, а все це логиться в MLflow (`fine-tune/fine-tune.py:537-566`). Тест рахує окремо `evaluation/evaluate_training.py`: чекпоінт першої епохи і фінальний (`fine-tune/fine-tune.py:5-8`, `evaluation/evaluate_training.py:1-18`).

#### 5.1.3 Модель і параметри

| Параметр | Значення | Джерело |
|---|---|---|
| Базова модель | `openai/whisper-medium` | `config.yaml:26` |
| Заморожені параметри (кодувальник) | 307 216 384 | `output/fine-tune/*/train.log` (рядок «frozen encoder parameters»), однаково в усіх запусках |
| Навчані параметри (декодувальник) | 456 641 536 | там само («trainable decoder parameters») |
| Разом | 763 857 920 (обчислено) | сума двох рядків вище |
| Частка навчаних | 59,8 % (обчислено) | 456 641 536 / 763 857 920 |
| Оптимізатор | AdamW (`torch.optim.AdamW`) | `fine-tune/fine-tune.py:354` |
| Learning rate | 1e-5 | `config.yaml:67`; `version_*/hparams.yaml` усіх запусків |
| Розігрів (warmup) | 50 кроків оптимізатора | `config.yaml:68`; `hparams.yaml` |
| Weight decay | 0,01 | `config.yaml:75`; `hparams.yaml` (у найстарішому `model-whisper-uk/01-common-voice-only/hparams.yaml` поля немає) |
| Планувальник LR | лінійний розігрів 0→1e-5 за 50 кроків, далі лінійне спадання до 0 на `estimated_stepping_batches` | `fine-tune/fine-tune.py:355-359`; графік `learning_rate.png` (кінцеве значення 1,084e-08) |
| Batch size (train, на GPU) | 16 (запуски від 14.11.2026); 32 (запуски 08–09.11.2026) | `config.yaml:56`; `version_*/config.yaml`, `meta.yaml` |
| Накопичення градієнтів | 2 → ефективний батч 32 (запуски від 14.11); до цього 1, ефективний батч 32 | `config.yaml:76-77`; `fine-tune/settings.py:224-229` (за відсутності ключа 1) |
| Batch size dev під час навчання | 24 (10 епох), 8 (5 епох, 5 епох+FLEURS) | `train.log` («testing_after_train_batch») |
| Точність | `bf16-mixed` (усі запуски від 07.11.2026); `bf16-true` (старі `model-whisper-uk`, `student_whisper-commonvoice-based`) | `config.yaml:71-72`; `version_*/config.yaml`; `evaluation/results.csv`, колонка `precision` |
| Gradient clipping | 1,0 (норма) | `config.yaml:74`; `fine-tune/fine-tune.py:508` |
| Сід | 42 | `config.yaml:73`; `meta.yaml` |
| Макс. довжина аудіо | 30 с (довші обрізаються) | `config.yaml:87`; `fine-tune/fine-tune.py:198-200` |
| Макс. довжина мітки | 448 токенів разом з 3 префіксними і EOS | `config.yaml:88`; `fine-tune/fine-tune.py:208` |
| Аугментація | немає (SpecAugment, шум, зміна швидкості в коді відсутні; є лише dropout декодера) | пошук `augment|specaug|mask_time` у `fine-tune/*.py` — збігів немає; `fine-tune/fine-tune.py:257-259` |
| Частота валідації | 0,25 епохи (4 рази на епоху) у 5-епохових запусках; 0,5 епохи в 10-епоховому; 1,0 у старих | `version_*/config.yaml` (`training.val_every`); `fine-tune/settings.py:237` (за замовчуванням 1,0) |
| Декодування на dev | жадібне, `num_beams=1` | `fine-tune/fine-tune.py:300-307` |
| Кількість епох | 5 (орієнтир), 10 (вибір кількості епох), 2 (ранні запуски); у поточному `config.yaml` стоїть 7 | `meta.yaml`; `config.yaml:69` |
| Чекпоінти | кожні 0,25 епохи + кінець епохи; проміжні видаляються | `config.yaml:39`, `43`; `fine-tune/checkpoints.py:163-217` |
| Розмір проміжних чекпоінтів | 67,1 ГБ на 10 штук (≈6,7 ГБ на чекпоінт з AdamW) | `output/fine-tune/model-whisper-m-fixed-leaks-10-epochs/train.log` («deleted 10 mid-epoch checkpoints (67.1 GB)») |
| GPU | одна GPU сервера (`nvidia0` або `nvidia1`); модель GPU не знайдено | `train.log` («gpus ['nvidia0']»); `config.yaml:33-35` |
| Пам'ять GPU | train batch 8 ≈ 20 ГБ; оцінка batch 32 ≈ 11 ГБ, 64 ≈ 16 ГБ, 96 — OOM | коментарі `config.yaml:57`, `61-63` |

**Чому `bf16-mixed`, а не `bf16-true`.** У `bf16-true` ваги й стан AdamW зберігаються в bf16, а це лише близько 3 значущих цифр. Оновлення порядку 1e-5 від ваги здебільшого округлюються до нуля, і декодер майже не вчиться (`REVIEW_FINDINGS.md:44-52`). За оцінкою з `todo.md:5`, за `lr 1e-5` змінюється лише близько 10 % ваг декодера. У `bf16-mixed` master-ваги й стан AdamW лишаються у fp32, а обчислення йдуть у bf16 (`config.yaml:71`). Конфіги виправлено 07.11.2026 (`REVIEW_FINDINGS.md:52`). Вимога до GPU: Ampere або новіша (`VAST_AI.md:181`).

#### 5.1.4 Дані

| Роль | Набори | Кліпів | Джерело |
|---|---|---|---|
| Train (орієнтир, від 14.11) | Common Voice uk train | 62 149 (62 460 у `train_validated.tsv` мінус 311 витоків) | `output/fine-tune/model-whisper-m-fixed-leaks-*/train.log`, `results.json` |
| Train (запуски 08–09.11, до фільтра) | Common Voice uk | 62 460 | `output/fine-tune/model-whisper-uk-cv-fixed/train.log` |
| Train (T0, 10.10) | Common Voice uk, що містив dev | 73 415 | `output/fine-tune/model-whisper-uk/train.log` |
| Dev | Common Voice dev | 10 955 | `train.log` |
| Dev (+FLEURS-запуски) | CV dev + FLEURS validation | 10 955 + 325 = 11 280 | `model-whisper-m-fixed-leaks-5-epochs+fleurs/train.log` |
| Test (окремо, `evaluate.py`) | CV test 10 956, FLEURS 750, ukr_dialects 3 435, cv10_test 3 204 | 18 345 | `train.log` («test clips») |

1. Ролі наборів і назви сплітів: `fine-tune/dataset_roles.py:9-16`. FLEURS і ukr_dialects мають dev-спліт `validation`, opentts і tg_voices — лише train, cv10_test — лише test.
2. Common Voice читається з TSV (`train_validated.tsv`, `dev.tsv`, `test.tsv`, `unlabelled.tsv`) з `quoting=QUOTE_NONE`. Порожні речення й рядки без аудіо відкидаються, а злиті рядки зупиняють запуск (`fine-tune/common_voice.py:56-88`). `fine-tune/read_common_voice.py` лише пише маніфести CSV у `datasets/common-voice/`.
3. **Фільтр витоків (`fine-tune/leak_filter.py`).** Він читає `output/leak-clean/sha_matches.csv` і `acoustic_matches.csv`. Кожен рядок — кліп dev/test, запис якого є і в train, з полями `dataset_train`, `split_train`, `id_train`. Ці train-кліпи видаляються, а dev і test не змінюються (`fine-tune/leak_filter.py:1-16`, `56-71`). Якщо CSV немає, запуск зупиняється (`fine-tune/leak_filter.py:42-47`). Фільтр застосовується до `common_voice` і `ukr_dialects` (`fine-tune/leak_filter.py:26`). Фактичний вміст (обчислено pandas): `acoustic_matches.csv` — 311 унікальних train-кліпів Common Voice; `sha_matches.csv` — 7 train-кліпів ukr_dialects. 62 460 − 311 = 62 149, що збігається з `train.log`. За `docs/thesis/thesis_plan.md` (розд. 4, п. 3) ці 311 кліпів — акустичні дублікати записів CV10 test.
4. Перетин CV train і test за реченнями й дикторами свідомо лишено (`todo.md:3`, `REVIEW_FINDINGS.md:30`).

#### 5.1.5 Результати

Див. зведену таблицю всіх запусків у розділі 3 і таблицю по епохах у розділі 4.

Підсумок орієнтира `model-whisper-m-fixed-leaks-5-epochs`:

| Метрика | Значення | Джерело |
|---|---|---|
| Dev WER до навчання | 18,88 % | `version_0/metrics.csv`, step 0 |
| Dev WER у кінці (епоха 5) | 10,12 % (CER 2,66 %) | `version_1/metrics.csv`, step 9709 |
| Найкращий dev WER | 10,11 % (step 9224, 75 % епохи 5) | `version_1/metrics.csv` |
| Test WER CV / FLEURS / ukr_dialects (epoch04) | 10,68 / 12,98 / 66,75 % | `evaluation/results.csv`, run `whisper-m-fixed-leaks-5-epochs-epoch04-01` |
| Час | ≈81,7 хв на епоху (з 4 dev-оцінками, dev batch 8) | обчислено з міток часу `train.log` (81,9; 81,8; 81,6; 81,5 хв для епох 1–4) |

#### 5.1.6 Графіки (типи, однакові для всіх запусків)

`fine-tune/plot_metrics.py` (функція `write_plots`, рядки 269-334) будує з `metrics.csv` такі графіки. Їх описано за `output/fine-tune/model-whisper-m-fixed-leaks-10-epochs/version_0/plots/` і `output/fine-tune/model-whisper-uk/02-all-train-sets/plots/`.

1. `learning_rate.png` — LR (`lr-AdamW`) по кроках оптимізатора: короткий розігрів до 1e-5, далі пряма лінійного спадання до ≈0 (для 10 епох кінцеве значення 1,084e-08).
2. `train_loss.png` — `train_loss_step` по кроках: шумна крива, яка падає з ≈0,27–0,44 на початку до 2,3e-4 у кінці (10 епох).
3. `train_loss_epoch.png` — середня train-втрата за епоху (`train_loss_epoch`): 0,2316 → 0,000266 за 10 епох.
4. `val_loss.png` — dev-втрата по кроках: різкий спад з 0,967 до ≈0,18, мінімум 0,1777 у кінці епохи 2, далі монотонне зростання до 0,282 (10 епох). Це ознака надмірної впевненості моделі (overconfidence) за стабільного WER.
5. `val_error_rates.png` — dev WER, CER і MER (%) по кроках, у легенді останні значення (10 епох: WER 9,91 %, CER 2,54 %, MER 9,81 %).
6. `val_edit_counts.png` — кількість замін, видалень і вставок на dev по кроках. Заміни переважають: 11 607 → 6 069; видалення 1 706 → 852; вставки 1 344 → 791 (10 епох; `results.json`).
7. `test_error_rates.png` — стовпчики WER, CER і MER на тесті. Є лише в старих запусках, де тест рахувався в тому самому скрипті (`model-whisper-uk/*`, `student_model-whisper-uk`). Приклад `model-whisper-uk/02-all-train-sets`: WER 32,47 %, CER 13,69 %, MER 29,88 % (combined test).

#### 5.1.7 Скрипти для відтворення

1. `python main.py fine-tune --note "..."` — навчання, потім `evaluation/evaluate_training.py` (`main.py:14-18`).
2. Напряму: `env/bin/python fine-tune/fine-tune.py` (без тесту).
3. Паралельний запуск на другій GPU: `python main.py fine-tune --config 1` (бере `config_parallel.yaml`, `main.py:4-6`).
4. Графіки: `python main.py plot --watch` або `python fine-tune/plot_metrics.py --version N` (`fine-tune/plot_metrics.py:8-10`).
5. Оцінка: `python evaluation/evaluate_training.py --stage fine-tune` або `--checkpoint-dir <dir>` (`evaluation/evaluate_training.py:16-18`).
6. Маніфести CV: `python fine-tune/read_common_voice.py`.


### 5.2 Студент і багатокрокове навчання (`fine-tune/fine-tune-student.py`, `fine-tune/multi_step.py`)

#### 5.2.1 Призначення

1. Донавчити Whisper-студента на VOA-сегментах з псевдомітками вчителя, відібраними за оцінкою LLM-судді 0–5 (`accepted_mark`), за потреби разом з розміченим Common Voice в одному запуску.
2. `multi_step.py` запускає ланцюжок кроків, де кожен стартує з фінального чекпоінта попереднього (`fine-tune/multi_step.py:15-18`).

#### 5.2.2 Принцип роботи (кроками)

1. Мітки читаються з `student.labels` (`labels.csv`), оцінки — з `student.marks` (`label_marks.csv`). `latest` означає найновіший файл для `pseudo_labelling.teacher_model` (`fine-tune/fine-tune-student.py:3-6`).
2. За `filter_marks: true` лишаються ID, остання оцінка яких входить у `accepted_mark` (`fine-tune/voa_slices.py:40-48`, `fine-tune/fine-tune-student.py:628-632`). Порожні тексти пропускаються (`fine-tune/voa_slices.py:51-77`).
3. Аудіо VOA: сегмент mp3 з паркетних шардів декодується ffmpeg тими самими функціями, що й під час псевдорозмітки. Якщо є кеш `student.voa_cache` (int16, 16 кГц, `python main.py cache-voa`), читається кеш (`fine-tune/fine-tune-student.py:658-667`, `fine-tune/voa_slices.py:80-117`). Log-mel рахується на GPU (`LogMel`, `fine-tune/fine-tune-student.py:393-423`, `470-475`).
4. `student.extra_train` (наприклад, `[common_voice]`) додає розмічені train-набори через `ConcatDataset` (`fine-tune/fine-tune-student.py:669-675`). До них оцінки судді не застосовуються, а фільтр витоків застосовується так само (`load_train_split` → `_train_part`, `fine-tune/speech_data.py:427-440`).
5. **Змішування в межах епохи — `RowGroupSampler` (`fine-tune/fine-tune-student.py:184-261`):**
   1. порядок груп (row groups паркету) і файлів у групі перемішується з сідом `seed + epoch`, сегменти одного файлу йдуть поруч, щоб файл декодувався один раз (`:219-227`);
   2. індекси розмічених кліпів перемішуються окремо (`:228-229`);
   3. спершу йдуть усі батчі VOA, потім усі батчі розмічених даних: «The labelled batches go last in the epoch» (`:235-236`). Отже, кожна епоха закінчується на Common Voice, і в одному батчі псевдомітки й CV не змішуються;
   4. батчі VOA переставляються так, щоб кожен воркер DataLoader отримав суцільний відрізок (`_per_worker`, `:238-249`).
6. Модель, заморожування кодувальника, оптимізатор і розклад LR такі самі, як у вчителя (`fine-tune/fine-tune-student.py:450-453`, `569-581`). Початкові ваги — `student.init` (`fine-tune/fine-tune-student.py:655`).
7. Dev (Common Voice dev) — кожні `val_every` = 0,25 епохи. Після навчання один раз рахується Common Voice test (`trainer.test`, `fine-tune/fine-tune-student.py:826`).
8. **`multi_step.py`:** для кожного кроку формується повний конфіг (`step_config`, `fine-tune/multi_step.py:191-210`) і пишеться в `output/multi-step/<name>/step<N>.yaml` (`:226-234`). Скрипт навчання запускається з `ASR_CONFIG=<step.yaml>`: `data: [pseudo, ...]` → `fine-tune-student.py`, інакше `fine-tune.py` (`:297-323`). Крок 1 стартує з `multi_step.init`, крок N — з фінального чекпоінта кроку N−1 (`previous = str(checkpoint)`, `:336`). Оптимізатор і розклад LR у кожному кроці починаються заново (`:16-18`). Завершений крок пропускається (`:310-312`). Після кроку запускається `evaluate_training.py --final-only` (`evaluate: steps`, `:326-331`).
9. Назва кроку: `<chain>-s<N>-<data>`, наприклад `student_mix-m345+cv-m5+cv-s1-pseudo-m345-common_voice` (`meta.yaml` запусків).

#### 5.2.3 Модель і параметри (ланцюжки `student_mix-*`)

| Параметр | Значення | Джерело |
|---|---|---|
| Ініціалізація кроку 1 | `checkpoints/model-whisper-uk/model-whisper-uk-epoch00.ckpt` (вчитель Т0) | `output/multi-step/*/step1.yaml:71`; `meta.yaml` |
| Ініціалізація кроку 2 | `…-s1-…_01/…-s1-…-epoch00.ckpt` (фінал кроку 1) | `output/multi-step/*/step2.yaml:71` |
| Псевдомітки | `output/pseudo-labelling/model-whisper-uk-epoch00-01/labels.csv` (вчитель Т0) | `meta.yaml` (поле `labels`) |
| Епох на крок | 1 | `output/multi-step/*/step*.yaml` (`student.max_epochs`) |
| LR / warmup / wd | 1e-5 / 50 / 0,01 | `step1.yaml:92-99`; `hparams.yaml` |
| Batch size train | 16, без накопичення градієнтів | `step1.yaml:89-91`; кроки: (342 105 + 62 460) / 16 = 25 285 ≈ max step 25 284 (обчислено) |
| Batch size eval | 8 | `step1.yaml:91` |
| Точність | `bf16-mixed` | `step1.yaml:96`; `meta.yaml` |
| Clip / seed | 1,0 / 42 | `step1.yaml:97-98` |
| val_every / checkpoint_every | 0,25 / 0,25 | `step1.yaml:105-106` |
| Воркери DataLoader | 4 у ланцюжках (зараз `student.num_workers: 12`) | `step1.yaml:95`; `config.yaml:120` |
| GPU | `nvidia1` (ланцюжок m2345), `nvidia0` (ланцюжок m345) | відповідні `train.log` |

Ланцюжки:

| Ланцюжок | Крок | accepted_mark | VOA-сегментів | CV train | Разом | Частка CV, % (обчислено) |
|---|---|---|---|---|---|---|
| `student_mix-m2345+cv-m45+cv` | 1 | {2,3,4,5} | 342 105 (17 825 файлів) | 62 460 | 404 565 | 15,4 |
| | 2 | {4,5} | 216 957 (17 570 файлів) | 62 460 | 279 417 | 22,4 |
| `student_mix-m345+cv-m5+cv` | 1 | {3,4,5} | 305 867 (17 775 файлів) | 62 460 | 368 327 | 17,0 |
| | 2 | {5} | 105 040 (16 409 файлів) | 62 460 | 167 500 | 37,3 |

Джерело кількостей: рядки `student slices` і `extra train common_voice` у `output/fine-tune/student_mix-*/train.log`. CV тут 62 460, бо ланцюжки прогнано 07–08.11.2026, до фільтра витоків (14.11.2026).

#### 5.2.4 Результати студентів (dev під час навчання; test CV)

| Запуск | Dev WER по 0,25 епохи, % | Dev loss по 0,25 епохи | Test CV у скрипті, % | Test CV / FLEURS / діалекти (`results.csv`), % |
|---|---|---|---|---|
| m2345 s1 | 15,47; 15,67; 14,82; 13,83 | 0,329; 0,355; 0,368; 0,208 | 15,49 | 15,50 / 11,61 / 70,00 |
| m45 s2 | 14,98; 14,73; 14,51; 13,36 | 0,334; 0,347; 0,355; 0,199 | 14,57 | 14,59 / 11,61 / 71,04 |
| m345 s1 | 15,73; 15,13; 14,81; 14,77* | 0,335; 0,343; 0,361; 0,396* | 16,75 | 16,77 / 11,14 / 62,24 |
| m5 s2 | 15,56; 15,15; 13,82; 12,55 | 0,407; 0,403; 0,214; 0,194 | 14,23 | 14,27 / 11,21 / 78,07 |

*m345 s1 перервався, а останню точку дав відновлений `version_1` (step 23015). Джерела: `output/fine-tune/student_mix-*/version_*/metrics.csv`, рядок `test {...}` у `train.log`, `evaluation/results.csv`.

Спостереження: dev-втрата різко падає на останній чверті епохи, коли йдуть батчі CV (наприклад, 0,368 → 0,208 у m2345 s1). Це прямий наслідок порядку «спершу VOA, потім CV».

#### 5.2.5 Графіки

1. Для `student_mix-*` графіків `plots/` у репозиторії немає, є лише `metrics.csv`.
2. `output/fine-tune/student_model-whisper-uk/version_1_voa-only/plots/{learning_rate,train_loss,train_loss_epoch,val_loss,val_error_rates,val_edit_counts,test_error_rates}.png` — старий студент лише на VOA (застарілий, див. розд. 3).

#### 5.2.6 Час

1. m5 s2: епоха 180,1 хв (15:41:25 → 18:41:29), далі тест CV ≈16 хв (`output/fine-tune/student_mix-m345+cv-m5+cv-s2-pseudo-m5-common_voice/train.log`).
2. m2345 s1: від старту до запису `results.json` 15:21 → 21:04 (≈5 год 43 хв разом з тестом); m45 s2: 21:46 → 02:21 (≈4 год 35 хв) (`train.log`).

#### 5.2.7 Скрипти для відтворення

1. `python main.py multi-step --dry-run`, потім `python main.py multi-step` (`main.py:28-32`), або `python fine-tune/multi_step.py`.
2. Один студент: `python main.py student` або `env/bin/python fine-tune/fine-tune-student.py`.
3. Кеш VOA: `python main.py cache-voa [--all]` (`main.py:24-27`).
4. Пайплайн цілком: `python main.py pipeline` (етап `multi_step: run`, `config.yaml:209`; `pipeline.py:378-380`).
5. Конфіги кроків для точного відтворення: `output/multi-step/student_mix-m2345+cv-m45+cv/step{1,2}.yaml`, `output/multi-step/student_mix-m345+cv-m5+cv/step{1,2}.yaml`, `evaluate.yaml`.


### 5.3 Усі запуски в `output/fine-tune/`

Precision, init, дані й epochs — з `version_*/meta.yaml` і `version_*/config.yaml`. Для студентів — секція `student`, бо `config.yaml` у теці версії є копією всього конфігу, і його секції `training`/`data` до студента не належать. Dev — з `metrics.csv` (Python). Test CV — з `evaluation/results.csv`.

| # | Запуск (тека) | Init | Train-дані | Рядків train | Епох | Precision | LR | Ефект. батч | Dev WER старт → кінець, % | Найкращий dev WER, % (крок) | Dev loss кінець | Test CV, % | Статус |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `model-whisper-uk/01-common-voice-only` (10.10) | whisper-medium | CV (з dev) | 73 415 | 2 | bf16-true | 1e-5 | 16 (обчислено: 73 415 / 4 587 кроків; 2 GPU DDP) | 18,85 → 14,83 | 14,83 (9175) | 0,2309 | 17,40 (epoch00, Т0) | Т0: застарілий, але потрібен як вчитель і старт студентів. Dev входив у train |
| 2 | `model-whisper-uk/02-all-train-sets` (26.10) | whisper-medium | CV, FLEURS, opentts, tg_voices, ukr_dialects | 139 797 | 2 | bf16-true | 1e-5 | 32 | 35,14 → 33,75 (dev CV+FLEURS+діал.) | 33,75 (8735) | 0,4007 | 17,64 / 17,54 | Застарілий: bf16-true, витік dev, тестові домени в train |
| 3 | `model-whisper-uk-cv-fixed` v0 (09.11) | whisper-medium | CV | 62 460 | 2 | bf16-mixed | 1e-5 | 32 | 18,88 → 10,86 | 10,86 (3901) | 0,1666 | 12,94 / 11,66 | Проміжний: до фільтра витоків (311 дублікатів CV10). v1 — перерване продовження до 5 епох (3 рядки) |
| 4 | `model-whisper-uk-cv-fleurs` (08–09.11) | whisper-medium | CV + FLEURS | 65 270 | 2 | bf16-mixed | 1e-5 | 32 | 18,42 → 11,01 (CV+FLEURS dev) | 11,01 (4077) | 0,1671 | 13,15 / 11,60 | Виключено: FLEURS у train |
| 5 | `model-whisper-m-fixed-leaks-5-epochs` (14–15.11) | whisper-medium | CV | 62 149 | 5 | bf16-mixed | 1e-5 | 32 | 18,88 → 10,12 | 10,11 (9224) | 0,2185 | 10,68 (epoch04) | **Орієнтир у тезах** |
| 6 | `model-whisper-m-fixed-leaks-5-epochs+fleurs` (16.11) | whisper-medium | CV + FLEURS | 64 959 | 5 | bf16-mixed | 1e-5 | 32 | 18,42 → 10,15 (CV+FLEURS dev) | 10,15 (10147) | 0,2214 | 10,67 | Виключено: FLEURS у train |
| 7 | `model-whisper-m-fixed-leaks-10-epochs` (16–17.11) | whisper-medium | CV | 62 149 | 10 | bf16-mixed | 1e-5 | 32 | 18,83 → 9,91 | 9,906 (19419) | 0,2823 | 13,10 (epoch00) / 10,29 (epoch09) | Лише для вибору кількості епох; у таблицю тез не входить |
| 8 | `student_model-whisper-uk/version_1_voa-only` (13–14.10) | не знайдено | VOA-псевдомітки | 216 957 сегментів | 2 | не знайдено (до виправлення 07.11) | 1e-5 | 32 | 17,16 → 16,86 | 16,83 (5081) | 0,3465 | 18,57 (тест у скрипті) | Застарілий |
| 9 | `student_whisper-commonvoice-based` v0 (mark 5) | Т0 | VOA {5} | ≈105 040 (обчислено: 3 281 крок × 32) | 2 | bf16-true | 1e-5 | 32 | 15,16 → 15,35 | 15,16 (819) | 0,3005 | 17,56 | Застарілий: bf16-true |
| 10 | `student_whisper-commonvoice-based` v3–v4 (усі оцінки) | Т0 | VOA {0..5} | ≈351 840 (10 995 кроків × 32) | 2 | bf16-true | 1e-5 | 32 | 15,88 → 16,01 | 15,74 (16493) | 0,3075 | 18,02 | Застарілий: bf16-true |
| 11 | `student_whisper-commonvoice-based` v5–v7 (mark 3) | Т0 | VOA {3} | 88 910 | 2 | bf16-true | 1e-5 | 32 | 15,31 → 15,33 | 15,26 (1387) | 0,2909 | 17,70 | Застарілий: bf16-true |
| 12 | `student_mix-m2345+cv-m45+cv-s1-…` | Т0 | VOA {2–5} + CV | 404 565 | 1 | bf16-mixed | 1e-5 | 16 | 15,47 → 13,83 | 13,83 (25283) | 0,2081 | 15,50 | Актуальний (старт від Т0) |
| 13 | `student_mix-m2345+cv-m45+cv-s2-…` | №12 | VOA {4,5} + CV | 279 417 | 1 | bf16-mixed | 1e-5 | 16 | 14,98 → 13,36 | 13,36 (17459) | 0,1992 | 14,59 | Актуальний |
| 14 | `student_mix-m345+cv-m5+cv-s1-…` | Т0 | VOA {3–5} + CV | 368 327 | 1 | bf16-mixed | 1e-5 | 16 | 15,73 → 14,77 | 14,77 (23015) | 0,3963 | 16,77 | Актуальний |
| 15 | `student_mix-m345+cv-m5+cv-s2-…` | №14 | VOA {5} + CV | 167 500 | 1 | bf16-mixed | 1e-5 | 16 | 15,56 → 12,55 | 12,55 (10467) | 0,1940 | 14,27 | Актуальний |

Примітки:
1. У студентів «старт» dev — перша оцінка після 0,25 епохи, бо `baseline_eval: false` (`config.yaml:138`).
2. Критерії застарілості: `bf16-true` (`REVIEW_FINDINGS.md:44-52`, `todo.md:5`); train, що містив CV dev (73 415 рядків; `todo.md:3-5`); тестові домени в train — FLEURS або ukr_dialects (`docs/thesis/thesis_plan.md`, розд. 4, п. 4); моделі в `bf16-true` у тези не брати, крім Т0 (`docs/thesis/thesis_plan.md`, розд. 1, п. 2; розд. 4, п. 1).
3. Train-loss за епоху для основних запусків (`train_loss_epoch`): №5 — 0,2312; 0,0982; 0,0373; 0,0118; 0,0036. №6 — 0,2230; 0,0948; 0,0363; 0,0116; 0,0035. №3 — 0,2289; 0,0971. №4 — 0,2202; 0,0930. №1 — 0,2835; 0,2471. №2 — 0,5098; 0,4614.
4. Dev WER по оцінках для №5 (val_every 0,25): 18,88 | 15,35; 13,61; 12,99; 12,03 | 11,81; 11,46; 11,28; 11,14 | 10,80; 10,68; 10,67; 10,43 | 10,93; 10,43; 10,30; 10,26 | 10,24; 10,27; 10,11; 10,12 (`version_0` + `version_1`).
5. Dev WER по оцінках для №6 (CV+FLEURS dev): 18,42 | 15,19; 13,69; 12,99; 12,16 | 12,16; 11,33; 11,13; 10,83 | 10,54; 10,73; 10,93; 10,55 | 10,38; 10,43; 10,29; 10,85 | 10,22; 10,27; 10,19; 10,15.

Графіки запусків (кожна тека містить 6 стандартних PNG: `learning_rate`, `train_loss`, `train_loss_epoch`, `val_edit_counts`, `val_error_rates`, `val_loss`):
1. `output/fine-tune/model-whisper-m-fixed-leaks-10-epochs/version_0/plots/*.png` — повні 10 епох.
2. `output/fine-tune/model-whisper-m-fixed-leaks-5-epochs/version_1/plots/*.png` — **лише відновлена частина (епоха 5, кроки 8253–9709)**: `version_0` графіків не має, тож повну криву орієнтира треба перебудувати з двох `metrics.csv`.
3. `output/fine-tune/model-whisper-m-fixed-leaks-5-epochs+fleurs/version_0/plots/*.png` — 5 епох, CV+FLEURS.
4. `output/fine-tune/model-whisper-uk-cv-fixed/version_0/plots/*.png` — 2 епохи.
5. `output/fine-tune/model-whisper-uk-cv-fleurs/version_0/plots/*.png` — 2 епохи.
6. `output/fine-tune/model-whisper-uk/01-common-voice-only/plots/*.png` і `.../02-all-train-sets/plots/*.png` — є ще `test_error_rates.png`.
7. `output/fine-tune/student_model-whisper-uk/version_1_voa-only/plots/*.png` — є ще `test_error_rates.png`.


### 5.4 Вибір кількості епох: `model-whisper-m-fixed-leaks-10-epochs`

Джерело: `output/fine-tune/model-whisper-m-fixed-leaks-10-epochs/version_0/metrics.csv` (21 dev-оцінка, `val_every` 0,5). Train — 62 149 кліпів CV, dev — 10 955 кліпів CV, bf16-mixed, ефективний батч 32, 1 942 кроки на епоху. Значення у %, крім втрат.

| Епоха | Крок (½) | Dev WER ½ | Dev loss ½ | Крок (кінець) | Dev WER кінець | Dev CER кінець | Dev loss кінець | Train loss епохи | ΔWER до попер. кінця, п.п. |
|---|---|---|---|---|---|---|---|---|---|
| 0 (до навчання) | — | — | — | 0 | 18,83 | 5,15 | 0,9668 | — | — |
| 1 | 970 | 13,71 | 0,2206 | 1941 | 12,29 | 3,20 | 0,1897 | 0,2316 | −6,54 |
| 2 | 2912 | 11,64 | 0,1875 | 3883 | 11,36 | 2,85 | **0,1777** (мін.) | 0,0993 | −0,93 |
| 3 | 4854 | 10,68 | 0,1876 | 5825 | 10,68 | 2,79 | 0,1827 | 0,0388 | −0,68 |
| 4 | 6796 | 10,65 | 0,2134 | 7767 | 10,39 | 2,67 | 0,2247 | 0,0145 | −0,29 |
| 5 | 8738 | 10,08 | 0,2260 | 9709 | 10,20 | 2,65 | 0,2332 | 0,0059 | −0,19 |
| 6 | 10680 | 10,09 | 0,2400 | 11651 | 10,13 | 2,62 | 0,2471 | 0,0029 | −0,07 |
| 7 | 12622 | 9,97 | 0,2429 | 13593 | 9,91 | 2,54 | 0,2558 | 0,0016 | −0,22 |
| 8 | 14564 | 9,91 | 0,2582 | 15535 | 10,15 | 2,79 | 0,2685 | 0,00084 | +0,24 |
| 9 | 16506 | 9,93 | 0,2730 | 17477 | 10,20 | 2,70 | 0,2783 | 0,00041 | +0,05 |
| 10 | 18448 | 10,19 | 0,2833 | 19419 | **9,91** (мін. 9,906) | 2,54 | 0,2823 | 0,00027 | −0,29 |

Висновки, які підтримують дані:
1. За перші 5 епох dev WER падає на 8,63 п.п. (18,83 → 10,20). За наступні 5 — лише на 0,29 п.п. (10,20 → 9,91), і крива коливається в межах 9,91–10,20 % (обчислено з таблиці).
2. Dev-втрата мінімальна в кінці епохи 2 (0,1777) і далі росте до 0,2823, тоді як train-втрата падає майже до нуля (0,00027). Модель запам'ятовує train і стає надмірно впевненою. WER від цього не погіршується, але й не покращується.
3. Найкращий dev WER 9,906 % (кінець епохи 10) практично дорівнює 9,908 % у кінці епохи 7 (`metrics.csv`, кроки 19419 і 13593).
4. Test CV для порівняння (`evaluation/results.csv`): 10 епох epoch09 — 10,29 %; орієнтир 5 епох epoch04 — 10,68 %. Вибір за тестом у `docs/thesis/thesis_plan.md` (розд. 3.3, примітка 5) прямо відкинуто.
5. Застереження: розклад LR залежить від `max_epochs`, тож кінець епохи 5 у 10-епоховому запуску (10,20 %) і кінець 5-епохового запуску (10,12 %) — різні моделі.

Графіки: `output/fine-tune/model-whisper-m-fixed-leaks-10-epochs/version_0/plots/val_error_rates.png` (плато WER), `.../val_loss.png` (U-подібна dev-втрата з мінімумом біля кроку 3 900), `.../train_loss_epoch.png`.

![output/fine-tune/model-whisper-m-fixed-leaks-10-epochs/version_0/plots/val_error_rates.png](../../output/fine-tune/model-whisper-m-fixed-leaks-10-epochs/version_0/plots/val_error_rates.png)


Час (обчислено з міток часу «epoch N decoder_training» у `train.log`): dev до навчання 5,5 хв (batch 24); епохи 1–10: 78,0; 71,9; 59,1; 58,9; 58,8; 58,8; 59,0; 59,4; 60,1; 65,4 хв (з двома dev-оцінками на епоху; остання — разом з видаленням чекпоінтів). Разом 10,58 год (21:48 16.11 → 08:23 17.11.2026). Для порівняння: 5 епох — ≈81,7 хв на епоху (4 dev-оцінки, dev batch 8); 5 епох + FLEURS — 87,8; 88,0; 87,5; 90,8; 111,4 хв.


### 5.5 Інфраструктура

#### 5.5.1 MLflow (`fine-tune/tracking.py`, `fine-tune/backfill_mlflow.py`)

1. Увімкнення: секція `mlflow` (`config.yaml:49-53`): `enabled: true`, сховище `sqlite:///output/mlflow/mlflow.db`, артефакти `output/mlflow/artifacts`, експеримент `asr-whisper-uk`. Оцінки йдуть в експеримент `asr-whisper-uk-evaluation`.
2. Що логиться (`fine-tune/tracking.py:1-22`):
   1. один training run на теку чекпоінтів, метрики Lightning по кроках (`train_loss_step`, `train_loss_epoch`, `lr-AdamW`, `val_loss`, `val_wer`, `val_cer`, …);
   2. ті самі метрики по епохах: `epoch/train_loss`, `epoch/val_*` (`_epoch_callback`, `fine-tune/tracking.py:247-285`);
   3. evaluation run на кожну теку `evaluation/<model>-XX` (`test/<metric>`, `test/<dataset>/<metric>`) з тегом `train_run_id`;
   4. батьківський run на кожен ланцюжок `multi_step`, кроки вкладено в нього;
   5. параметри конфігу, кількості кліпів, git commit (`config_params`, `clip_params`, `git_commit`).
3. Продовжене навчання пише в той самий run: `mlflow_run_id` зберігається в `meta.yaml` (`fine-tune/tracking.py:20-21`). Помилка трекінгу лише логиться і навчання не зупиняє (`fine-tune/tracking.py:4-5`).
4. `backfill_mlflow.py` імпортує старі `version_N` (`metrics.csv`, `meta.yaml`, `config.yaml`) і `evaluation/<model>-XX`. Уже імпортовані теки (тег `source_dir`) пропускаються (`fine-tune/backfill_mlflow.py:1-15`).
5. Вміст сховища (запит до `output/mlflow/mlflow.db` у режимі read-only): 37 runs — 20 в `asr-whisper-uk` і 17 в `asr-whisper-uk-evaluation`.
6. Перегляд: `mlflow ui --backend-store-uri sqlite:///output/mlflow/mlflow.db` (`config.yaml:46`).

#### 5.5.2 Оцінка після навчання (`evaluation/evaluate_training.py`)

1. Оцінюються два чекпоінти: `<name>-epoch00.ckpt` і той, на який указує `last.ckpt`. Для одноепохових кроків це один файл. `--final-only` — лише фінальний. Уже наявні рядки `results.csv` пропускаються (`evaluation/evaluate_training.py:1-18`).
2. Тест-скор, порахований у `fine-tune-student.py`, і рядок `evaluate.py` у `results.csv` трохи різняться: 15,49 / 15,50; 14,57 / 14,59; 16,75 / 16,77; 14,23 / 14,27 % (`train.log` проти `evaluation/results.csv`). Для тез брати `results.csv`.

#### 5.5.3 vast.ai (`VAST_AI.md`, `config.vast.yaml`) і HF (`HF_UPLOAD.md`)

1. `config.vast.yaml` розширює `config.yaml` (`extends: config.yaml`, `config.vast.yaml:13`): дані в `/workspace/data`, чекпоінти в `/workspace/checkpoints`, одна GPU `nvidia0` (`config.vast.yaml:15-28`). У ньому лишилися застарілі `data.train` з п'ятьма наборами, `model.name: model-whisper-uk` і `student.accepted_mark: [0..5]` (`config.vast.yaml:20-35`).
2. Вимоги: GPU Ampere+ для bf16 (A100, A6000, RTX 3090/4090, L40S, H100), ≥ 24 ГБ, диск ≥ 200 ГБ (≥ 300 ГБ для всіх етапів) (`VAST_AI.md:177`, `181`, `232`).
3. `HF_UPLOAD.md`: у приватний HF-репозиторій заливаються Common Voice, FLEURS, чекпоінт вчителя `checkpoints/model-whisper-uk/model-whisper-uk-epoch00.ckpt` і псевдорозмітка в теці `model-whisper-uk-epoch00-01` (`HF_UPLOAD.md:17`, `22`, `82-91`). Публічні набори завантажує `scripts/vast/download_data.py` (`config.yaml:220-230`).
4. Чи реально проводилося навчання на vast.ai, з логів не видно. Усі `train.log` показують сервер `/mnt/data/rberehun` і GPU `nvidia0`/`nvidia1`.

#### 5.5.4 GPU і час

1. Модель GPU не знайдено: ні в логах, ні в MLflow, ні в конфігах. Відомо лише, що на сервері дві GPU (`nvidia0`, `nvidia1`) і кожен запуск займає одну (`train.log`; `config.yaml:31-35`). Т0 (10.10.2026) навчався на двох GPU через DDP (`output/fine-tune/model-whisper-uk/train.log`: «devices ['nvidia0', 'nvidia1']»).
2. Час на епоху — у розд. 1.5, 2.6 і 4.


### 5.6 Знайдені невідповідності

1. **Хронологія вибору 5 епох.** Орієнтир на 5 епох навчено 14–15.11.2026, а запуск на 10 епох, яким обґрунтовано «5 епох», — 16–17.11.2026 (`train.log`). Обґрунтування апостеріорне; формулювання в тезах краще пом'якшити.
2. **Т0 — чекпоінт `epoch00` двоепохового запуску.** Його LR-розклад був розрахований на 2 епохи, тож Т0 — модель з середини розкладу (LR не спав до 0). Dev Т0 (14,85 %) нечесний: CV dev був у train (73 415 рядків).
3. **Вчитель у поточному `config.yaml` не той, що дав псевдомітки.** `pseudo_labelling.teacher_model` указує на `model-whisper-m-fixed-leaks-10-epochs-epoch05.ckpt` (`config.yaml:99`, кінець 6-ї епохи), а всі наявні студенти вчилися на мітках `model-whisper-uk-epoch00-01` (`meta.yaml`, поле `labels`).
4. **`cv_train_rows: 62460` у `meta.yaml` і `results.csv`** записано і для запусків після фільтра витоків, хоча реально в train було 62 149 кліпів (`train.log`). У `thesis_plan.md` (2.1) орієнтир описано як «train 62 460».
5. **Кількість параметрів.** У тезах — 769 млн (`thesis_plan.md`, 3.2, п. 4), у логах — 763 857 920 (307 216 384 + 456 641 536). Ймовірно, різниця в тому, як рахуються спільні (tied) ваги вихідного шару. Рекомендовано писати число з логу.
6. **Студенти в ланцюжках вчилися на CV до фільтра витоків** (62 460), а орієнтир — після (62 149).
7. **Графіки орієнтира неповні**: `model-whisper-m-fixed-leaks-5-epochs/version_1/plots` покривають лише відновлену епоху 5.
8. **`config.yaml` у теці версії студента** — копія всього конфігу. Його `training.precision`/`data.train` (наприклад, 5 наборів і `bf16-true` у `student_whisper-commonvoice-based`) вводять в оману. Справжні параметри студента — у секції `student` і в `meta.yaml`.
9. Різні `val_every` (0,25 у запусках на 5 епох, 0,5 у 10-епоховому) і різний dev batch (8 проти 24) роблять час на епоху непорівнянним: 81,7 проти ≈59 хв.
10. Поточний `config.yaml` (`max_epochs: 7`, `model.name: …5-epochs+fleurs-parallel`) не відповідає жодному з фінальних запусків. Відтворювати треба з `output/fine-tune/<run>/version_*/config.yaml` або з `output/multi-step/*/step*.yaml`.

## 6. Оцінювання, результати, витоки даних


Джерела: репозиторій `asr_ssl` станом на 2026-11-24. Усі шляхи відносні до кореня репозиторію. Значення WER/CER у відсотках, якщо не зазначено інше. Позначка «обчислено» означає, що число отримано під час підготовки цього матеріалу власним скриптом (`docs/diploma/materials_scripts/utt.py`, `docs/diploma/materials_scripts/clean.py`, пословна й посимвольна відстань Левенштейна з тією самою нормалізацією, що й у `fine-tune/fine-tune.py`), а не прочитано з файлу.


### 6.1 Метод оцінювання моделей (`evaluation/evaluate.py`)

#### 6.1.1 Призначення

Оцінити модель Whisper (оригінальну з Hugging Face, теку моделі або Lightning-чекпоінт `.ckpt`) на кожному тестовому наборі з `config.yaml → data.test` окремо та на всіх разом, зберегти транскрипції, метрики, графіки й рядок у зведеній таблиці `evaluation/results.csv`.

#### 6.1.2 Принцип роботи (кроками)

1. Визначити джерело моделі: без `--model` — `model.id` з конфігу (`hf`); файл `.ckpt` — ваги чекпоінта поверх `model.id`; тека — `from_pretrained` (`evaluation/evaluate.py:141-154`, `:181-201`). З чекпоінта беруться лише ключі з префіксом `model.` (`:190-194`).
2. Створити теку прогону `evaluation/<slug>-NN/` з автонумерацією (`:168-178`), записати `meta.yaml` (поля навчання з meta чекпоінта: `init`, `train_data`, `accepted_mark`, `precision`, `max_epochs`, …, плюс `evaluated_at`, `git_commit`, `note`; `:95-138`).
3. Завантажити тестові спліти через `ft.load_splits()["test"]` (`:572`), тобто `fine-tune/speech_data.py:load_configured_splits` (`:443-450`). Тестові спліти не фільтруються (фільтр витоків застосовується лише до train, `fine-tune/speech_data.py:427-434`).
4. Для кожного кліпу: ресемплінг до 16 кГц, обрізання до `max_audio_seconds`, log-mel через `WhisperFeatureExtractor`; еталон токенізується, обрізається до `max_label_tokens − 4` токенів і отримує префікс `<|uk|><|transcribe|><|notimestamps|>` + EOS (`fine-tune/fine-tune.py:180-215`).
5. Інференс у `torch.inference_mode()` + `autocast(bfloat16)`: обчислюється teacher-forced loss і жадібне декодування `model.generate(num_beams=1, do_sample=False, use_cache=True, language, task="transcribe")` (`evaluation/evaluate.py:256-266`).
6. Гіпотези й еталони декодуються `batch_decode(skip_special_tokens=True, clean_up_tokenization_spaces=False)`; еталон відновлюється з токенізованих міток (`:269-279`), тобто це той самий текст, що бачить модель під час навчання.
7. Метрики рахуються функцією `ft.metrics_by_source` (`fine-tune/fine-tune.py:122-157`) після нормалізації тексту (п. 1.3).
8. Записуються `predictions.csv` (dataset, audio_path, reference, hypothesis), `results.json`, 4 графіки, додається рядок у `results.csv` (`evaluation/results_table.py:add_run`) і запис у MLflow (`tracking.log_evaluation`).

#### 6.1.3 Нормалізація тексту (дослівно)

`fine-tune/fine-tune.py:87-98`:

```python
_PUNCTUATION = re.compile(r"[^\w\s']+", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = _PUNCTUATION.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
```

Правила:

1. Нижній регістр (`str.lower`, Unicode).
2. Типографський апостроф `’` (U+2019) замінюється на ASCII `'`. Інші варіанти (`ʼ`, `‘`, `` ` ``) не замінюються — це робить лише `datasets/leak-clean/leak_common.py:28,58-71`, який використовується для пошуку витоків, не для WER.
3. Усі символи, що не є «словесними» (`\w`: літери, цифри, `_`), пробілами чи `'`, замінюються пробілом (пунктуація, дефіс, лапки).
4. Послідовності пробільних символів стискаються до одного пробілу, краї обрізаються.
5. Числа не перетворюються на слова, `ґ`/`г` не уніфікуються, дефісні слова розбиваються на два.
6. Пари з порожнім нормалізованим еталоном відкидаються з підрахунку (`fine-tune/fine-tune.py:103`). У всіх 8 перевірених прогонах таких пар 0 (обчислено, `docs/diploma/materials_scripts/utt.py`).

Та сама функція скопійована в `fine-tune/fine-tune-student.py:91`, `datasets/leak-clean/clean_common_voice_train.py:41-48` і `docs/thesis/figures/combined_without_cv10.py:36-43`.

#### 6.1.4 Обчислення WER/CER

Бібліотека `jiwer` (`fine-tune/fine-tune.py:56,101-119`). Версія в `requirements.txt` не закріплена (`REVIEW_FINDINGS.md`, п. 19.3).

1. `jiwer.process_words(refs, hyps)` над усіма парами корпусу → `wer`, `mer`, `wil`, `wip`, кількості `substitutions`, `deletions`, `insertions`, `hits`.
2. `jiwer.cer(refs, hyps)` — посимвольна метрика по корпусу.
3. Метрики корпусні (micro-average): WER = (S + D + I) / N, де суми беруться по всіх реченнях набору, а не середнє WER речень.

Формула для тексту: WER = (S + D + I) / (S + D + H), де H — кількість правильно розпізнаних слів.

#### 6.1.5 `combined_weighted` і `combined_mean`

| Назва | Визначення | Джерело |
|---|---|---|
| `combined_weighted` | корпусна метрика по всіх кліпах усіх тестових наборів разом; кожен набір важить пропорційно кількості слів (WER) чи символів (CER) | `fine-tune/fine-tune.py:130-138` |
| `combined_mean` | просте середнє арифметичне per-dataset метрик (`wer`, `cer`, `mer`, `wil`, `wip`); кожен набір важить однаково; додається лише якщо наборів ≥ 2 | `fine-tune/fine-tune.py:153-176` |
| `combined` | стара назва `combined_weighted` у ранніх `results.json`; перейменовується при читанні | `evaluation/results_table.py:67-68`, `evaluation/batch-compare.py:45-46` |
| `combined_no_cv10` | `combined_weighted`, перерахований з `predictions.csv` лише для `common_voice`, `fleurs`, `ukr_dialects` (15 141 кліп) | `docs/thesis/figures/combined_without_cv10.py:33,46-53` |

У `batch-compare.py` (`:97-113`) для старих результатів без `combined_mean` воно добудовується як середнє по наборах, назва яких не починається з `combined`.

#### 6.1.6 Параметри

| Параметр | Значення | Джерело |
|---|---|---|
| Базова модель | `openai/whisper-medium` | `config.yaml:26` |
| Мова / задача | `uk` / `transcribe` | `config.yaml:29`, `evaluation/evaluate.py:233-234,261-262` |
| Декодування | жадібне: `num_beams=1`, `do_sample=False`, `use_cache=True` | `evaluation/evaluate.py:263-265` |
| Максимальна довжина генерації | явно не задано; береться з `generation_config` моделі (значення в репозиторії не знайдено) | `evaluation/evaluate.py:258-266` |
| Захист від повторів (`no_repeat_ngram_size`, compression ratio) | немає | `evaluation/evaluate.py:258-266`; `REVIEW_FINDINGS.md` п. 13.2 (для labelling) |
| Точність інференсу | `autocast(bfloat16)`, `float32_matmul_precision("high")` | `evaluation/evaluate.py:256,575` |
| Batch size | 64 | `config.yaml:64`; `eval_batch_size: 64` у всіх 21 `results.json` |
| DataLoader | `shuffle=False`, `num_workers=0` | `evaluation/evaluate.py:240-247` |
| Макс. тривалість аудіо | 30,0 с (довше — обрізається) | `config.yaml:87`, `fine-tune/fine-tune.py:198-200` |
| Макс. токенів мітки | 448 (текст обрізається до 444) | `config.yaml:88`, `fine-tune/fine-tune.py:208` |
| GPU | рівно одна (`gpus`) | `config.yaml:33-35` |
| Тестові набори | `common_voice`, `fleurs`, `ukr_dialects`, `cv10_test` | `config.yaml:96` |

#### 6.1.7 Дані (тестові набори)

| Набір | Спліт на диску | Кліпів | Джерело числа |
|---|---|---|---|
| `common_voice` (CV 27.0 uk) | `datasets/common-voice/raw/test.tsv` | 10 956 | `results.json → test_sets.common_voice.clips` |
| `fleurs` | `<paths.fleurs>/test` (Arrow) | 750 | те саме |
| `ukr_dialects` | `<paths.ukr_dialects>/data/test-*.parquet` | 3 435 | те саме |
| `cv10_test` | `<paths.cv10_test>/data/train-*.parquet` (так названо в джерелі) | 3 204 | те саме; `fine-tune/dataset_roles.py:15` |
| Разом | — | 18 345 | `results.csv → test_clips` |
| Разом без CV10 | — | 15 141 | `docs/thesis/figures/combined_without_cv10.csv` |

Частки в `combined_weighted` за кліпами: CV 59,7 %, FLEURS 4,1 %, діалекти 18,7 %, CV10 17,5 % (підписи осі `evaluation/<run>/plots/error_rates_by_dataset.png`, `evaluation/evaluate.py:476-484`).

#### 6.1.8 Супутні скрипти

1. `evaluation/results_table.py` — веде `results.csv`: один рядок на прогін, ключ — шлях теки; повторна оцінка замінює рядок; запуск скрипта без аргументів перебудовує всю таблицю з `results.json`/`meta.yaml`/`config.yaml` усіх тек; запис під `fcntl`-блокуванням (`:189-198`). Колонки: `RUN_COLUMNS` (`:42-63`) + `wer_<набір>`, `cer_<набір>` у відсотках з 4 знаками (`:154-157`).
2. `evaluation/evaluate_training.py` — оцінює дві точки навчання: `*-epoch00.ckpt` (кінець 1-ї епохи) і чекпоінт, на який вказує `last.ckpt`; пропускає вже оцінені (`:61-73`, `:86-96`).
3. `evaluation/batch-compare.py` — для теки з кількома прогонами будує `model_comparison.png`: дві панелі (WER, CER), по осі X набори, одна лінія на модель (`:153-189`).

#### 6.1.9 Графіки

| Файл | Опис |
|---|---|
| `evaluation/<run>/plots/error_rates.png` | стовпці WER, CER, MER для `combined_weighted` з підписами у % (`evaluate.py:382-411`) |
| `evaluation/<run>/plots/edit_counts.png` | кількості замін, видалень і вставок для `combined_weighted` (`evaluate.py:413-442`) |
| `evaluation/<run>/plots/error_rates_by_dataset.png` | парні стовпці WER/CER: спершу `combined_weighted` і `combined_mean`, пунктир, далі кожен набір із часткою кліпів у підписі (переглянуто для `student_mix-m345-cv-m5-cv-s1-…`: 28,81/12,01; 26,15/11,62; CV 16,77/4,14 (59,7 %); FLEURS 11,14/3,51 (4,1 %); діалекти 62,24/35,70 (18,7 %); CV10 14,45/3,14 (17,5 %)) |
| `evaluation/<run>/plots/utterance_wer.png` | гістограма WER окремих речень, біни по 10 п.п. від 0 до 200 + окремий стовпець «200+» (`evaluate.py:444-469`); для `student_mix-m345-…-s1` найвищий бін 0–10 % ≈ 7 900 кліпів, помітний пік у біні 100–110 % (≈ 700 кліпів — повністю хибні або порожні гіпотези) |
| `evaluation/baseline/model_comparison.png` | `batch-compare.py` для 4 прогонів у `evaluation/baseline/` (common-voice-only-baseline, original-whisper-medium, whisper-uk-epoch00-01, whisper-uk-epoch01-01): лінії WER і CER за наборами; видно, що діалекти (≈ 70–75 % WER) різко вищі за інші набори, а оригінальний Whisper гірший на CV і CV10 |

![evaluation/baseline/model_comparison.png](../../evaluation/baseline/model_comparison.png)


Ці 4 графіки є в усіх 21 теці прогонів (перевірено `ls evaluation/*/plots`, `evaluation/baseline/*/plots`).

#### 6.1.10 Скрипти для відтворення

```bash
env/bin/python evaluation/evaluate.py                                   # оригінальний model.id
env/bin/python evaluation/evaluate.py --model <ckpt> --note "..."
env/bin/python evaluation/evaluate_training.py --stage fine-tune        # epoch00 + last
env/bin/python evaluation/results_table.py                              # перебудувати results.csv
env/bin/python evaluation/batch-compare.py evaluation/baseline          # model_comparison.png
```


### 6.2 Результати всіх прогонів (`evaluation/results.csv`)

#### 6.2.1 Призначення

Зведена таблиця всіх 21 оцінених моделей з позначкою придатності для дисертації.

#### 6.2.2 Критерії придатності

Взято з `docs/thesis/thesis_plan.md` (розд. 1 і 4) і `todo.md` (п. 1, 3):

1. `bf16-true` при lr 1e-5 — декодер майже не оновлюється; такі моделі не використовувати (`thesis_plan.md` розд. 1 п. 1–2; `REVIEW_FINDINGS.md` п. 4; `todo.md` п. 1). Виняток — вчитель Т0 `baseline/common-voice-only-baseline`, бо від нього стартували студенти.
2. Train CV 73 415 рядків містив увесь dev CV (`REVIEW_FINDINGS.md` п. 1) — dev-метрики таких запусків недійсні; на test не впливає (`thesis_plan.md` розд. 4 п. 1.4).
3. FLEURS або ukr_dialects у train — модель бачила розмітку тестового домену, порівняння на цьому наборі некоректне (`thesis_plan.md` розд. 4 п. 4).
4. CV10 test виключено з підсумку: 311 з 3 204 кліпів мають акустичні дублікати в train CV27 (`thesis_plan.md` розд. 4 п. 3). Для моделей, навчених до фільтра витоків (14.11.2026), WER на CV10 занижений.
5. Запуск на 10 епох — не брати (вибір за тестом), лише згадати (`thesis_plan.md` розд. 3.3, примітка 5).

#### 6.2.3 Таблиця прогонів (WER, %)

Позначення: ПМ — псевдомітки VOA з дозволеними оцінками судді; cv_rows — `cv_train_rows`; Ep — `max_epochs` / індекс епохи чекпоінта (0 = після 1-ї епохи). Усі lr = 1e-5. Тестових кліпів 18 345 у всіх рядках.

| № | Прогін (`run`) | Тип | Старт (init) | Дані навчання | Оцінки | cv_rows | Точність | Ep (max/ckpt) | CV27 | FLEURS | Діалекти | CV10 | comb_w | comb_mean | Статус |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | baseline/original-whisper-medium | pretrained | — | — | — | — | — | — | 20,92 | 11,54 | 73,66 | 18,29 | 34,64 | 31,10 | дійсний (відправна точка) |
| 2 | baseline/common-voice-only-baseline (Т0) | fine-tune | openai/whisper-medium | CV | — | 73 415 | bf16-true | 2/0 | 17,40 | 11,59 | 69,77 | 15,65 | 31,46 | 28,60 | виняток: лише як вчитель Т0 |
| 3 | baseline/whisper-uk-epoch00-01 | fine-tune | — | CV+FLEURS+OpenTTS+TG+ukr_dialects | — | 73 415 | bf16-true | 2/0 | 17,64 | 11,35 | 74,89 | 15,31 | 32,95 | 29,80 | застарілий (bf16-true; FLEURS і діалекти в train) |
| 4 | baseline/whisper-uk-epoch01-01 | fine-tune | — | те саме | — | 73 415 | bf16-true | 2/1 | 17,54 | 11,34 | 73,87 | 15,23 | 32,60 | 29,50 | застарілий (те саме) |
| 5 | student_whisper-commonvoice-based-mark5-epoch01-01 | student | Т0 (`model-whisper-uk-epoch00`) | ПМ | 5 | 73 415 | bf16-true | 2/1 | 17,56 | 11,68 | 61,69 | 15,73 | 29,27 | 26,66 | застарілий (bf16-true) |
| 6 | student_whisper-commonvoice-based-epoch01-01 | student | Т0 | ПМ | 0–5 | 73 415 | bf16-true | 2/1 | 18,02 | 11,75 | 74,25 | 15,91 | 33,08 | 29,98 | застарілий (bf16-true) |
| 7 | student_whisper-commonvoice-based-epoch01-02 | student | Т0 | ПМ | 3 | 73 415 | bf16-true | 2/1 | 17,70 | 11,94 | 65,19 | 15,96 | 30,38 | не знайдено (старий формат `combined`) | застарілий (bf16-true) |
| 8 | student_mix-m2345-cv-m45-cv-s1-… (крок 1) | student | Т0 | ПМ + CV | 2–5 | 62 460 | bf16-mixed | 1/0 | 15,50 | 11,61 | 70,00 | 12,75 | 30,19 | 27,46 | дійсний (табл. 1 тез) |
| 9 | student_mix-m2345-cv-m45-cv-s2-… (крок 2) | student | прогін 8 | ПМ + CV | 4–5 | 62 460 | bf16-mixed | 1/0 | 14,59 | 11,61 | 71,04 | 11,69 | 29,89 | 27,23 | дійсний |
| 10 | student_mix-m345-cv-m5-cv-s1-… (крок 1) | student | Т0 | ПМ + CV | 3–5 | 62 460 | bf16-mixed | 1/0 | 16,77 | 11,14 | 62,24 | 14,45 | 28,81 | 26,15 | дійсний (найкращий студент) |
| 11 | student_mix-m345-cv-m5-cv-s2-… (крок 2) | student | прогін 10 | ПМ + CV | 5 | 62 460 | bf16-mixed | 1/0 | 14,27 | 11,21 | 78,07 | 11,29 | 31,63 | 28,71 | дійсний |
| 12 | whisper-uk-cv-fleurs-epoch00-01 | fine-tune | openai/whisper-medium | CV+FLEURS | — | 62 460 | bf16-mixed | 2/0 | 13,15 | 10,58 | 74,69 | 10,52 | 29,96 | 27,23 | не для порівняння (FLEURS у train) |
| 13 | whisper-uk-cv-fleurs-epoch01-01 | fine-tune | те саме | CV+FLEURS | — | 62 460 | bf16-mixed | 2/1 | 11,60 | 10,37 | 68,01 | 7,90 | 26,93 | 24,47 | не для порівняння (FLEURS у train) |
| 14 | whisper-uk-cv-fixed-epoch00-01 | fine-tune | те саме | CV | — | 62 460 | bf16-mixed | 2/0 | 12,94 | 12,95 | 73,35 | 9,44 | 29,54 | 27,17 | допоміжний (до фільтра витоків) |
| 15 | whisper-uk-cv-fixed-epoch01-01 | fine-tune | те саме | CV | — | 62 460 | bf16-mixed | 2/1 | 11,66 | 12,78 | 69,85 | 7,78 | 27,68 | 25,52 | допоміжний (до фільтра витоків) |
| 16 | whisper-m-fixed-leaks-5-epochs-epoch00-01 | fine-tune | те саме | CV | — | 62 460 | bf16-mixed | 5/0 | 13,32 | 13,29 | 79,00 | 10,20 | 31,46 | 28,95 | проміжна точка орієнтира |
| 17 | whisper-m-fixed-leaks-5-epochs-epoch04-01 | fine-tune | те саме | CV | — | 62 460 | bf16-mixed | 5/4 | 10,68 | 12,98 | 66,75 | 7,35 | 26,28 | 24,44 | дійсний (орієнтир «CV-only 5 епох») |
| 18 | whisper-m-fixed-leaks-5-epochs-fleurs-epoch00-01 | fine-tune | те саме | CV+FLEURS | — | 62 460 | bf16-mixed | 5/0 | 13,55 | 11,00 | 73,43 | 10,10 | 29,78 | 27,02 | не для порівняння (FLEURS у train) |
| 19 | whisper-m-fixed-leaks-5-epochs-fleurs-epoch04-01 | fine-tune | те саме | CV+FLEURS | — | 62 460 | bf16-mixed | 5/4 | 10,67 | 10,35 | 67,25 | 7,20 | 26,16 | 23,87 | не для порівняння (FLEURS у train); «верхня межа з розміткою в домені» |
| 20 | whisper-m-fixed-leaks-10-epochs-epoch00-01 | fine-tune | те саме | CV | — | 62 460 | bf16-mixed | 10/0 | 13,10 | 13,49 | 82,72 | 10,27 | 32,43 | 29,89 | проміжна точка |
| 21 | whisper-m-fixed-leaks-10-epochs-epoch09-01 | fine-tune | те саме | CV | — | 62 460 | bf16-mixed | 10/9 | 10,29 | 13,57 | 63,52 | 6,89 | 25,16 | 23,57 | лише згадати (вибір за тестом); див. примітку 4 |

Джерело чисел: `evaluation/results.csv` (колонки `wer_*`, `init`, `train_data`, `accepted_marks`, `cv_train_rows`, `precision`, `max_epochs`, `epoch`). Поле `init` для fine-tune у таблиці порожнє; «openai/whisper-medium» для них узято з `config.yaml:26` і `thesis_plan.md` (стовпець «Старт» табл. 1). «Т0» для студентів — `init = /mnt/data/rberehun/checkpoints/model-whisper-uk/model-whisper-uk-epoch00.ckpt`.

Примітки:

1. Т0 у `results.csv` має `max_epochs = 2`, `epoch = 0`, тобто це чекпоінт після 1-ї епохи 2-епохового запуску; `note`: «Бейзлан на CV 27.0 (витік dev у train)».
2. Прогони 1–7 оцінено 26.10–06.11.2026, прогони 8–21 — 07.11–17.11.2026 (`evaluated_at`).
3. Фільтр витоків `fine-tune/leak_filter.py` датований у `thesis_plan.md` 14.11.2026; отже прогони 16–21 (оцінені 15–17.11) навчено вже з фільтром, прогони 1–15 — без нього. Логів «leak filter … removed» у `output/` і `evaluation/` не знайдено, тож це висновок за датами.
4. `config.yaml:99` вказує `pseudo_labelling.teacher_model` на `model-whisper-m-fixed-leaks-10-epochs-epoch05.ckpt`; оцінки саме цього чекпоінта (епоха 5) у `results.csv` немає.

#### 6.2.4 CER, %

| № | Прогін | CV27 | FLEURS | Діалекти | CV10 | comb_w | comb_mean |
|---|---|---|---|---|---|---|---|
| 1 | original-whisper-medium | 5,91 | 3,70 | 46,66 | 4,69 | 15,94 | 15,24 |
| 2 | Т0 common-voice-only-baseline | 4,36 | 3,69 | 37,74 | 3,58 | 12,72 | 12,34 |
| 3 | whisper-uk-epoch00-01 | 4,37 | 3,33 | 42,50 | 3,41 | 13,89 | 13,40 |
| 4 | whisper-uk-epoch01-01 | 4,35 | 3,37 | 42,56 | 3,40 | 13,89 | 13,42 |
| 5 | student…mark5 | 4,38 | 3,75 | 35,40 | 3,57 | 12,14 | 11,77 |
| 6 | student…epoch01-01 (0–5) | 4,49 | 3,77 | 41,56 | 3,60 | 13,78 | 13,35 |
| 7 | student…epoch01-02 (3) | 4,37 | 3,83 | 36,86 | 3,61 | 12,52 | не знайдено |
| 8 | mix m2345 s1 | 3,94 | 4,05 | 41,02 | 2,86 | 13,28 | 12,97 |
| 9 | mix m2345 s2 (m45) | 3,75 | 4,23 | 41,80 | 2,70 | 13,39 | 13,12 |
| 10 | mix m345 s1 | 4,14 | 3,51 | 35,70 | 3,14 | 12,01 | 11,62 |
| 11 | mix m345 s2 (m5) | 3,63 | 4,14 | 45,60 | 2,58 | 14,28 | 13,99 |
| 12 | cv-fleurs ep0 | 3,39 | 2,97 | 40,22 | 2,49 | 12,64 | 12,27 |
| 13 | cv-fleurs ep1 | 3,03 | 2,97 | 37,93 | 1,89 | 11,79 | 11,46 |
| 14 | cv-fixed ep0 | 3,30 | 5,15 | 40,67 | 2,16 | 12,89 | 12,82 |
| 15 | cv-fixed ep1 | 3,04 | 5,23 | 39,96 | 1,84 | 12,54 | 12,52 |
| 16 | fixed-leaks-5 ep0 | 3,43 | 4,95 | 44,72 | 2,33 | 13,99 | 13,86 |
| 17 | fixed-leaks-5 ep4 | 2,84 | 5,13 | 38,62 | 1,77 | 12,08 | 12,09 |
| 18 | fixed-leaks-5+fleurs ep0 | 3,42 | 3,17 | 40,29 | 2,28 | 12,67 | 12,29 |
| 19 | fixed-leaks-5+fleurs ep4 | 2,86 | 3,19 | 39,30 | 1,76 | 12,06 | 11,78 |
| 20 | fixed-leaks-10 ep0 | 3,35 | 5,02 | 46,84 | 2,35 | 14,51 | 14,39 |
| 21 | fixed-leaks-10 ep9 | 2,74 | 6,39 | 36,43 | 1,70 | 11,58 | 11,81 |

Джерело: `evaluation/results.csv`, колонки `cer_*`.

#### 6.2.5 Підсумок без CV10 («Разом», 15 141 кліп)

| Прогін | WER | CER |
|---|---|---|
| original-whisper-medium | 37,34 | 17,95 |
| Т0 common-voice-only-baseline | 34,06 | 14,36 |
| mix m345 s1 (ПМ 3–5 + CV) | 31,18 | 13,60 |
| mix m345 s2 (ПМ 5 + CV) | 34,98 | 16,37 |
| mix m2345 s1 (ПМ 2–5 + CV) | 33,06 | 15,15 |
| mix m2345 s2 (ПМ 4–5 + CV) | 32,89 | 15,30 |
| whisper-uk-cv-fixed-epoch00-01 | 32,85 | 14,81 |
| fixed-leaks-5 ep0 | 34,96 | 16,08 |
| fixed-leaks-5 ep4 (CV-only 5 епох) | 29,40 | 13,92 |
| fixed-leaks-10 ep9 | 28,17 | 13,35 |

Джерело: `docs/thesis/figures/combined_without_cv10.csv` (округлено до 0,01).

##### Скрипт `docs/thesis/figures/combined_without_cv10.py`

1. Читає `evaluation/<run>/predictions.csv` для 10 фіксованих прогонів (`:21-32`).
2. Повторює `normalize_text` і `corpus_metrics` (порожні еталони відкидаються, `jiwer.wer`/`jiwer.cer` по всіх парах; `:40-53`).
3. Контроль: перерахунок з CV10 має збігтися з `wer/cer_combined_weighted` у `results.csv` з допуском 0,01 п.п., інакше зупинка (`:60-63`).
4. Рахує ті самі метрики лише для `common_voice`, `fleurs`, `ukr_dialects` і пише CSV (`:64-69`).

#### 6.2.6 Графіки результатів для тез

| Файл | Опис |
|---|---|
| `docs/thesis/figures/figure1.png` | групова стовпчикова діаграма, панелі (а) WER і (б) CER; осі X: CV27, FLEURS, Діалекти, «Разом» (без CV10, відокремлено пунктиром); 4 моделі: Whisper Medium (штрихування), Вчитель Т0, Студент ПМ 3–5, Студент ПМ 5; Y від 0, межі 80 і 50; 300 dpi, 17 × 7,2 см, Times New Roman (`make_figure1.py`) |
| `docs/thesis/figures/figure2.png` | те саме з 7 моделями: Whisper Medium, Т0 і CV-only 5 епох заштриховані; студенти ПМ 5, ПМ 4–5, ПМ 2–5, ПМ 3–5 (`make_figure2.py`) |

![docs/thesis/figures/figure1.png](../../docs/thesis/figures/figure1.png)

![docs/thesis/figures/figure2.png](../../docs/thesis/figures/figure2.png)


Скрипти: `docs/thesis/figures/make_figure1.py`, `make_figure2.py` — читають `evaluation/results.csv` і `combined_without_cv10.csv`, `math.ceil(max/крок)·крок` для верхньої межі (крок WER 20, CER 10).

#### 6.2.7 Скрипти для відтворення

```bash
env/bin/python evaluation/results_table.py
env/bin/python docs/thesis/figures/combined_without_cv10.py
env/bin/python docs/thesis/figures/make_figure1.py
env/bin/python docs/thesis/figures/make_figure2.py
```

Увага: після переміщення в `docs/thesis/figures/` у цих трьох скриптах `ROOT = Path(__file__).resolve().parents[2]` вказує на `docs/`, а не на корінь репозиторію, тому шлях `ROOT / "evaluation" / "results.csv"` не існує (`combined_without_cv10.py:18`, `make_figure1.py:16`, `make_figure2.py:16`). Для запуску треба `parents[3]`. Docstring-и також досі згадують `docs/figures/`.


### 6.3 Аналіз помилок

#### 6.3.1 Призначення

Розкласти WER на заміни (S), видалення (D) і вставки (I) та описати розподіл WER окремих речень.

#### 6.3.2 Принцип роботи

1. S/D/I/H зберігає `evaluate.py` у `results.json → test_sets.<набір>` (із `jiwer.process_words`).
2. WER на рівні речення рахується в `write_plots` для гістограми (`evaluate.py:444-449`); числові статистики нижче обчислено окремо з `predictions.csv` (`docs/diploma/materials_scripts/utt.py`, пословний Левенштейн, та сама нормалізація).

#### 6.3.3 Результати: S/D/I на `combined_weighted` (18 345 кліпів)

| Прогін | S | D | I | H | WER |
|---|---|---|---|---|---|
| original-whisper-medium | 36 163 | 10 581 | 9 249 | 114 894 | 34,64 |
| Т0 common-voice-only-baseline | 33 446 | 7 853 | 9 548 | 120 339 | 31,46 |
| whisper-uk-epoch00-01 | 33 332 | 5 221 | 14 711 | 123 085 | 32,95 |
| whisper-uk-epoch01-01 | 33 069 | 5 396 | 14 237 | 123 173 | 32,60 |
| student…mark5 | 33 088 | 8 423 | 5 794 | 120 127 | 29,27 |
| student…epoch01-01 (0–5) | 33 544 | 8 191 | 11 730 | 119 903 | 33,08 |
| student…epoch01-02 (3) | 33 438 | 8 231 | 7 442 | 119 969 | 30,38 |
| mix m2345 s1 | 32 149 | 7 420 | 9 225 | 122 069 | 30,19 |
| mix m2345 s2 | 31 102 | 7 440 | 9 771 | 123 096 | 29,89 |
| mix m345 s1 | 32 872 | 7 763 | 5 937 | 121 003 | 28,81 |
| mix m345 s2 | 30 793 | 7 351 | 12 984 | 123 494 | 31,63 |
| cv-fleurs ep0 | 30 264 | 6 254 | 11 912 | 125 120 | 29,96 |
| cv-fleurs ep1 | 28 604 | 7 011 | 7 914 | 126 023 | 26,93 |
| cv-fixed ep0 | 30 226 | 7 078 | 10 441 | 124 334 | 29,54 |
| cv-fixed ep1 | 28 913 | 7 013 | 8 812 | 125 712 | 27,68 |
| fixed-leaks-5 ep0 | 30 694 | 6 758 | 13 400 | 124 186 | 31,46 |
| fixed-leaks-5 ep4 | 27 700 | 7 406 | 7 368 | 126 532 | 26,28 |
| fixed-leaks-5+fleurs ep0 | 30 445 | 6 807 | 10 886 | 124 386 | 29,78 |
| fixed-leaks-5+fleurs ep4 | 27 539 | 7 339 | 7 406 | 126 760 | 26,16 |
| fixed-leaks-10 ep0 | 30 857 | 6 790 | 14 774 | 123 991 | 32,43 |
| fixed-leaks-10 ep9 | 27 380 | 7 276 | 6 012 | 126 982 | 25,16 |

Джерело: `evaluation/<run>/results.json → test_sets.combined_weighted` (для прогону 7 — `combined`). N = S + D + H = 161 388 слів еталона (наприклад, 33 446 + 7 853 + 120 339; обчислено).

#### 6.3.4 S/D/I на `ukr_dialects` (3 435 кліпів) для дійсних моделей

| Прогін | S | D | I | WER |
|---|---|---|---|---|
| original-whisper-medium | 18 614 | 8 081 | 6 993 | 73,66 |
| Т0 | 18 758 | 6 015 | 7 137 | 69,77 |
| mix m345 s1 (ПМ 3–5) | 18 529 | 5 953 | 3 985 | 62,24 |
| mix m345 s2 (ПМ 5) | 18 689 | 5 806 | 11 213 | 78,07 |
| mix m2345 s1 (ПМ 2–5) | 18 897 | 5 738 | 7 381 | 70,00 |
| mix m2345 s2 (ПМ 4–5) | 18 637 | 5 856 | 7 999 | 71,04 |
| fixed-leaks-5 ep4 (CV-only) | 18 338 | 6 166 | 6 025 | 66,75 |
| fixed-leaks-10 ep9 | 18 232 | 6 053 | 4 768 | 63,52 |

Джерело: `results.json → test_sets.ukr_dialects`. Спостереження: заміни на діалектах майже сталі (18,2–18,9 тис.), а різниця між моделями визначається головно вставками: студент ПМ 5 має 11 213 вставок проти 3 985 у студента ПМ 3–5 (обчислено з таблиці). Те саме видно на кінці 1-ї епохи fine-tune: fixed-leaks-10 ep0 — 13 145 вставок на діалектах, ep9 — 4 768.

#### 6.3.5 Розподіл WER на рівні речень (обчислено, `docs/diploma/materials_scripts/utt.py`)

| Прогін | Медіана, % | Середнє, % | Речень з WER = 0 | Речень з WER ≥ 100 % | Речень з WER > 200 % |
|---|---|---|---|---|---|
| original-whisper-medium | 20,0 | 33,2 | 30,9 % | 6,6 % | 46 |
| Т0 | 14,3 | 33,9 | 37,3 % | 5,8 % | 41 |
| mix m345 s1 (ПМ 3–5) | 14,3 | 26,9 | 37,9 % | 5,5 % | 21 |
| mix m345 s2 (ПМ 5) | 10,5 | 31,1 | 43,0 % | 5,5 % | 68 |
| mix m2345 s1 (ПМ 2–5) | 12,5 | 29,4 | 40,2 % | 5,7 % | 51 |
| mix m2345 s2 (ПМ 4–5) | 11,1 | 27,2 | 42,1 % | 5,5 % | 55 |
| fixed-leaks-5 ep4 (CV-only) | 0,0 | 22,5 | 51,1 % | 4,6 % | 35 |
| fixed-leaks-10 ep9 | 0,0 | 21,1 | 52,0 % | 4,4 % | 23 |

По наборах (медіана WER речення / частка WER = 0 / частка ≥ 100 %):

| Прогін | CV27 | FLEURS | Діалекти | CV10 |
|---|---|---|---|---|
| original-whisper-medium | 16,7 / 36,5 % / 3,3 % | 8,9 / 24,0 % / 0,1 % | 60,0 / 5,6 % / 21,2 % | 12,5 / 40,6 % / 3,7 % |
| Т0 | 11,1 / 44,1 % / 2,3 % | 8,6 / 22,3 % / 0,0 % | 53,3 / 7,5 % / 21,2 % | 7,7 / 49,4 % / 2,8 % |
| mix m345 s1 | 10,0 / 44,8 % / 2,2 % | 8,3 / 22,8 % / 0,0 % | 50,0 / 8,3 % / 20,6 % | 7,1 / 49,5 % / 2,2 % |
| mix m345 s2 | 0,0 / 50,8 % / 1,8 % | 8,3 / 24,1 % / 0,0 % | 50,0 / 8,1 % / 22,0 % | 0,0 / 58,1 % / 1,9 % |
| mix m2345 s1 | 8,3 / 47,6 % / 2,0 % | 8,7 / 23,2 % / 0,0 % | 50,0 / 7,8 % / 21,7 % | 0,0 / 53,7 % / 2,4 % |
| mix m2345 s2 | 7,1 / 49,8 % / 1,8 % | 8,7 / 23,2 % / 0,0 % | 50,0 / 7,9 % / 21,6 % | 0,0 / 56,9 % / 2,1 % |
| fixed-leaks-5 ep4 | 0,0 / 61,2 % / 1,2 % | 10,3 / 18,9 % / 0,0 % | 50,0 / 8,7 % / 19,9 % | 0,0 / 69,9 % / 1,2 % |
| fixed-leaks-10 ep9 | 0,0 / 62,1 % / 1,2 % | 10,5 / 18,7 % / 0,1 % | 50,0 / 9,0 % / 19,0 % | 0,0 / 71,7 % / 1,1 % |

Інтерпретація для тексту: близько п'ятої частини речень діалектного набору мають WER ≥ 100 % у всіх моделях; хвіст > 200 % (зациклені вставки) найбільший у студента ПМ 5 (68 речень) і найменший у студента ПМ 3–5 (21), що узгоджується з кількістю вставок у п. 3.4.

#### 6.3.6 Графіки

1. `evaluation/<run>/plots/edit_counts.png` — S/D/I для кожного прогону.
2. `evaluation/<run>/plots/utterance_wer.png` — гістограма WER речень.

#### 6.3.7 Скрипти для відтворення

1. S/D/I: `evaluation/evaluate.py` (поля в `results.json`).
2. Статистики речень: власний скрипт за `predictions.csv` (у репозиторії відповідника немає; гістограму будує `evaluate.py:write_plots`).


### 6.4 Виявлення витоків даних (`datasets/leak-clean/`)

#### 6.4.1 Призначення

Перевірити, чи не потрапили кліпи, тексти або диктори dev/test у train (всередині датасетів і між ними), і сформувати список train-кліпів, які обов'язково вилучаються перед навчанням.

#### 6.4.2 Рівні повтору (з `datasets/leak-clean/README.md`, розд. 1)

1. Той самий файл (ім'я аудіо).
2. Ті самі байти (SHA-256 закодованого файлу).
3. Той самий звук після перекодування (акустичний збіг).
4. Той самий нормалізований текст.
5. Той самий диктор (лише Common Voice).

Рівні 1–3 — жорсткий витік (прибирати обов'язково), 4–5 — м'який.

#### 6.4.3 Принцип роботи (кроками)

1. `collect_metadata.py` — один рядок на кліп усіх датасетів (`common_voice`, `fleurs`, `opentts`, `tg_voices`, `ukr_dialects`, `cv10_test`, `voa`) у `metadata.parquet`: датасет, спліт, ID, ім'я аудіо, SHA-256 закодованих байтів (`leak_common.py:74-78`), місце, текст і нормалізований текст. Аудіо не декодується. Хеш не рахується для VOA (слайси довгих файлів) і `common_voice/unlabelled`. Для VOA текстом є псевдомітка вчителя. Також пише `coverage.csv` (аудіо без метаданих тощо).
2. `find_overlaps.py` — групує кліпи за нормалізованим текстом (`leak_common.normalize_text`: WER-нормалізація + уніфікація апострофів `’ʼ‘`` ` `` → `'` і вилучення апострофа поза словом, `leak_common.py:58-71`), класифікує повтор як `within_split`, `between_splits`, `between_datasets`; пише `duplicates.csv`, `units.csv`. Опція `--min-words` (звіт показує і всі тексти, і тексти ≥ 4 слів, `plot_overlaps.py:39`).
3. `find_audio_matches.py` — групує train/dev/test окремо за двома ключами: `name` (ім'я файлу) і `sha256`. Ім'я, що повторюється всередині одного спліту, ігнорується. Збіг за іменем між датасетами зараховується лише якщо схожість нормалізованих текстів за `difflib.SequenceMatcher.ratio` ≥ 0,6 або байти однакові (`find_audio_matches.py:43-45,106-110`). Поле `same_bytes` = yes/no/unknown. Пише `audio_matches.csv`.
4. `plot_overlaps.py` — графіки `overlap_matrix.png`, `eval_leaks.png`, `coverage.png`.
5. `check_leaks.py` — запускає кроки 1–4 і пише `output/leak-clean/report.md`.
6. Ручна перевірка: `review_leaks.ipynb` (генератор `make_review_notebook.py`) — прослуховування випадкових позначених пар.
7. `review_sha.ipynb` (`make_sha_notebook.py`) — пари eval/train з однаковим SHA-256 → `sha_matches.csv`.
8. `review_acoustic.ipynb` (`make_acoustic_notebook.py`) — пошук перекодованих копій → `acoustic_matches.csv` (п. 4.4).
9. `fine-tune/leak_filter.py` перед кожним навчанням читає `sha_matches.csv` і `acoustic_matches.csv` і вилучає з train (`common_voice`, `ukr_dialects`) кліпи з колонок `dataset_train`/`split_train`/`id_train`; якщо котрогось CSV немає, навчання зупиняється (`fine-tune/leak_filter.py:1-16,25-27`, `fine-tune/speech_data.py:427-434`). Dev і test не змінюються.

#### 6.4.4 Акустичний метод (`make_acoustic_notebook.py` → `review_acoustic.ipynb`)

Це не Chromaprint і не нейронні ембединги, а пряма кореляція log-mel спектрограм.

1. Кандидати: пари «eval-кліп (`cv10_test` dev/test) — train-кліп (`common_voice` train)» з однаковим нормалізованим текстом і/або однаковим ім'ям файлу (`CANDIDATE_KEYS = ("text", "name")`); пара, знайдена обома ключами, лишається один раз з `key = "name+text"`.
2. Обидва кліпи декодуються `ffmpeg` у 16 кГц mono s16le.
3. Пари з різницею тривалостей > `MAX_DURATION_DIFF` відкидаються.
4. Log-mel: вікно Ганна 400 відліків (25 мс), крок 160 (10 мс), 40 трикутних mel-фільтрів 50 Гц – 8 кГц, `log(power·mel + 1e-10)`.
5. Від кожного mel-біна віднімається його середнє (лишається часовий патерн, а не середній спектр).
6. `score` — максимальна кореляція Пірсона двох матриць по всіх зсувах у межах ±`MAX_SHIFT` (кодеки додають тишу).
7. Пари сортуються за `score` для прослуховування; поріг підбирається на слух; пари зі `score ≥ THRESHOLD` пишуться в `acoustic_matches.csv` (лише якщо `MAX_PAIRS = None`, тобто оцінено всі кандидати).

| Параметр | Значення | Джерело |
|---|---|---|
| `EVAL_DATASETS` / `TRAIN_DATASETS` | `["cv10_test"]` / `["common_voice"]` | `make_acoustic_notebook.py:92-95` |
| `MIN_WORDS` | 1 | `make_acoustic_notebook.py:97` |
| `MAX_PAIRS` | 2000 (вибірка); `None` для запису CSV | `make_acoustic_notebook.py:98,296` |
| `MAX_DURATION_DIFF` | 0,5 с | `make_acoustic_notebook.py:99` |
| `MAX_SHIFT` | 0,3 с (= 30 кадрів) | `make_acoustic_notebook.py:100,225` |
| `THRESHOLD` у коді | 0,8 | `make_acoustic_notebook.py:101` |
| Рекомендований поріг | між 0,93 і 1,0, напр. 0,97 | `datasets/leak-clean/README.md:197-199` |
| `SAMPLE_RATE`, `N_FFT`, `HOP`, `N_MELS` | 16000, 400, 160, 40 | `make_acoustic_notebook.py:150-153` |
| `SEED` | 0 | `make_acoustic_notebook.py:102` |

#### 6.4.5 Дані (кількість кліпів з текстом, `output/leak-clean/units.csv`)

| Спліт | Кліпів |
|---|---|
| common_voice/train | 62 460 |
| common_voice/dev | 10 955 |
| common_voice/test | 10 956 |
| fleurs/train / dev / test | 2 810 / 325 / 750 |
| opentts/train | 23 983 |
| tg_voices/train | 12 069 |
| ukr_dialects/train / dev / test | 27 520 / 3 347 / 3 435 |
| cv10_test/test | 3 204 |
| voa/train | 351 274 |
| Разом | 513 088 (`report.md`, «Головне» п. 1) |

#### 6.4.6 Результати

Звіт `output/leak-clean/report.md` створено 2026-11-08 09:07, коміт `d6296e4`.

Текстові збіги dev/test з будь-яким train (`report.md`, розд. 3):

| Спліт | Кліпів | Усі тексти | Тексти ≥ 4 слів |
|---|---|---|---|
| common_voice/dev | 10 955 | 3 431 (31,3 %) | 2 996 (27,3 %) |
| common_voice/test | 10 956 | 5 010 (45,7 %) | 4 414 (40,3 %) |
| fleurs/dev | 325 | 0 | 0 |
| fleurs/test | 750 | 0 | 0 |
| ukr_dialects/dev | 3 347 | 175 (5,2 %) | 140 (4,2 %) |
| ukr_dialects/test | 3 435 | 153 (4,5 %) | 126 (3,7 %) |
| cv10_test/test | 3 204 | 2 492 (77,8 %) | 2 178 (68,0 %) |

Збіги аудіо dev/test з train (`report.md`, розд. 2):

| Спліт | За іменем | Із них ті самі байти | За SHA-256 | Разом |
|---|---|---|---|---|
| common_voice/dev, test | 0 | 0 | 0 | 0 |
| fleurs/dev, test | 0 | 0 | 0 | 0 |
| ukr_dialects/dev | 5 | 5 | 6 | 6 (0,2 %) |
| ukr_dialects/test | 1 | 1 | 1 | 1 |
| cv10_test/test | 2 597 | 0 | 0 | 2 597 (81,1 %), з `ukr_dialects/train` |

Повтори текстів (`report.md`, розд. 4): між датасетами 9 894 тексти / 44 625 кліпів; між сплітами одного датасету 6 409 / 23 112; всередині спліту 35 334 / 89 606.

Остаточні списки для фільтра:

| Файл | Рядків | Зміст | Джерело |
|---|---|---|---|
| `output/leak-clean/sha_matches.csv` | 7 | `ukr_dialects` dev (6) і test (1) ↔ `ukr_dialects/train`, ті самі SHA-256, той самий текст | файл |
| `output/leak-clean/acoustic_matches.csv` | 311 | `cv10_test/test` ↔ `common_voice/train`, `key = text`, `score = 1.0` у всіх, `shift_ms = 0`, `duration_diff = 0`, `same_sha = False`; 311 унікальних eval- і 311 train-кліпів; середній розмір файлу 167 515 Б (cv10) проти 39 019 Б (CV27) | файл; статистика обчислена |
| `output/leak-clean/review_verdicts_sha256.csv` | 7 вердиктів `leak` | ручна перевірка SHA-збігів | файл |
| `output/leak-clean/review_verdicts_name.csv` | 6 вердиктів `leak` | ручна перевірка збігів за іменем (усі — `ukr_dialects`) | файл |

Ручне прослуховування (`datasets/leak-clean/README.md`, розд. 6):

1. Збіг за SHA-256 — справжній витік.
2. Збіг лише за іменем — у майже всіх перевірених випадках не витік (різні аудіо й тексти).
3. Однакові тексти Common Voice начитані різними дикторами — м'який витік.
4. З 40 прослуханих пар `cv10_test`–`common_voice` з однаковим текстом 6 (~15 %) звучать ідентично при різних байтах (~200 КБ проти ~50 КБ).
5. Акустичний пошук дав 311 збігів зі `score` 1,0; наступні пари мають `score` ≤ 0,93 і є різними записами.

Висновок: 311 / 3 204 = 9,7 % кліпів CV10 test мають копію в train CV27 (`thesis_plan.md` розд. 4 п. 3), тому CV10 вилучено з підсумкової оцінки в тезах.

Common Voice: побудова `train_validated` і `test_cleaned` (`datasets/common-voice/README.md`, `clean_common_voice_train.py`):

1. `validated.tsv` (усі записи, що пройшли голосування) мінус кліпи `test.tsv` → 73 415 рядків (стара версія train, у якій був увесь dev — 10 955 кліпів).
2. `clean_common_voice_train.py` вилучає train-рядки, `path` яких є в `dev.tsv` або `test.tsv` (`:103-105`) → 62 460 рядків, 376 дикторів. Рядки копіюються побайтово, повторний запуск нічого не змінює.
3. Спільні з dev/test речення й диктори свідомо лишено. Повне розділення дало б 38 849 рядків і 74 дикторів.
4. `test_cleaned.tsv` — рядки test, у яких ні `client_id`, ні нормалізоване речення не трапляються в train (`:113-115`): 5 703 кліпи, 503 диктори. У навчанні й `evaluate.py` не використовується.
5. Перетин test із train: 3 932 кліпи за диктором (36 %), 5 001 за нормалізованим реченням (README; 4 992 за дослівним реченням — `REVIEW_FINDINGS.md` п. 2).

| Файл `datasets/common-voice/raw/` | Роль | Рядків | Дикторів |
|---|---|---|---|
| `train_validated.tsv` | train | 62 460 | 376 |
| `dev.tsv` | dev | 10 955 | 140 |
| `test.tsv` | test | 10 956 | 938 |
| `test_cleaned.tsv` | закрита перевірка | 5 703 | 503 |
| `unlabelled.tsv` (= `other.tsv`) | не використовується | 9 787 | 696 |

Джерело: `datasets/common-voice/README.md`.

##### Результати на `test_cleaned` (README, 2026-11-08)

| Прогін | Навчання | test | test_cleaned | різниця | поправка (різниця − (−0,31)) |
|---|---|---|---|---|---|
| original-whisper-medium | — | 20,92 | 20,61 | −0,31 | 0 |
| common-voice-only-baseline | bf16-true, 73 415 | 17,40 | 17,39 | −0,01 | +0,30 |
| whisper-uk-epoch00-01 | bf16-true | 17,64 | 17,37 | −0,27 | +0,04 |
| whisper-uk-epoch01-01 | bf16-true | 17,54 | 17,26 | −0,28 | +0,03 |
| student…epoch01-01 (0–5) | bf16-true | 18,02 | 17,88 | −0,14 | +0,17 |
| student…epoch01-02 (3) | bf16-true | 17,70 | 17,86 | +0,16 | +0,47 |
| student…mark5 | bf16-true | 17,56 | 17,67 | +0,12 | +0,43 |
| student_mix-m2345 s1 | bf16-mixed, 62 460 | 15,50 | 15,76 | +0,26 | +0,57 |
| student_mix-m2345 s2 | bf16-mixed, 62 460 | 14,59 | 15,13 | +0,54 | +0,85 |

##### Перерахунок на `test_cleaned` для всіх 21 прогону (обчислено, `docs/diploma/materials_scripts/clean.py`)

Метод той самий, що в `score_common_voice_cleaned.py`: кліпи `test_cleaned.tsv` (5 703) вибираються з рядків `common_voice` у `predictions.csv`, WER/CER корпусні після `normalize_text`. Значення для 9 прогонів з README відтворено точно (наприклад, 20,92 → 20,61; 14,59 → 15,13), що підтверджує коректність перерахунку. «Поправка» = різниця − (−0,31).

| Прогін | WER test | WER cleaned | різниця | поправка | CER test | CER cleaned | різниця CER |
|---|---|---|---|---|---|---|---|
| original-whisper-medium | 20,92 | 20,61 | −0,31 | 0 | 5,91 | 6,12 | +0,21 |
| Т0 common-voice-only-baseline | 17,40 | 17,39 | −0,01 | +0,30 | 4,36 | 4,69 | +0,33 |
| whisper-uk-epoch00-01 | 17,64 | 17,37 | −0,27 | +0,04 | 4,37 | 4,61 | +0,24 |
| whisper-uk-epoch01-01 | 17,54 | 17,26 | −0,28 | +0,03 | 4,35 | 4,58 | +0,24 |
| student…epoch01-01 (0–5) | 18,02 | 17,88 | −0,14 | +0,17 | 4,49 | 4,76 | +0,28 |
| student…epoch01-02 (3) | 17,70 | 17,86 | +0,16 | +0,47 | 4,37 | 4,76 | +0,38 |
| student…mark5 | 17,56 | 17,67 | +0,12 | +0,43 | 4,38 | 4,75 | +0,37 |
| mix m2345 s1 (ПМ 2–5 + CV) | 15,50 | 15,76 | +0,26 | +0,57 | 3,94 | 4,28 | +0,34 |
| mix m2345 s2 (ПМ 4–5 + CV) | 14,59 | 15,13 | +0,54 | +0,85 | 3,75 | 4,15 | +0,40 |
| mix m345 s1 (ПМ 3–5 + CV) | 16,77 | 16,58 | −0,19 | +0,12 | 4,14 | 4,37 | +0,22 |
| mix m345 s2 (ПМ 5 + CV) | 14,27 | 14,81 | +0,55 | +0,86 | 3,63 | 4,02 | +0,39 |
| whisper-uk-cv-fleurs ep0 | 13,15 | 14,59 | +1,45 | +1,76 | 3,39 | 3,99 | +0,60 |
| whisper-uk-cv-fleurs ep1 | 11,60 | 13,33 | +1,73 | +2,04 | 3,03 | 3,63 | +0,59 |
| whisper-uk-cv-fixed ep0 | 12,94 | 14,26 | +1,32 | +1,63 | 3,30 | 3,83 | +0,54 |
| whisper-uk-cv-fixed ep1 | 11,66 | 13,55 | +1,89 | +2,20 | 3,04 | 3,68 | +0,64 |
| fixed-leaks-5 ep0 | 13,32 | 14,80 | +1,48 | +1,79 | 3,43 | 4,05 | +0,62 |
| fixed-leaks-5 ep4 (CV-only 5 епох) | 10,68 | 13,11 | +2,43 | +2,74 | 2,84 | 3,56 | +0,72 |
| fixed-leaks-5+fleurs ep0 | 13,55 | 15,06 | +1,51 | +1,82 | 3,42 | 4,02 | +0,60 |
| fixed-leaks-5+fleurs ep4 | 10,67 | 13,08 | +2,41 | +2,72 | 2,86 | 3,63 | +0,78 |
| fixed-leaks-10 ep0 | 13,10 | 14,38 | +1,28 | +1,59 | 3,35 | 3,89 | +0,54 |
| fixed-leaks-10 ep9 | 10,29 | 12,80 | +2,51 | +2,82 | 2,74 | 3,50 | +0,75 |

Висновок для методології: оцінка «0,6–0,85 п.п.» з README і `thesis_plan.md` (розд. 4 п. 2) справедлива лише для студентів, а моделі, що вчилися на CV train кілька епох від оригінального Whisper, завищені на 1,6–2,8 п.п. (поправка). Розрив між орієнтиром CV-only 5 епох і найкращим студентом на CV27 скорочується з 6,09 п.п. (10,68 проти 16,77) до 3,47 п.п. на `test_cleaned` (13,11 проти 16,58) (обчислено).

#### 6.4.7 Графіки

| Файл | Опис |
|---|---|
| `output/leak-clean/plots/eval_leaks.png` | горизонтальні стовпці для кожного dev/test: частка кліпів, текст яких є в будь-якому train — усі тексти (синій) і тексти ≥ 4 слів (помаранчевий); максимум у `cv10_test/test` (77,8 % / 68,0 %), далі `common_voice/test` (45,7 % / 40,3 %), FLEURS — 0 % |
| `output/leak-clean/plots/overlap_matrix.png` | теплова карта 13 × 13 сплітів (логарифмічна шкала): кількість різних нормалізованих текстів, спільних для пари сплітів, на діагоналі — повтори всередині спліту; найбільші позадіагональні клітинки: CV train–test 4 985, CV train–dev 3 405, CV test–CV10 2 885, CV train–CV10 2 288, OpenTTS–TG Voices 6 798; FLEURS не перетинається ні з чим |
| `output/leak-clean/plots/coverage.png` | аудіо й метадані без пари: `opentts/train` 753 аудіо без метаданих, `voa/train` 630 аудіо без тексту |

![output/leak-clean/plots/eval_leaks.png](../../output/leak-clean/plots/eval_leaks.png)

![output/leak-clean/plots/overlap_matrix.png](../../output/leak-clean/plots/overlap_matrix.png)

![output/leak-clean/plots/coverage.png](../../output/leak-clean/plots/coverage.png)


#### 6.4.8 Скрипти для відтворення

```bash
env/bin/python datasets/leak-clean/check_leaks.py                 # усе + report.md
env/bin/python datasets/leak-clean/check_leaks.py --no-audio-hash # без SHA-256
env/bin/python datasets/leak-clean/find_audio_matches.py --min-text-similarity 0.6
env/bin/python datasets/leak-clean/make_sha_notebook.py && jupyter lab datasets/leak-clean/review_sha.ipynb
env/bin/python datasets/leak-clean/make_acoustic_notebook.py && jupyter lab datasets/leak-clean/review_acoustic.ipynb
env/bin/python datasets/leak-clean/clean_common_voice_train.py --dry-run
env/bin/python datasets/leak-clean/score_common_voice_cleaned.py --csv cleaned_scores.csv
```


### 6.5 Датасети (`datasets/info.yaml`)

| Назва | Версія | Ліцензія | Посилання | Годин (усього / з мітками) | Роль у проєкті |
|---|---|---|---|---|---|
| Common Voice uk | 27.0 | CC0-1.0 | https://mozilladatacollective.com/datasets/cmu5wgebl00d0nq07ktip1i6c | 121,37 / 108,75 (12,62 без міток) | train (62 460), dev, test |
| VOA (новини) | не опубліковано | — | — | 2 313,36 / 0 | нерозмічений train для псевдоміток |
| FLEURS uk | не знайдено | CC-BY-4.0 | https://huggingface.co/datasets/google/fleurs | 12,24 / 12,24 | test (train лише в прогонах «+fleurs») |
| OpenTTS-UK | не знайдено | за голосами: kateryna CC-BY-NC-2.0 (2,83 год); lada, mykyta, oleksa, tetiana Apache-2.0 (10,62; 9,07; 6,22; 8,54 год) | https://huggingface.co/datasets/Yehor/opentts-uk | 37,28 / 37,28 | train лише в застарілих `baseline/whisper-uk-*` |
| TG Voices UK | не знайдено | CC0-1.0 | https://huggingface.co/datasets/Yehor/tg-voices-uk | 13,30 / 13,30 | те саме |
| Ukrainian Dialects | не знайдено | CC-BY-4.0 | https://huggingface.co/datasets/KSE-RESEARCH-Group/ukr-dialects-audio-dataset | 60,63 / 60,63 | test (train лише в `baseline/whisper-uk-*`) |
| Common Voice 10 uk test (clean) | 10.0 | MPL-2.0 | https://huggingface.co/datasets/Yehor/cv10-uk-testset-clean | 4,63 / 4,63 | test; виключено з підсумку тез через витік |
| Broadcast Speech UK | не знайдено | не знайдено | https://huggingface.co/datasets/Yehor/broadcast-speech-uk | 300,18 / 0 (мітки ненадійні) | не використовується |
| YODAS2 uk000 | не знайдено | CC-BY-3.0 | https://huggingface.co/datasets/espnet/yodas2/tree/main/data/uk000 | 629,35 / 0 | не використовується |
| YODAS2 uk100 | не знайдено | CC-BY-3.0 | https://huggingface.co/datasets/espnet/yodas2/tree/main/data/uk100 | 518,62 / 0 | не використовується |

Джерело ролей: `config.yaml:94-96` (поточні `train: [common_voice]`, `dev: [common_voice]`, `test: [common_voice, fleurs, ukr_dialects, cv10_test]`), `results.csv → train_data`, коментар `config.yaml:93` (Broadcast Speech і YODAS2 вилучено через ненадійні мітки).


### 6.6 Знахідки аудиту, важливі для методології (`REVIEW_FINDINGS.md`, 2026-11-03, статуси до 2026-11-08)

| № | Знахідка | Вплив на методологію | Статус |
|---|---|---|---|
| 1 | Увесь dev CV (10 955 кліпів) був у train (73 415 рядків) | dev-WER недійсний, вибір чекпоінта за dev некоректний | виправлено в даних (train 62 460); перевірки перетину в завантажувачі немає (`todo.md` п. 2) |
| 2 | Test CV перетинається з train за реченнями (4 992; 45,6 %) і дикторами (3 932; 36 %) | WER на CV27 оптимістичний | свідомо лишено; кількісна оцінка через `test_cleaned` (+0,6…0,85 п.п. для bf16-mixed) |
| 4 | `precision: bf16-true` при lr 1e-5 — оновлення округлюються до нуля, декодер майже не вчиться | усі моделі в bf16-true занижені й непорівнянні | виправлено в конфігах (`bf16-mixed`, `config.yaml:72`); вчителя Т0 не перенавчено (`todo.md` п. 1) |
| 7 | `accepted_mark: [0..5]` вимикає фільтр | — | навмисно (ablation) |
| 8 | 1 487 VAD-сегментів < 0,5 с, 2 975 < 1 с, найкоротший 5 мс | галюцинації вчителя потрапляють у псевдомітки | відкрито (статусу немає) |
| 13 | LLM-суддя бачить лише текст, не аудіо; `generate` без захисту від повторів; можлива prompt-injection | суддя не ловить правдоподібні вигадки; зациклені мітки | відкрито |
| 14.2 | `evaluate.py` для student бере базу з `model.id`, а не `student.base` | зараз однакові | відкрито |
| 16 | `evaluate.py` будує всі спліти (train теж) щоб отримати test; `num_workers=0` | повільно; оцінка падає без train-даних і без CSV фільтра витоків | відкрито |
| 19 | версії `torch`, `transformers`, `lightning`, `jiwer` не закріплені | відтворюваність метрик | частково |
| 20 | відновлення посеред епохи повторює дані, LR може стати 0 | відтворюваність навчання | відкрито |
| 24 | відсутні аудіо CV відкидаються мовчки | тест може відрізнятися між машинами | відкрито |
| 26 | `torch.load(weights_only=False)` | безпека | відкрито |
| 32 | `results.json` перезаписується в тій самій теці; метрики розкидані по `version_N` | облік експериментів | відкрито |

Додаткові обмеження з інших файлів:

1. Нормалізація однакова для WER і пошуку витоків, крім уніфікації апострофів `ʼ‘`` ` `` (лише в `leak_common.py`).
2. Один сід (`training.seed: 42`, `config.yaml:73`), довірчих інтервалів немає (`thesis_plan.md` розд. 4 п. 5).
3. Перевірки дикторів для датасетів, крім Common Voice, немає (`datasets/leak-clean/README.md`, «Обмеження» п. 1).


### 6.7 Виявлені неузгодженості

1. `report.md` (2026-11-08) показує для `cv10_test/test` 2 597 збігів «за іменем файлу» з `ukr_dialects/train` (81,1 %) з `same_bytes = no`; імена `100.wav`, `1000.wav` тощо, тексти різні. Це хибні збіги: фільтр схожості тексту ≥ 0,6 (`find_audio_matches.py:45`) у звіті ще не застосовано, а README стверджує, що імена CV10 мають вигляд `common_voice_uk_*.mp3`, що не так. Звіт і `audio_matches.csv` (6 442 рядки) слід перегенерувати.
2. Поріг `THRESHOLD = 0.8` у `make_acoustic_notebook.py:101`, а README рекомендує 0,97. На результат не вплинуло: усі 311 збігів мають `score = 1.0`.
3. `score = 1.0`, `shift_ms = 0`, `duration_diff = 0` у всіх 311 рядках `acoustic_matches.csv` при різних розмірах файлів (~167 КБ проти ~39 КБ) — тобто після декодування сигнал ідентичний; точне значення до округлення в CSV не збережено.
4. Скрипти `docs/thesis/figures/*.py` після переміщення мають хибний `ROOT` (`parents[2]` = `docs/`) і досі згадують `docs/figures/`.
5. `figure1.png` містить 4 моделі, а план тез (`thesis_plan.md` розд. 3.3, «Рисунок 1» п. 3) описує 5 (з CV-only 5 епох); CV-only є лише на `figure2.png`.
6. `results.csv`: прогін `student_whisper-commonvoice-based-epoch01-02` має старий формат (`combined`), `combined_mean` порожній.
7. Кількість тестових кліпів CV з реченням із train: 4 992 (дослівно, `REVIEW_FINDINGS.md`), 5 001 (нормалізація WER, `common-voice/README.md`), 5 010 (нормалізація `leak_common` + усі train-набори, `report.md`).
8. `MANIFEST_SCRIPTS_REVIEW.md` описує стару версію очищення (38 849 рядків, 74 диктори з 435), тоді як актуальний train — 62 460 рядків, 376 дикторів.
9. Вчитель для нових псевдоміток у `config.yaml:99` — `…10-epochs-epoch05.ckpt`, оцінки якого в `results.csv` немає.
10. Фільтр витоків застосовано лише до прогонів після 14.11; прогони `whisper-uk-cv-fixed-*` (09.11) і студенти (07–08.11) навчені на train з 311 кліпами CV10, тож їхній WER на CV10 занижений.
11. Оцінка завищення WER на CV27 через спільних дикторів і речення («0,6–0,85 п.п.», `thesis_plan.md` розд. 4 п. 2, `datasets/leak-clean/README.md:13`) застаріла: для орієнтира CV-only 5 епох поправка +2,74 п.п., для 10 епох +2,82 п.п. (п. 4.6, обчислено).
12. `todo.md` п. 1 згадує `batch_size.train: 32`, а `config.yaml:56` — 16 (з `accumulate_grad_batches: 2`).

## 7. Зведений каталог готових графіків

Повні описи — у розділах, указаних у стовпці «Опис». Позначки: **МД** — рекомендовано для дисертації; **перебудувати** — графік застарілий або неповний; **ні** — модель застаріла або непридатна для порівняння (див. 6.2.2).

### 7.1 Дані та попередня обробка

| Графік | Що показує | Опис | Статус |
|---|---|---|---|
| `analysis/voa-meta-plots/language_agreement.png` | Збіг мови MMS і Whisper: 19 756 проти 751 | 3.6 | МД |
| `analysis/voa-meta-plots/language_pie.png` | Мови VOA за MMS | 3.6 | МД |
| `analysis/voa-meta-plots/confidence.png` | Впевненість MMS | 3.6 | МД |
| `analysis/voa-meta-plots/language.png` | Мови за Whisper (часткові метадані, 2 613 файлів) | 2.3 | перебудувати |
| `analysis/voa-meta-plots/duration_*.png`, `whisper_length.png` | Тривалості файлів VOA, довжина транскриптів | 2.3 | перебудувати (`meta_stats.py`) |
| `analysis/voa-meta-plots/sample_rate.png`, `num_channels.png`, `codec.png`, `split.png`, `is_labelled.png` | Однорідність формату | 2.3 | у МД достатньо таблиці 2.2 |
| `docs/thesis/practic.pdf`, рис. 3.1–3.3 | Збіжність мов, розподіл мов, тривалості (версії з практики) | 3.6 | як резерв |

![analysis/voa-meta-plots/language.png](../../analysis/voa-meta-plots/language.png)


### 7.2 Псевдомітки та LLM-суддя

| Графік | Що показує | Опис | Статус |
|---|---|---|---|
| `output/pseudo-labelling/model-whisper-uk-epoch00-01/label_quality.png` | Розподіл оцінок судді для псевдоміток Т0 | 4.2–4.3 | МД |
| `analysis/label-judge/google_gemma-4-26b-a4b-it-02/plots/mark_distribution.png` | Розподіл оцінок Gemma на людських мітках за датасетами | 4.4 | МД |
| `analysis/label-judge/google_gemma-4-26b-a4b-it-02/plots/mean_mark.png` | Середня оцінка Gemma за датасетами (діалекти найнижчі) | 4.4 | МД (рис. 3 тез) |
| `analysis/label-judge/google_gemma-4-26b-a4b-it-02/plots/poor_share.png` | Частка низьких оцінок за датасетами | 4.4 | МД |
| `analysis/label-judge/typesafe_jev-1.14-02/plots/*.png` | Ті самі три графіки для судді Jev | 4.4 | для порівняння суддів |

### 7.3 Навчання

Кожна тека `output/fine-tune/<запуск>/<версія>/plots/` містить `learning_rate.png`, `train_loss.png`, `train_loss_epoch.png`, `val_edit_counts.png`, `val_error_rates.png`, `val_loss.png` (деякі ще й `test_error_rates.png`). Опис типів — 5.1.6.

| Запуск | Статус |
|---|---|
| `model-whisper-m-fixed-leaks-10-epochs/version_0` | МД: обґрунтування числа епох (dev-WER на плато, dev-loss росте) — 5.4 |
| `model-whisper-m-fixed-leaks-5-epochs/version_1` | перебудувати: лише відновлена 5-та епоха, повної кривої орієнтира немає |
| `model-whisper-m-fixed-leaks-5-epochs+fleurs/version_0` | ні (FLEURS у train) |
| `model-whisper-uk-cv-fixed/version_0`, `model-whisper-uk-cv-fleurs/version_0` | ні / допоміжні |
| `model-whisper-uk/01-common-voice-only` | лише як крива навчання Т0 |
| `model-whisper-uk/02-all-train-sets`, `student_model-whisper-uk/version_1_voa-only` | ні (застарілі) |
| `student_mix-*` | графіків немає — побудувати `python main.py plot` з `metrics.csv` |

`output/mlflow/artifacts/*/artifacts/plots/` — копії тих самих графіків у MLflow, окремо не використовувати.

### 7.4 Оцінювання

Кожна тека `evaluation/<прогін>/plots/` містить `error_rates.png` (WER/CER), `error_rates_by_dataset.png` (за тестовими наборами), `edit_counts.png` (заміни / пропуски / вставки), `utterance_wer.png` (розподіл WER на рівні речень). Опис — 6.1.9, 6.3.6.

| Графік / прогони | Статус |
|---|---|
| `docs/thesis/figures/figure1.png` (`make_figure1.py`) | 4 моделі; МД, але додати орієнтир CV-only і виправити `ROOT` у скрипті (6.7) |
| `docs/thesis/figures/figure2.png` (`make_figure2.py`) | 7 моделей (рис. 2 тез); МД |
| `evaluation/baseline/model_comparison.png` | перебудувати (`evaluation/batch-compare.py`): містить старі моделі |
| `evaluation/baseline/original-whisper-medium/plots/` | МД (відправна точка) |
| `evaluation/baseline/common-voice-only-baseline/plots/` | МД (вчитель Т0) |
| `evaluation/student_mix-*/plots/` (4 прогони) | МД (студенти) |
| `evaluation/whisper-m-fixed-leaks-5-epochs-epoch04-01/plots/` | МД (орієнтир) |
| `evaluation/whisper-m-fixed-leaks-10-epochs-*`, `*-5-epochs-epoch00-01` | лише згадати |
| `evaluation/baseline/whisper-uk-*`, `student_whisper-commonvoice-based-*`, `whisper-uk-cv-*`, `*-fleurs-*` | ні |

### 7.5 Витоки даних

| Графік | Що показує | Опис | Статус |
|---|---|---|---|
| `output/leak-clean/plots/overlap_matrix.png` | Теплова карта 13 × 13 сплітів: кількість спільних нормалізованих текстів | 6.4.7 | МД |
| `output/leak-clean/plots/eval_leaks.png` | Частка кліпів кожного dev/test, текст яких є в будь-якому train (усі тексти і ≥ 4 слів) | 6.4.7 | МД |
| `output/leak-clean/plots/coverage.png` | Аудіо й метадані без пари (OpenTTS, VOA) | 6.4.7 | ні |

### 7.6 Графіки, яких бракує для МД

1. Схема конвеєра методу (розд. 1.1).
2. Архітектура ПЗ.
3. Узгодженість класифікаторів мови за 4 категоріями у файлах і годинах (табл. 3.1).
4. Години псевдоміток за порогами оцінки (4.3).
5. Повна крива навчання орієнтира CV-only 5 епох.
6. Криві навчання студентів `student_mix-*`.

### 7.7 Галерея графіків окремих запусків

Мініатюри; клік відкриває повний розмір. Статус — як у 7.3–7.4.

#### 7.7.1 Інші метадані VOA

| is_labelled | split |
|---|---|
| ![is_labelled.png](../../analysis/voa-meta-plots/is_labelled.png) | ![split.png](../../analysis/voa-meta-plots/split.png) |

#### 7.7.2 Навчання (`output/fine-tune/`)

##### `model-whisper-m-fixed-leaks-10-epochs/version_0` — МД: вибір кількості епох

| learning_rate | train_loss | train_loss_epoch | val_loss | val_error_rates | val_edit_counts |
|---|---|---|---|---|---|
| ![learning_rate.png](../../output/fine-tune/model-whisper-m-fixed-leaks-10-epochs/version_0/plots/learning_rate.png) | ![train_loss.png](../../output/fine-tune/model-whisper-m-fixed-leaks-10-epochs/version_0/plots/train_loss.png) | ![train_loss_epoch.png](../../output/fine-tune/model-whisper-m-fixed-leaks-10-epochs/version_0/plots/train_loss_epoch.png) | ![val_loss.png](../../output/fine-tune/model-whisper-m-fixed-leaks-10-epochs/version_0/plots/val_loss.png) | ![val_error_rates.png](../../output/fine-tune/model-whisper-m-fixed-leaks-10-epochs/version_0/plots/val_error_rates.png) | ![val_edit_counts.png](../../output/fine-tune/model-whisper-m-fixed-leaks-10-epochs/version_0/plots/val_edit_counts.png) |

##### `model-whisper-m-fixed-leaks-5-epochs/version_1` — орієнтир CV-only 5 епох; лише відновлена 5-та епоха

| learning_rate | train_loss | train_loss_epoch | val_loss | val_error_rates | val_edit_counts |
|---|---|---|---|---|---|
| ![learning_rate.png](../../output/fine-tune/model-whisper-m-fixed-leaks-5-epochs/version_1/plots/learning_rate.png) | ![train_loss.png](../../output/fine-tune/model-whisper-m-fixed-leaks-5-epochs/version_1/plots/train_loss.png) | ![train_loss_epoch.png](../../output/fine-tune/model-whisper-m-fixed-leaks-5-epochs/version_1/plots/train_loss_epoch.png) | ![val_loss.png](../../output/fine-tune/model-whisper-m-fixed-leaks-5-epochs/version_1/plots/val_loss.png) | ![val_error_rates.png](../../output/fine-tune/model-whisper-m-fixed-leaks-5-epochs/version_1/plots/val_error_rates.png) | ![val_edit_counts.png](../../output/fine-tune/model-whisper-m-fixed-leaks-5-epochs/version_1/plots/val_edit_counts.png) |

##### `model-whisper-uk/01-common-voice-only` — крива навчання Т0 (bf16-true)

| learning_rate | train_loss | train_loss_epoch | val_loss | val_error_rates | val_edit_counts | test_error_rates |
|---|---|---|---|---|---|---|
| ![learning_rate.png](../../output/fine-tune/model-whisper-uk/01-common-voice-only/plots/learning_rate.png) | ![train_loss.png](../../output/fine-tune/model-whisper-uk/01-common-voice-only/plots/train_loss.png) | ![train_loss_epoch.png](../../output/fine-tune/model-whisper-uk/01-common-voice-only/plots/train_loss_epoch.png) | ![val_loss.png](../../output/fine-tune/model-whisper-uk/01-common-voice-only/plots/val_loss.png) | ![val_error_rates.png](../../output/fine-tune/model-whisper-uk/01-common-voice-only/plots/val_error_rates.png) | ![val_edit_counts.png](../../output/fine-tune/model-whisper-uk/01-common-voice-only/plots/val_edit_counts.png) | ![test_error_rates.png](../../output/fine-tune/model-whisper-uk/01-common-voice-only/plots/test_error_rates.png) |

##### `model-whisper-uk-cv-fixed/version_0` — допоміжний, до фільтра витоків

| learning_rate | train_loss | train_loss_epoch | val_loss | val_error_rates | val_edit_counts |
|---|---|---|---|---|---|
| ![learning_rate.png](../../output/fine-tune/model-whisper-uk-cv-fixed/version_0/plots/learning_rate.png) | ![train_loss.png](../../output/fine-tune/model-whisper-uk-cv-fixed/version_0/plots/train_loss.png) | ![train_loss_epoch.png](../../output/fine-tune/model-whisper-uk-cv-fixed/version_0/plots/train_loss_epoch.png) | ![val_loss.png](../../output/fine-tune/model-whisper-uk-cv-fixed/version_0/plots/val_loss.png) | ![val_error_rates.png](../../output/fine-tune/model-whisper-uk-cv-fixed/version_0/plots/val_error_rates.png) | ![val_edit_counts.png](../../output/fine-tune/model-whisper-uk-cv-fixed/version_0/plots/val_edit_counts.png) |

##### `model-whisper-m-fixed-leaks-5-epochs+fleurs/version_0` — ні: FLEURS у train

| learning_rate | train_loss | train_loss_epoch | val_loss | val_error_rates | val_edit_counts |
|---|---|---|---|---|---|
| ![learning_rate.png](../../output/fine-tune/model-whisper-m-fixed-leaks-5-epochs+fleurs/version_0/plots/learning_rate.png) | ![train_loss.png](../../output/fine-tune/model-whisper-m-fixed-leaks-5-epochs+fleurs/version_0/plots/train_loss.png) | ![train_loss_epoch.png](../../output/fine-tune/model-whisper-m-fixed-leaks-5-epochs+fleurs/version_0/plots/train_loss_epoch.png) | ![val_loss.png](../../output/fine-tune/model-whisper-m-fixed-leaks-5-epochs+fleurs/version_0/plots/val_loss.png) | ![val_error_rates.png](../../output/fine-tune/model-whisper-m-fixed-leaks-5-epochs+fleurs/version_0/plots/val_error_rates.png) | ![val_edit_counts.png](../../output/fine-tune/model-whisper-m-fixed-leaks-5-epochs+fleurs/version_0/plots/val_edit_counts.png) |

##### `model-whisper-uk-cv-fleurs/version_0` — ні: FLEURS у train

| learning_rate | train_loss | train_loss_epoch | val_loss | val_error_rates | val_edit_counts |
|---|---|---|---|---|---|
| ![learning_rate.png](../../output/fine-tune/model-whisper-uk-cv-fleurs/version_0/plots/learning_rate.png) | ![train_loss.png](../../output/fine-tune/model-whisper-uk-cv-fleurs/version_0/plots/train_loss.png) | ![train_loss_epoch.png](../../output/fine-tune/model-whisper-uk-cv-fleurs/version_0/plots/train_loss_epoch.png) | ![val_loss.png](../../output/fine-tune/model-whisper-uk-cv-fleurs/version_0/plots/val_loss.png) | ![val_error_rates.png](../../output/fine-tune/model-whisper-uk-cv-fleurs/version_0/plots/val_error_rates.png) | ![val_edit_counts.png](../../output/fine-tune/model-whisper-uk-cv-fleurs/version_0/plots/val_edit_counts.png) |

##### `model-whisper-uk/02-all-train-sets` — ні: застарілий

| learning_rate | train_loss | train_loss_epoch | val_loss | val_error_rates | val_edit_counts | test_error_rates |
|---|---|---|---|---|---|---|
| ![learning_rate.png](../../output/fine-tune/model-whisper-uk/02-all-train-sets/plots/learning_rate.png) | ![train_loss.png](../../output/fine-tune/model-whisper-uk/02-all-train-sets/plots/train_loss.png) | ![train_loss_epoch.png](../../output/fine-tune/model-whisper-uk/02-all-train-sets/plots/train_loss_epoch.png) | ![val_loss.png](../../output/fine-tune/model-whisper-uk/02-all-train-sets/plots/val_loss.png) | ![val_error_rates.png](../../output/fine-tune/model-whisper-uk/02-all-train-sets/plots/val_error_rates.png) | ![val_edit_counts.png](../../output/fine-tune/model-whisper-uk/02-all-train-sets/plots/val_edit_counts.png) | ![test_error_rates.png](../../output/fine-tune/model-whisper-uk/02-all-train-sets/plots/test_error_rates.png) |

##### `student_model-whisper-uk/version_1_voa-only` — ні: застарілий (18,57 % з практики)

| learning_rate | train_loss | train_loss_epoch | val_loss | val_error_rates | val_edit_counts | test_error_rates |
|---|---|---|---|---|---|---|
| ![learning_rate.png](../../output/fine-tune/student_model-whisper-uk/version_1_voa-only/plots/learning_rate.png) | ![train_loss.png](../../output/fine-tune/student_model-whisper-uk/version_1_voa-only/plots/train_loss.png) | ![train_loss_epoch.png](../../output/fine-tune/student_model-whisper-uk/version_1_voa-only/plots/train_loss_epoch.png) | ![val_loss.png](../../output/fine-tune/student_model-whisper-uk/version_1_voa-only/plots/val_loss.png) | ![val_error_rates.png](../../output/fine-tune/student_model-whisper-uk/version_1_voa-only/plots/val_error_rates.png) | ![val_edit_counts.png](../../output/fine-tune/student_model-whisper-uk/version_1_voa-only/plots/val_edit_counts.png) | ![test_error_rates.png](../../output/fine-tune/student_model-whisper-uk/version_1_voa-only/plots/test_error_rates.png) |

#### 7.7.3 Оцінювання (`evaluation/`)

##### `baseline/original-whisper-medium` — МД: відправна точка

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/baseline/original-whisper-medium/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/baseline/original-whisper-medium/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/baseline/original-whisper-medium/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/baseline/original-whisper-medium/plots/utterance_wer.png) |

##### `baseline/common-voice-only-baseline` — МД: вчитель Т0

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/baseline/common-voice-only-baseline/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/baseline/common-voice-only-baseline/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/baseline/common-voice-only-baseline/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/baseline/common-voice-only-baseline/plots/utterance_wer.png) |

##### `student_mix-m345-cv-m5-cv-s1-pseudo-m345-common_voice-epoch00-01` — МД: студент крок 1, ПМ 3–5 (найкращий)

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/student_mix-m345-cv-m5-cv-s1-pseudo-m345-common_voice-epoch00-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/student_mix-m345-cv-m5-cv-s1-pseudo-m345-common_voice-epoch00-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/student_mix-m345-cv-m5-cv-s1-pseudo-m345-common_voice-epoch00-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/student_mix-m345-cv-m5-cv-s1-pseudo-m345-common_voice-epoch00-01/plots/utterance_wer.png) |

##### `student_mix-m345-cv-m5-cv-s2-pseudo-m5-common_voice-epoch00-01` — МД: студент крок 2, ПМ 5

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/student_mix-m345-cv-m5-cv-s2-pseudo-m5-common_voice-epoch00-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/student_mix-m345-cv-m5-cv-s2-pseudo-m5-common_voice-epoch00-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/student_mix-m345-cv-m5-cv-s2-pseudo-m5-common_voice-epoch00-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/student_mix-m345-cv-m5-cv-s2-pseudo-m5-common_voice-epoch00-01/plots/utterance_wer.png) |

##### `student_mix-m2345-cv-m45-cv-s1-pseudo-m2345-common_voice-epoch00-01` — МД: студент крок 1, ПМ 2–5

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/student_mix-m2345-cv-m45-cv-s1-pseudo-m2345-common_voice-epoch00-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/student_mix-m2345-cv-m45-cv-s1-pseudo-m2345-common_voice-epoch00-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/student_mix-m2345-cv-m45-cv-s1-pseudo-m2345-common_voice-epoch00-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/student_mix-m2345-cv-m45-cv-s1-pseudo-m2345-common_voice-epoch00-01/plots/utterance_wer.png) |

##### `student_mix-m2345-cv-m45-cv-s2-pseudo-m45-common_voice-epoch00-01` — МД: студент крок 2, ПМ 4–5

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/student_mix-m2345-cv-m45-cv-s2-pseudo-m45-common_voice-epoch00-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/student_mix-m2345-cv-m45-cv-s2-pseudo-m45-common_voice-epoch00-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/student_mix-m2345-cv-m45-cv-s2-pseudo-m45-common_voice-epoch00-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/student_mix-m2345-cv-m45-cv-s2-pseudo-m45-common_voice-epoch00-01/plots/utterance_wer.png) |

##### `whisper-m-fixed-leaks-5-epochs-epoch04-01` — МД: орієнтир CV-only 5 епох

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/whisper-m-fixed-leaks-5-epochs-epoch04-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/whisper-m-fixed-leaks-5-epochs-epoch04-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/whisper-m-fixed-leaks-5-epochs-epoch04-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/whisper-m-fixed-leaks-5-epochs-epoch04-01/plots/utterance_wer.png) |

##### `whisper-m-fixed-leaks-5-epochs-epoch00-01` — проміжна точка орієнтира

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/whisper-m-fixed-leaks-5-epochs-epoch00-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/whisper-m-fixed-leaks-5-epochs-epoch00-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/whisper-m-fixed-leaks-5-epochs-epoch00-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/whisper-m-fixed-leaks-5-epochs-epoch00-01/plots/utterance_wer.png) |

##### `whisper-m-fixed-leaks-10-epochs-epoch09-01` — лише згадати: 10 епох

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/whisper-m-fixed-leaks-10-epochs-epoch09-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/whisper-m-fixed-leaks-10-epochs-epoch09-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/whisper-m-fixed-leaks-10-epochs-epoch09-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/whisper-m-fixed-leaks-10-epochs-epoch09-01/plots/utterance_wer.png) |

##### `whisper-m-fixed-leaks-10-epochs-epoch00-01` — лише згадати

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/whisper-m-fixed-leaks-10-epochs-epoch00-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/whisper-m-fixed-leaks-10-epochs-epoch00-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/whisper-m-fixed-leaks-10-epochs-epoch00-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/whisper-m-fixed-leaks-10-epochs-epoch00-01/plots/utterance_wer.png) |

##### `whisper-uk-cv-fixed-epoch00-01` — допоміжний

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/whisper-uk-cv-fixed-epoch00-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/whisper-uk-cv-fixed-epoch00-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/whisper-uk-cv-fixed-epoch00-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/whisper-uk-cv-fixed-epoch00-01/plots/utterance_wer.png) |

##### `whisper-uk-cv-fixed-epoch01-01` — допоміжний

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/whisper-uk-cv-fixed-epoch01-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/whisper-uk-cv-fixed-epoch01-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/whisper-uk-cv-fixed-epoch01-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/whisper-uk-cv-fixed-epoch01-01/plots/utterance_wer.png) |

##### `whisper-m-fixed-leaks-5-epochs-fleurs-epoch00-01` — ні: FLEURS у train

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/whisper-m-fixed-leaks-5-epochs-fleurs-epoch00-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/whisper-m-fixed-leaks-5-epochs-fleurs-epoch00-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/whisper-m-fixed-leaks-5-epochs-fleurs-epoch00-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/whisper-m-fixed-leaks-5-epochs-fleurs-epoch00-01/plots/utterance_wer.png) |

##### `whisper-m-fixed-leaks-5-epochs-fleurs-epoch04-01` — ні: FLEURS у train

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/whisper-m-fixed-leaks-5-epochs-fleurs-epoch04-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/whisper-m-fixed-leaks-5-epochs-fleurs-epoch04-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/whisper-m-fixed-leaks-5-epochs-fleurs-epoch04-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/whisper-m-fixed-leaks-5-epochs-fleurs-epoch04-01/plots/utterance_wer.png) |

##### `whisper-uk-cv-fleurs-epoch00-01` — ні: FLEURS у train

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/whisper-uk-cv-fleurs-epoch00-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/whisper-uk-cv-fleurs-epoch00-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/whisper-uk-cv-fleurs-epoch00-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/whisper-uk-cv-fleurs-epoch00-01/plots/utterance_wer.png) |

##### `whisper-uk-cv-fleurs-epoch01-01` — ні: FLEURS у train

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/whisper-uk-cv-fleurs-epoch01-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/whisper-uk-cv-fleurs-epoch01-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/whisper-uk-cv-fleurs-epoch01-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/whisper-uk-cv-fleurs-epoch01-01/plots/utterance_wer.png) |

##### `baseline/whisper-uk-epoch00-01` — ні: застарілий

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/baseline/whisper-uk-epoch00-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/baseline/whisper-uk-epoch00-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/baseline/whisper-uk-epoch00-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/baseline/whisper-uk-epoch00-01/plots/utterance_wer.png) |

##### `baseline/whisper-uk-epoch01-01` — ні: застарілий

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/baseline/whisper-uk-epoch01-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/baseline/whisper-uk-epoch01-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/baseline/whisper-uk-epoch01-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/baseline/whisper-uk-epoch01-01/plots/utterance_wer.png) |

##### `student_whisper-commonvoice-based-mark5-epoch01-01` — ні: bf16-true

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/student_whisper-commonvoice-based-mark5-epoch01-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/student_whisper-commonvoice-based-mark5-epoch01-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/student_whisper-commonvoice-based-mark5-epoch01-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/student_whisper-commonvoice-based-mark5-epoch01-01/plots/utterance_wer.png) |

##### `student_whisper-commonvoice-based-epoch01-01` — ні: bf16-true

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/student_whisper-commonvoice-based-epoch01-01/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/student_whisper-commonvoice-based-epoch01-01/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/student_whisper-commonvoice-based-epoch01-01/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/student_whisper-commonvoice-based-epoch01-01/plots/utterance_wer.png) |

##### `student_whisper-commonvoice-based-epoch01-02` — ні: bf16-true

| error_rates | error_rates_by_dataset | edit_counts | utterance_wer |
|---|---|---|---|
| ![error_rates.png](../../evaluation/student_whisper-commonvoice-based-epoch01-02/plots/error_rates.png) | ![error_rates_by_dataset.png](../../evaluation/student_whisper-commonvoice-based-epoch01-02/plots/error_rates_by_dataset.png) | ![edit_counts.png](../../evaluation/student_whisper-commonvoice-based-epoch01-02/plots/edit_counts.png) | ![utterance_wer.png](../../evaluation/student_whisper-commonvoice-based-epoch01-02/plots/utterance_wer.png) |

## 8. Головні застереження для тексту МД

Повні списки — у 4.7, 5.6, 6.7. Найважливіше:

1. **Т0** — чекпоінт після 1-ї епохи 2-епохового запуску в `bf16-true`, на train з витоком dev. Подавати лише як вчителя, від якого стартували студенти (5.6).
2. **Обґрунтування 5 епох.** Орієнтир навчено 14–15.11, а запуск на 10 епох, що показує плато dev-WER, — 16–17.11. У тексті не стверджувати, що 5 епох обрано заздалегідь за цим запуском; писати, що запуск на 10 епох підтверджує вибір (5.4, 5.6).
3. **Поправка на `test_cleaned`.** Для орієнтира CV-only 5 епох WER на CV зростає з 10,68 до 13,11 % (+2,74 п.п.), для студентів поправка менша. Оцінка «0,6–0,85 п.п.» з `thesis_plan.md` правильна лише для студентів. Розрив орієнтира й найкращого студента на CV скорочується з 6,09 до 3,47 п.п. (6.4.6).
4. **Вчитель у `config.yaml`** (`10-epochs-epoch05`) не відповідає жодній наявній розмітці. Усі оцінені студенти вчилися на мітках `model-whisper-uk-epoch00-01` (4.2, 5.6).
5. **Train CV:** у метаданих 62 460 рядків, після фільтра витоків реально 62 149. Студенти навчалися до фільтра (5.6, 6.7).
6. **Кількість параметрів:** у тезах 769 млн (офіційна цифра Whisper Medium), у логах 763 857 920; заморожено 307 216 384, навчається 456 641 536 (5.1).
7. **Захисту від галюцинацій у розмітці немає:** жадібне декодування; 925 сегментів мають понад 200 слів на ≤ 30 с (4.2).
8. **Модель судді для VOA ніде не записана.** Для розмітки Т0 — бекенд `llm` (Gemma 4 26B за практикою й тезами) (4.3).
9. **Тривалість роботи суддів** у звітах неправдоподібна; не наводити як факт (4.4).
10. **Звіт про витоки** `output/leak-clean/report.md` містить хибні збіги за іменем (CV10 ↔ ukr_dialects), його треба перегенерувати (6.7).
11. Число **18,57 %** з практики — тест застарілого запуску `student_model-whisper-uk` (5.6).
12. Параметри Silero VAD у коді не задано (значення за замовчуванням), версія пакета не зафіксована (4.1).
