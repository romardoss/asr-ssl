# Що треба зробити, щоб запускати репозиторій на vast.ai

Дата: 2026-11-03. Документ описує стан репозиторію на коміті `77a9889` і перелічує, чого бракує для запуску на орендованому інстансі vast.ai. Код поки не змінювався.

Зараз проєкт прив'язаний до одного сервера: дані й чекпоінти лежать у `/mnt/data/rberehun/...`, середовище — локальний venv `env/`, а GPU вибираються за фізичними номерами `nvidia0`/`nvidia1`. На vast.ai інстанс щоразу новий: це Docker-контейнер з порожнім диском, власною нумерацією GPU і без потрібних даних.

---

## 1. Блокери: без них запуск не стартує

### 1.1. Жорстко прописані шляхи `/mnt/data/rberehun` — ✅ виконано

Що зроблено:

1. Новий модуль `project_config.py` у корені: визначає, який конфіг читати, і резолвить шляхи. Файл вибирається змінною `ASR_CONFIG` (за замовчуванням `config.yaml`). Відносні шляхи, і в `ASR_CONFIG`, і всередині конфігу, рахуються від кореня репозиторію.
2. `fine-tune/settings.py`, `pseudo-labelling/settings.py`, `fine-tune/run_meta.py` і `pipeline.py` беруть `CONFIG_PATH` з `project_config.py`. `pipeline.py` явно передає `ASR_CONFIG` усім підпроцесам.
3. У коді більше немає шляхів `/mnt/data/rberehun` і `/home/romardoss`, а змінна `VOA_UKR_ROOT` більше не використовується. Корінь VOA береться лише з `pseudo_labelling.voa`.
4. Шляхи, які раніше були зашиті в коді, перенесено в конфіг:
   1. `paths.common_voice_splits`: TSV-спліти Common Voice;
   2. `paths.fine_tune_output`: метрики й логи fine-tune і student;
   3. `paths.evaluation`: результати `evaluation/evaluate.py`;
   4. нова секція `voa_prep`: `whisper_meta`, `mms_languages`, `other_languages`, `vad_progress`, `language_figures`, `meta_figures`.
5. Скрипти підготовки VOA (`scripts/monitor/*`, `pseudo-labelling/vad-chunking/chunk.py`) читають вхідні й вихідні файли з `pseudo_labelling.ukrainian_meta`, `pseudo_labelling.timestamps` і `voa_prep.*`. Тому перевірку `require_script_paths` у `pipeline.py` прибрано: вона була потрібна лише через те, що скрипти писали у фіксовані файли.
6. Типові значення CLI тепер беруться з конфігу:
   1. `predict_languages.py`: `--model-id` і `--batch-size` з `pipeline.mms_*`;
   2. `judge.py`: `--workers` і `--retries` з `pipeline.judge_*`; `--labels` — найновіший `labels.csv` для `pseudo_labelling.teacher_model`;
   3. `plot_marks.py`: `--marks` — найновіший `label_marks.csv` для того самого вчителя;
   4. `plot_metrics.py`: `--model-name` з `model.name`.
7. Створено `config.vast.yaml` з тими самими ключами, що й `config.yaml`. Відрізняються лише значення, які залежать від машини:
   1. дані в `/workspace/data/<набір>`, чекпоінти в `/workspace/checkpoints`;
   2. одна GPU `nvidia0` і в `gpus`, і в `student.gpus`;
   3. `pipeline.language` і `pipeline.timestamps` мають значення `skip`, бо їхні результати вже є в git.

Запуск на інстансі:

```bash
export ASR_CONFIG=config.vast.yaml
python main.py pipeline --dry-run
```

Що лишилось: `config.vast.yaml` треба тримати синхронним із `config.yaml`, коли додаються нові ключі. Якщо ключа бракує, завантажувач конфігу зупиняється з помилкою `missing <key>`, тож розбіжність одразу видно.

### 1.2. Вибір GPU через `nvidia0`/`nvidia1`

Де: `config.yaml` (`gpus`, `student.gpus`), `fine-tune/settings.py:296`, `pseudo-labelling/settings.py:90-101`, `fine-tune/fine-tune.py:39-40`, `fine-tune/fine-tune-student.py:36-37`, `pseudo-labelling/label.py:38-39`.

Проблема:

1. Код сам виставляє `CUDA_VISIBLE_DEVICES` за номером із конфігу.
2. У поточному `config.yaml` студент працює на `student.gpus.nvidia1: true`. На інстансі з однією GPU пристрою з індексом 1 немає, тому `CUDA_VISIBLE_DEVICES=1` приховає єдину карту й навчання впаде.
3. Етап `label` (з `run.json`) теж запускався на `nvidia1`.

Що зробити:

1. ✅ У `config.vast.yaml` увімкнено лише `nvidia0` і в `gpus`, і в `student.gpus`.
2. Краще: якщо `CUDA_VISIBLE_DEVICES` уже задано ззовні, не перезаписувати його. Так GPU можна вибирати й засобами vast.ai.
3. Орендувати інстанс з однією GPU: після незакомічених змін у `settings.py` код дозволяє рівно одну GPU, тож кілька карт не дадуть прискорення.

### 1.3. Немає опису середовища (Docker-образ або скрипт встановлення)

Зараз є лише `requirements.txt` і локальний venv `env/` (Python 3.12.3, у git не входить).

Проблеми в `requirements.txt`:

1. `torch`, `torchaudio`, `transformers`, `lightning`, `faster-whisper`, `ctranslate2` без версій. Свіжий `pip install` поставить torch під іншу версію CUDA, ніж у драйвері хоста vast.ai, або перезапише torch з образу.
2. ✅ `av` (PyAV) імпортується напряму (`fine-tune/speech_data.py:17`, `fine-tune/fine-tune-student.py:52`). Раніше його не було в `requirements.txt` і він приходив лише транзитивно через `faster-whisper`; тепер додано явно.
3. `pandas==3.0.6` і `numpy==2.5.3` потребують Python ≥ 3.11. Частина стандартних образів `pytorch/pytorch` має старіший Python.
4. `wandb` у списку є, але ніде не використовується: логування йде через `CSVLogger`.

Системні залежності:

1. `ffmpeg` — потрібен як бінарник: його викликають через `subprocess` у `pseudo-labelling/preprocess.py:93`, `pseudo-labelling/vad-chunking/chunk.py:123`, `scripts/monitor/predict_languages.py:131`, `scripts/monitor/meta_collector.py:198`. У базових PyTorch-образах його немає.
2. cuDNN 9 і cuBLAS 12 для `ctranslate2`/`faster-whisper` (етап `language`). Локально шлях до них задається через `LD_LIBRARY_PATH` у `.env`.

Що зробити (на вибір):

1. **Варіант A, рекомендований: власний Docker-образ.** Додати `Dockerfile`, наприклад:

   ```dockerfile
   FROM pytorch/pytorch:<версія>-cuda12.x-cudnn9-runtime   # Python >= 3.11
   RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git tmux rsync \
       && rm -rf /var/lib/apt/lists/*
   COPY requirements.lock.txt /tmp/
   RUN pip install --no-cache-dir -r /tmp/requirements.lock.txt
   WORKDIR /workspace/asr_ssl
   ```

   Зібрати, запушити в Docker Hub або GHCR і вказати образ у шаблоні vast.ai. Інстанс стартує за хвилину, і версії не «пливуть».
2. **Варіант B: скрипт `scripts/vast/setup.py`** — ✅ створено. Його можна запускати вручну або з On-start Script шаблону vast.ai (`cd /workspace/asr_ssl && python3 scripts/vast/setup.py`). Він ставить `ffmpeg`, `tmux`, `rsync` і `git`, виконує `pip install -r requirements.txt` і не дає pip замінити `torch` з образу. Також пише `.vast/env.sh` (`ASR_CONFIG`, `HF_HOME`, `LD_LIBRARY_PATH`) і перевіряє інстанс. Що саме він робить, описано в його докстрінгу. Простіше за Docker-образ, але кожен старт займає кілька хвилин і залежить від PyPI.
3. В обох варіантах:
   1. Зафіксувати версії: згенерувати `requirements.lock.txt` (`pip freeze`) з робочого серверного середовища, де вже йшло навчання.
   2. Додати `av` у `requirements.txt`.
   3. Не перевстановлювати `torch`, якщо він уже є в образі, або явно ставити його з `--index-url https://download.pytorch.org/whl/cu12x`.
   4. Прибрати `wandb` або почати його використовувати.

### 1.4. Завантаження даних — ✅ виконано

Код читає дані з локального диска у специфічних форматах. Тепер їх завантажує `scripts/vast/download_data.py`: окремо або через `python3 scripts/vast/setup.py --download`. Звідки береться кожен набір, задано в секції `downloads` конфігу.

| Ключ у конфігу | Джерело | Формат, якого чекає код |
|---|---|---|
| `paths.common_voice` | приватний репозиторій HF, `common-voice/*.tar` (Common Voice 27.0 з сервера) | `<root>/clips/*.mp3`; TSV-сплити беруться з `datasets/common-voice/raw/` у git |
| `paths.fleurs` | приватний репозиторій HF, `fleurs-uk/` (копія з сервера) | Формат `datasets.save_to_disk`: `<root>/{train,validation,test}/state.json` + `.arrow` |
| `paths.opentts` | `Yehor/opentts-uk` | Сирий знімок репозиторію HF: `voices/<voice>/train.jsonl` + `data/<voice>.zip` |
| `paths.tg_voices` | `Yehor/tg-voices-uk` | `<root>/data/train-*.parquet` |
| `paths.ukr_dialects` | `KSE-RESEARCH-Group/ukr-dialects-audio-dataset` | `<root>/data/{train,validation,test}-*.parquet` |
| `paths.cv10_test` | `Yehor/cv10-uk-testset-clean` | `<root>/data/train-*.parquet` |
| `pseudo_labelling.voa`, `student.voa` | `robinhad/VOA-ukr` (gated) | `<root>/data/train-*.parquet`, ≈ 128 ГБ |
| `model.id`, `student.base` | `openai/whisper-medium` | Кеш HF, завантажується автоматично |
| `pipeline.mms_model` | `facebook/mms-lid-256` | Кеш HF, потрібен лише для етапу `language` |

Що зроблено і як працює:

1. Структуру публічних репозиторіїв перевірено через API HF (2026-11-03). Вони вже мають той формат, який читає код, тож завантажуються як є, лише потрібні файли.
2. `google/fleurs` має інший формат (tsv + tar.gz або parquet), тому FLEURS береться з копії на сервері через приватний репозиторій.
3. Common Voice, FLEURS, чекпоінт учителя і `labels.csv` лежать у приватному репозиторії `downloads.private_repo`. Як його наповнити, описано в `HF_UPLOAD.md`.
4. Скрипт можна перезапускати: набори, які вже на місці, пропускаються, а обірване завантаження продовжується. Частину наборів можна вибрати через `--only` / `--skip`.
5. `robinhad/VOA-ukr` gated: акаунт, чий `HF_TOKEN` стоїть на інстансі, має отримати доступ.

Чому дані завантажуються, а не стрімляться:

1. Код читає файли з довільним доступом (row group у parquet, член zip-архіву, батч arrow). Стрімінг HF дає лише послідовне читання, тож довелося б переписати всі завантажувачі.
2. Навчання проходить дані кілька епох і вміє продовжувати з чекпоінта. Зі стрімінгом кожна епоха і кожне продовження заново тягнули б ≈ 128 ГБ VOA через мережу, а перемішування було б лише в межах буфера.
3. Швидкість мережі на vast.ai різна, а HF обмежує частоту запитів. GPU простоювала б в очікуванні даних, а час GPU коштує дорожче за час завантаження.
4. Одноразове завантаження ≈ 150 ГБ на машині з каналом ≥ 1 Гбіт/с займає приблизно від пів години до години. Під час вибору пропозиції на vast.ai варто фільтрувати за швидкістю завантаження (Inet Down).
5. Щоб не завантажувати повторно між сесіями, інстанс можна **зупиняти (stop), а не видаляти (destroy)**. Диск зберігається, хоча за нього йде плата, і наступний старт миттєвий. Мінус: після зупинки GPU цієї машини може зайняти хтось інший, і доведеться чекати.

### 1.5. Артефакти, яких немає в git — ✅ виконано

1. **Чекпоінт учителя** `model-whisper-uk-epoch00.ckpt` і **`labels.csv`** заливаються в приватний репозиторій HF (`checkpoints/` і `pseudo-labelling/`, див. `HF_UPLOAD.md`). `download_data.py` кладе їх у `paths.checkpoints` і `pseudo_labelling.output_dir`.
2. Ім'я теки розмітки будується з імені файлу вчителя після `Path.resolve()` (`pseudo-labelling/label.py:42`, `project_config.py`). Тому:
   1. файл чекпоінта має називатися так само: `model-whisper-uk-epoch00.ckpt`;
   2. якщо зробити symlink з іншою назвою цілі, `resolve()` візьме ім'я цілі, `labels.csv` не знайдеться, і розмітка почнеться заново.

### 1.6. Секрети та `.env`

1. ✅ У `.env.example` ключ називався `HF-TOKEN`, і `huggingface_hub` його не читав. Тепер він називається `HF_TOKEN`. `download_data.py` для сумісності читає зі старого `.env` обидва варіанти.
2. `.env` не потрапляє в git, тож на інстансі його немає. Секрети `HF_TOKEN`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` краще задавати як Environment Variables у шаблоні vast.ai (вони приватні для акаунта), а не копіювати файл.
3. `LD_LIBRARY_PATH` у `.env` — шлях конкретної машини. На vast.ai його треба або прописати в `Dockerfile`, або обчислювати з `site-packages/nvidia/{cudnn,cublas}/lib`. ✅ Це робить `scripts/vast/setup.py` (файл `.vast/env.sh`). Це потрібно лише для `scripts/monitor/meta_collector.py`, тобто етапу `language`.
4. `pipeline.judge_model` у конфігу порожній. Якщо `pipeline.clean: run`, його треба заповнити.

---

## 2. Важливо: без цього запуск ризикований або дорогий

### 2.1. Збереження результатів (інстанс тимчасовий)

Після видалення інстансу все на його диску зникає. Пишуть туди:

1. `paths.checkpoints`: чекпоінти fine-tune і student;
2. `output/fine-tune/<name>/version_*/metrics.csv` і графіки;
3. `output/pseudo-labelling/...`: `labels.csv`, `label_marks.csv`;
4. `evaluation/<model>/`: `results.json`, `predictions.csv`, графіки.

Що зробити:

1. Додати `scripts/vast/sync_results.sh`, який через `rclone` (S3/B2/GDrive) або `huggingface-cli upload` вивантажує ці теки. Запускати його:
   1. після кожного етапу в `pipeline.py` (або обгортці);
   2. періодично через `cron`/цикл, щоб чекпоінти не загубилися, якщо інстанс перервуть.
2. Альтернатива: vast.ai Cloud Sync або окремий том (volume), якщо він доступний у вибраному регіоні.
3. Писати лише `last.ckpt` і найкращий чекпоінт або чистити старі. Кожен чекпоінт whisper-medium зі станом оптимізатора займає кілька гігабайтів, а `checkpoint_every: 0.05` дає до 20 файлів на епоху.

### 2.2. Розмір диска

Розмір диска на vast.ai задається під час створення інстансу, і збільшити його потім не можна. Орієнтовна оцінка:

1. VOA-ukr: ≈ 128 ГБ;
2. Common Voice uk + FLEURS + OpenTTS + TG Voices + Ukrainian Dialects + CV10 test: десятки гігабайтів;
3. кеш HF (whisper-medium, mms-lid-256): ≈ 5 ГБ;
4. чекпоінти: від кількох до десятків гігабайтів залежно від `checkpoint_every`;
5. образ, середовище, тимчасові файли.

Рекомендація: **не менше 300 ГБ**, якщо потрібні всі етапи. Лише для студента й оцінки (VOA + Common Voice + чекпоінт учителя) достатньо ≈ 200 ГБ. Точні розміри наборів варто записати в `datasets/info.yaml` (поле `size_gb`).

### 2.3. Вибір GPU і batch size

1. `precision: bf16-true` і `torch.autocast(..., dtype=torch.bfloat16)` потребують GPU покоління Ampere або новішої: A100, A6000, RTX 3090/4090, L40S, H100. На V100, T4 і подібних bf16 не буде.
2. Batch size у `config.yaml` (`train: 32`, `testing: 64`, `pseudo_labelling.batch_size: 64`) підібрані під серверну карту. З коментарів: `testing: 64` займає ≈ 16 ГБ, `96` дає OOM. Для 24-гігабайтних карт `batch_size.train` і `student.batch_size.train` треба зменшити й перевірити на короткому запуску (`limit_train_batches: 20`).
3. У `config.vast.yaml` варто мати окремі профілі або коментарі з batch size для типових карт (24 ГБ, 48 ГБ, 80 ГБ).

### 2.4. Ресурси контейнера

1. `/dev/shm`: `DataLoader` з `num_workers: 4` передає тензори через спільну пам'ять. Якщо в контейнері `/dev/shm` малий, воркери падають з `Bus error`. Перевірити `df -h /dev/shm` на старті. Якщо там менше кількох гігабайтів, зменшити `num_workers` або запускати з більшим `--shm-size` (параметр Docker options у шаблоні).
2. `pipeline.chunk_processes: 6` і `num_workers: 4` треба узгодити з кількістю CPU, яку дає вибрана пропозиція vast.ai.

### 2.5. Робота при обриві SSH і перерваних (interruptible) інстансах

1. Довгі етапи треба запускати в `tmux`/`screen` або через `nohup ... > log 2>&1 &`, інакше обрив SSH зупинить навчання. Варто додати `scripts/vast/run_pipeline.sh`, який це робить і пише лог у теку, що синхронізується.
2. `training.resume`, `student.resume` і `pseudo_labelling.resume` уже є. Для interruptible-інстансів треба перевірити, що відновлення працює після повного перезапуску контейнера: дані й чекпоінти мають підтягуватися з віддаленого сховища перед стартом.

---

## 3. Бажано: зручність і відтворюваність

1. **Інструкція `README.md`** (зараз README немає): як створити шаблон на vast.ai, які змінні оточення задати, порядок команд:

   ```bash
   export ASR_CONFIG=config.vast.yaml
   python scripts/vast/download_data.py --sets voa common_voice teacher
   python main.py pipeline --dry-run
   tmux new -s train 'python main.py pipeline 2>&1 | tee logs/pipeline.log'
   ```

2. ✅ **Перевірка середовища перед стартом**: `python3 scripts/vast/setup.py --check-only`. Що перевіряється:
   1. `torch.cuda.is_available()` і підтримка bf16;
   2. наявність `ffmpeg`;
   3. існування всіх шляхів із конфігу;
   4. вільне місце на диску й розмір `/dev/shm`;
   5. наявність потрібних секретів.

   Це дешевше, ніж дізнатися про помилку через годину оплаченого часу.
3. **Етап `language` не запускати на vast.ai без потреби.** Його результати (`datasets/voa-ukr/ukrainian.csv`, `ukrainian_30s.csv`, `voa_ukr_meta.json`) уже є в git. У `config.vast.yaml` поставити `pipeline.language: skip` і `pipeline.timestamps: skip`, щоб не витрачати GPU-години й не налаштовувати `ctranslate2`/cuDNN.
4. **Записувати в `run.json` і `meta.yaml` інформацію про хост:** модель GPU, ID інстансу vast.ai, версії `torch`/`transformers`. Так результати з різних машин можна буде порівнювати.
5. **Перед дорогим запуском виправити критичні пункти з `REVIEW_FINDINGS.md`**, насамперед перетин dev/test з train у Common Voice і `bf16-true` при `lr=1e-5`. Інакше орендовані GPU-години підуть на навчання з некоректними метриками.

---

## 4. Мінімальний план, щоб запустити студента та оцінку

Якщо потрібно якнайшвидше запустити лише `student` + `evaluate`:

1. ✅ Вибір конфігу через `ASR_CONFIG` (п. 1.1).
2. ✅ `config.vast.yaml` створено. Лишилось підібрати batch size під вибрану карту.
3. Залити в приватний репозиторій HF чекпоінт учителя, `labels.csv`, Common Voice і FLEURS за `HF_UPLOAD.md` і вписати його назву в `downloads.private_repo` у `config.vast.yaml`.
4. ✅ `scripts/vast/setup.py` створено, `av` додано в `requirements.txt`. Лишилось зафіксувати версії пакетів (`requirements.lock.txt`).
5. ✅ `scripts/vast/download_data.py` створено. На інстансі: `python3 scripts/vast/setup.py --download`.
6. Написати `scripts/vast/sync_results.sh` для `paths.checkpoints`, `output/` та `evaluation/`.
7. Орендувати 1×GPU (Ampere+, ≥ 24 ГБ) з диском ≥ 200 ГБ, задати секрети в шаблоні, виконати `python main.py pipeline --dry-run`, потім повний запуск у `tmux`.
