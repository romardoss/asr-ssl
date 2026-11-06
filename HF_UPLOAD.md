# Як залити дані на Hugging Face для запуску на vast.ai

На інстансі vast.ai датасети й чекпоінти завантажує `scripts/vast/download_data.py`. Публічні набори він бере з їхніх репозиторіів на HF. Решта має лежати в одному **приватному** dataset-репозиторії на HF, і його треба наповнити один раз із сервера, де є дані. Ця інструкція описує, як це зробити.

Команди нижче виконуються на сервері, де лежать дані (`/mnt/data/rberehun/...`). Шляхи взято з `config.yaml`.

---

## 1. Що куди заливати

Приватний репозиторій, далі `<user>/asr-ssl-vast` (тип `dataset`):

| Тека в репозиторії | Що в ній | Звідки на сервері | Куди потрапить на vast.ai |
|---|---|---|---|
| `common-voice/` | `cv-uk-clips.tar`: архів з текою `clips/` | `/mnt/data/rberehun/common-voice/clips` | `paths.common_voice` (розпаковується) |
| `fleurs-uk/` | `train/`, `validation/`, `test/` у форматі `save_to_disk` | `/mnt/data/rberehun/fleurs-uk` | `paths.fleurs` |
| `checkpoints/` | `model-whisper-uk/model-whisper-uk-epoch00.ckpt` | `/mnt/data/rberehun/checkpoints/model-whisper-uk/` | `paths.checkpoints` |
| `pseudo-labelling/` | `model-whisper-uk-epoch00-01/` з `labels.csv`, `label_marks.csv`, `run.json` | `output/pseudo-labelling/` у репозиторії коду | `pseudo_labelling.output_dir` |

Заливати **не треба**: OpenTTS, TG Voices, Ukrainian Dialects, CV10 test і VOA. Вони публічні, і скрипт бере їх з оригінальних репозиторіїв (секція `downloads` у `config.vast.yaml`).

Назви тек у репозиторії фіксовані: скрипт шукає саме `common-voice/`, `fleurs-uk/`, `checkpoints/` і `pseudo-labelling/`. Назви файлу вчителя і теки розмітки теж змінювати не можна. Код шукає `labels.csv` у теці `<назва файлу вчителя без .ckpt>-NN`, тобто `model-whisper-uk-epoch00-01`.

---

## 2. Підготовка (один раз)

1. Встановіть CLI. Потрібен `huggingface_hub` версії 1.0 або новішої, де є команда `hf`:

   ```bash
   pip install -U huggingface_hub
   hf --help
   ```

2. Створіть токен на <https://huggingface.co/settings/tokens> з правом **Write** і увійдіть:

   ```bash
   hf auth login
   hf auth whoami        # покаже <user>
   ```

3. Створіть приватний dataset-репозиторій:

   ```bash
   hf repos create <user>/asr-ssl-vast --type dataset --private
   ```

Великі файли заливайте в `tmux`, щоб обрив SSH не зупинив завантаження. Якщо заливання обірвалося, повторіть ту саму команду: для тек `hf upload` продовжує з місця зупинки.

---

## 3. Заливання

### 3.1. Common Voice

Тисячі дрібних mp3 краще пакувати в один архів: HF погано працює з десятками тисяч файлів в одному репозиторії. Стискати не потрібно, бо mp3 вже стиснуті.

```bash
cd /mnt/data/rberehun/common-voice
tar -cf /mnt/data/rberehun/cv-uk-clips.tar clips
ls -lh /mnt/data/rberehun/cv-uk-clips.tar
hf upload <user>/asr-ssl-vast /mnt/data/rberehun/cv-uk-clips.tar common-voice/cv-uk-clips.tar --type dataset
```

Тека `clips/` має бути на верхньому рівні архіву, а не всередині `cv-corpus-27.0-.../uk/`, інакше скрипт її не знайде. Перевірити можна так: `tar -tf cv-uk-clips.tar | head -3` має показати рядки на зразок `clips/common_voice_uk_....mp3`.

Якщо архів вийде більшим за кілька десятків гігабайтів, розбийте його на кілька `.tar` за підмножинами файлів, наприклад `cv-uk-clips-1.tar` і `cv-uk-clips-2.tar`. Скрипт розпакує всі `.tar` у `common-voice/`. Розрізати один архів командою `split` не можна: такі частини не є окремими tar-архівами.

TSV-сплити Common Voice заливати не треба, вони вже є в git (`datasets/common-voice/raw/`).

### 3.2. FLEURS

На сервері FLEURS лежить у форматі `datasets.save_to_disk`, і саме цей формат читає код. Заливайте теку як є:

```bash
ls /mnt/data/rberehun/fleurs-uk          # має бути train/ validation/ test/, у кожній state.json
hf upload <user>/asr-ssl-vast /mnt/data/rberehun/fleurs-uk fleurs-uk --type dataset
```

Оригінальний `google/fleurs` на HF має інший формат (tsv і tar.gz або parquet), тому його не використовуємо.

### 3.3. Чекпоінт учителя

```bash
hf upload <user>/asr-ssl-vast \
  /mnt/data/rberehun/checkpoints/model-whisper-uk/model-whisper-uk-epoch00.ckpt \
  checkpoints/model-whisper-uk/model-whisper-uk-epoch00.ckpt \
  --type dataset
```

Інші чекпоінти заливайте за тим самим шаблоном: `checkpoints/<тека>/<файл>.ckpt` на HF відповідає `<paths.checkpoints>/<тека>/<файл>.ckpt` на vast.ai.

### 3.4. Псевдорозмітки

Команду виконуйте з кореня репозиторію коду на сервері:

```bash
ls output/pseudo-labelling/model-whisper-uk-epoch00-01      # labels.csv label_marks.csv run.json ...
hf upload <user>/asr-ssl-vast \
  output/pseudo-labelling/model-whisper-uk-epoch00-01 \
  pseudo-labelling/model-whisper-uk-epoch00-01 \
  --type dataset
```

Нові прогони розмітки заливайте поруч як `pseudo-labelling/model-whisper-uk-epoch00-02` і так далі. Код бере найновішу теку, у якій є `labels.csv`.

---

## 4. Перевірка

1. Відкрийте `https://huggingface.co/datasets/<user>/asr-ssl-vast/tree/main` і переконайтеся, що є чотири теки з розділу 1.
2. Або перевірте з консолі, нічого не завантажуючи:

   ```bash
   python -c "from huggingface_hub import list_repo_files as f; print('\n'.join(f('<user>/asr-ssl-vast', repo_type='dataset')))"
   ```

---

## 5. Налаштування для vast.ai

1. Пропишіть назву репозиторію в `config.vast.yaml` замість заглушки і закомітьте зміну:

   ```yaml
   downloads:
     private_repo: <user>/asr-ssl-vast
   ```

2. Запросіть доступ до `robinhad/VOA-ukr` на <https://huggingface.co/datasets/robinhad/VOA-ukr>. Цей датасет gated, тож доступ потрібен акаунту, чий токен буде на vast.ai. Якщо VOA на сервер уже завантажувався з вашого акаунта, доступ, найімовірніше, є.
3. Для vast.ai створіть окремий токен з правом лише на читання (**Read**). Задайте його як змінну оточення `HF_TOKEN` у шаблоні vast.ai, а не в git. Токен з правом запису на інстанс не кладіть.
4. На інстансі:

   ```bash
   cd /workspace/asr_ssl
   python3 scripts/vast/setup.py --download
   ```

   Або лише частину наборів:

   ```bash
   export ASR_CONFIG=config.vast.yaml
   python scripts/vast/download_data.py --only voa common_voice teacher labels
   ```

---

## 6. Ліцензії та приватність

1. Репозиторій має лишатися приватним.
2. Common Voice має ліцензію CC0. Умови Mozilla Data Collective, серед іншого, забороняють намагатися встановити особи дикторів. Перечитайте їх перед заливанням.
3. FLEURS має ліцензію CC-BY-4.0, тож копію можна зберігати із зазначенням авторства.
4. Перед заливанням великих файлів перевірте ліміт приватного сховища свого акаунта HF (Settings → Billing).
