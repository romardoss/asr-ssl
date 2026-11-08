# asr_ssl: напівкероване навчання Whisper для української

Проєкт дотреновує `openai/whisper-medium` на українському мовленні й перевіряє, чи допомагають псевдомітки на нерозміченому аудіо. Енкодер заморожено, навчається лише декодер.

1. **Вчитель.** Whisper дотреновується на розмічених датасетах: Common Voice, FLEURS, OpenTTS, TG Voices і Ukrainian Dialects.
2. **Псевдомітки.** Вчитель розпізнає нерозмічене аудіо VOA (близько 2 300 год), попередньо відфільтроване за мовою й нарізане на шматки до 30 с.
3. **Оцінка міток.** Суддя через OpenRouter ставить кожній псевдомітці оцінку від 0 до 5. Суддею може бути чат-LLM або decision-модель, наприклад Jev (`USE_DECISION_INSTEAD_OF_LLM`).
4. **Студент.** Whisper навчається на псевдомітках, відібраних за оцінкою.
   Навчання може йти ланцюжком кроків (секція `multi_step`), наприклад: псевдомітки -> Common Voice -> псевдомітки -> Common Voice. Кожен крок стартує з фінальних ваг попереднього. Крок `[pseudo, common_voice]` вчиться на обох наборах в одному запуску: спершу псевдомітки, а в кінці кожної епохи Common Voice.
5. **Оцінка моделей.** Тестові набори: Common Voice, FLEURS, Ukrainian Dialects і тест Common Voice 10.

Усі налаштування лежать у `config.yaml`, а для vast.ai — у `config.vast.yaml`. Кілька загальних налаштувань діють на всі етапи:

1. `gpus` — єдина GPU для всього. Щоб запустити другий `multi-step` паралельно, змініть у `config.yaml` і `gpus`, і `multi_step.name`: перший запуск уже зберіг свою копію конфігу й зміни не побачить. Інший варіант — окремі конфіги через `ASR_CONFIG=...`.
2. `checkpoint_every: 0.25` — проміжні чекпоінти на 25, 50 і 75 % епохи.
3. `remove_mid_epoch_checkpoints: true` — після завершеного навчання проміжні чекпоінти видаляються, а кінець кожної епохи лишається.

```bash
python main.py pipeline --dry-run              # показати етапи пайплайну
python main.py fine-tune --note "коментар"     # навчити вчителя й оцінити його
python main.py pipeline --note "коментар"      # псевдомітки -> студент -> оцінка
python main.py multi-step --dry-run           # план багатокрокового навчання з multi_step
python main.py evaluate --model /path/to.ckpt  # оцінити один чекпоінт
python main.py judge --backend decisions       # оцінити псевдомітки decision-моделлю
```

Повний список команд є в docstring `main.py`.

## Результати

| Файл | Що там |
|---|---|
| [`evaluation/results.csv`](evaluation/results.csv) | Головна таблиця: одна оцінка в рядку, WER і CER по кожному тестовому набору, параметри навчання, ланцюжок і крок `multi_step`, нотатка. Оновлюється автоматично після кожного `evaluate.py`. Таблиця збирається з `meta.yaml` і `results.json` прогонів, тож нотатку варто задавати через `--note` або в `meta.yaml` прогону: правка лише в CSV зникне після `python evaluation/results_table.py`. |
| [`evaluation/baseline/model_comparison.png`](evaluation/baseline/model_comparison.png) | Графік WER і CER базових моделей по наборах. |
| `evaluation/<прогін>/` | Окремий прогін: `results.json`, `predictions.csv`, `meta.yaml`, графіки. |

## Документація

| Файл | Що там |
|---|---|
| [`todo.md`](todo.md) | Поточні задачі. |
| [`datasets/info.yaml`](datasets/info.yaml) | Усі датасети: версія, ліцензія, посилання, години. |
| [`datasets/common-voice/README.md`](datasets/common-voice/README.md) | Common Voice: файли сплітів, як зібрано `train_validated.tsv`, що таке `test_cleaned.tsv`. |
| [`datasets/leak-clean/README.md`](datasets/leak-clean/README.md) | Перевірка датасетів на повтори й витоки між train, dev і test. |
| [`datasets/leak-clean/MANIFEST_SCRIPTS_REVIEW.md`](datasets/leak-clean/MANIFEST_SCRIPTS_REVIEW.md) | Аудит скриптів, що збирають спліти (2026-11-03). Частину знахідок уже виправлено. |
| [`REVIEW_FINDINGS.md`](REVIEW_FINDINGS.md) | Аудит коду (2026-11-03): помилки й вразливості. |
| [`pseudo-labelling/llm-as-a-judge/README.md`](pseudo-labelling/llm-as-a-judge/README.md) | Суддя якості міток: бекенди `llm` і `decisions`, змінні `.env`, оцінка псевдоміток і оригінальних міток датасетів. |
| [`analysis/language/README.md`](analysis/language/README.md) | Мова кліпів у розмічених датасетах за Whisper і MMS: запуск, файли, звіт. |
| [`VAST_AI.md`](VAST_AI.md) | Як запускати проєкт на vast.ai. |
| [`HF_UPLOAD.md`](HF_UPLOAD.md) | Як залити приватні дані на Hugging Face для vast.ai. |

## Структура

| Тека | Що там |
|---|---|
| `fine-tune/` | Навчання вчителя (`fine-tune.py`), студента (`fine-tune-student.py`) і ланцюжка кроків (`multi_step.py`), завантаження даних (`speech_data.py`, `dataset_roles.py`), чекпоінти. |
| `pseudo-labelling/` | Нарізка VOA, псевдомітки вчителя, суддя якості міток. |
| `decisions.py` | Бекенд судді на decision-моделі (Decisions API OpenRouter). |
| `analysis/` | Разові аналізи: `label-judge/` (оцінка оригінальних міток датасетів суддею), `language/` (мова кліпів датасетів за Whisper і MMS), графіки метаданих VOA. |
| `output/` | Не в git: метрики навчання (`fine-tune/<name>/version_N`), псевдомітки, конфіги кроків `multi-step`, звіт перевірки витоків. |
| `evaluation/` | `evaluate.py`, оцінка після навчання (`evaluate_training.py`), таблиця результатів (`results_table.py`), порівняння моделей (`batch-compare.py`). |
| `datasets/` | Спліти Common Voice, опис датасетів, перевірка на витоки. |
| `scripts/` | Збір метаданих VOA, визначення мови, аналіз токенів датасетів (`monitor/unique_tokens.py`), допоміжні скрипти для сервера й vast.ai. |
