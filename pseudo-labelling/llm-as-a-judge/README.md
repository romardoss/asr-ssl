# Суддя якості міток

Суддя ставить кожній транскрипції оцінку від 0 до 5 за шкалою з [`prompt.txt`](prompt.txt):

| Оцінка | Значення |
|---:|---|
| 0 | нісенітниця, не українське висловлювання |
| 1 | переважно зламано, зрозумілі лише фрагменти |
| 2 | зрозуміло, але багато лексичних, граматичних або морфологічних помилок |
| 3 | чітка українська з кількома помилками |
| 4 | добра українська з дрібними помилками |
| 5 | правильна мітка без помилок |

Оцінки використовуються для двох задач:

1. **Псевдомітки VOA** (`judge.py`): оцінки пишуться в `label_marks.csv` поруч із `labels.csv`. Студент навчається тільки на мітках з оцінками зі списку `accepted_mark`.
2. **Оригінальні мітки розмічених датасетів** (`analysis/label-judge/judge_labels.py`): разовий аналіз якості міток по кожному датасету й спліту. Опис нижче.

## Бекенди

Обидва бекенди працюють через OpenRouter з одним ключем `OPENROUTER_API_KEY` і з тією самою шкалою з `prompt.txt`.

| | `llm` (за замовчуванням) | `decisions` |
|---|---|---|
| Модель | чат-модель: `OPENROUTER_MODEL`, у пайплайні `pipeline.judge_model` | decision-модель: `OPENROUTER_DECISION_MODEL`, наприклад `typesafe/jev-1.14` |
| API | `/api/v1/chat/completions` | `/api/alpha/decisions` (Decisions API, альфа) |
| Запит | батч із 16 рядків `<index> <текст>`, відповідь рядками `<index> <оцінка>` | одна транскрипція на запит (`state`), одне питання типу `score` з рівнями 0–5 із `prompt.txt` |
| Що зберігається | `index,ID,mark` | `index,ID,mark,score,confidence`: `mark` — найімовірніший рівень, `score` — зважений за ймовірностями рівень, `confidence` — наскільки ймовірності зосереджені на одному рівні |
| Код | `judge.py` | [`decisions.py`](../../decisions.py) у корені репозиторію, його імпортують `judge.py`, `pipeline.py` і `judge_labels.py` |

Бекенд обирається змінною `USE_DECISION_INSTEAD_OF_LLM`: `true` вмикає `decisions`, `false` або відсутня змінна — `llm`. Параметр `--backend llm|decisions` у `judge.py`, `main.py judge` і `judge_labels.py` має пріоритет над змінною.

Налаштування в `.env` (зразок у [`.env.example`](../../.env.example)):

```
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=...                      # для llm
USE_DECISION_INSTEAD_OF_LLM=true
OPENROUTER_DECISION_MODEL=typesafe/jev-1.14
```

Особливості `decisions`:

1. Decision-модель не генерує текст, тому одна заміна моделі в `OPENROUTER_MODEL` не працює. Потрібен саме цей бекенд.
2. Один запит оцінює одну транскрипцію, тож для повного прогону варто збільшити `--workers` (наприклад, до 16).
3. Відповідь `refusal` вважається невдачею: запит повторюється `--retries` разів, а потім рядок лишається неоціненим до наступного запуску.
4. Документація Decisions API нічого не каже про українську мову. Перед повним прогоном перевірте модель на невеликій вибірці.
5. Якщо `label_marks.csv` уже існує зі старим заголовком `index,ID,mark`, дописування зберігає цей заголовок. Оцінки різних суддів при цьому опиняються в одному файлі, тож для нового судді краще почати новий файл.

## Псевдомітки VOA

```bash
python main.py judge --test                                  # пробна оцінка для ручної перевірки
python main.py judge --max-examples 32                       # бекенд з USE_DECISION_INSTEAD_OF_LLM
python main.py judge --backend decisions --workers 16        # примусово decision-модель
python main.py marks                                         # графік розподілу оцінок
```

### Пробна оцінка (`--test`)

Перед повним прогоном нового судді варто подивитися, як він оцінює:

1. Скрипт оцінює 100 випадкових рядків `labels.csv` (seed фіксований, тож вибірка щоразу та сама).
2. Якщо серед цих 100 рядків якась оцінка від 0 до 5 жодного разу не трапилася, він оцінює ще 100, разом 200.
3. Результат пишеться в `judge_test-<модель>-<дата-час>.csv` поруч із `labels.csv`, відсортований за оцінкою. Колонки: `mark`, для `decisions` ще `score` і `confidence`, `text`, `ID`, `index`. Файл у кодуванні UTF-8 з BOM, тож Excel і LibreOffice відкривають українські тексти без питань про кодування.
4. Скрипт виводить, скільки разів трапилася кожна оцінка і які оцінки не трапилися зовсім.
5. `label_marks.csv` не змінюється, а кожен запуск пише новий файл.

У пайплайні суддю запускає етап `pipeline.clean`. Якщо `USE_DECISION_INSTEAD_OF_LLM=true`, пайплайн вимагає `OPENROUTER_DECISION_MODEL`, а `pipeline.judge_model` може лишатися порожнім. `--workers` і `--retries` беруться з `pipeline.judge_workers` і `pipeline.judge_retries`. Перерваний запуск продовжується з тих рядків, яких ще немає в `label_marks.csv`.

## Оригінальні мітки датасетів

`analysis/label-judge/judge_labels.py` оцінює мітки всіх сплітів (train/dev/test) усіх розмічених датасетів із `fine-tune/dataset_roles.py`, які є на диску. Скрипт читає тільки тексти, аудіо не декодує. За замовчуванням оцінюються всі транскрипції, `--per-split N` бере випадкову вибірку.

Кожен запуск отримує окрему папку, названу за моделлю, тож звіти різних суддів не перезаписуються:

```
analysis/label-judge/<модель>-NN/
    report-<модель>.md          таблиця по dataset/split, графіки, найгірше оцінені приклади
    plots/mark_distribution.png розподіл оцінок 0–5
    plots/mean_mark.png         середня оцінка з 95% довірчим інтервалом
    plots/poor_share.png        частка міток з оцінками 0–2
    sample.csv                  транскрипції, надіслані судді
    marks.csv                   оцінки (для decisions ще score і confidence)
    run.json                    модель, бекенд, хеш промпту, seed, розмір вибірки
```

```bash
python analysis/label-judge/judge_labels.py --backend decisions --per-split 200 --workers 16   # пробний прогін
python analysis/label-judge/judge_labels.py --backend decisions --workers 16                   # усі мітки
python analysis/label-judge/judge_labels.py --model google/gemini-2.5-pro                      # інший чат-суддя
python analysis/label-judge/judge_labels.py --resume analysis/label-judge/<модель>-01          # продовжити
python analysis/label-judge/judge_labels.py --report-only analysis/label-judge/<модель>-01     # перебудувати звіт
```

`--resume` продовжує запуск із тим самим бекендом і моделлю, що записані в `run.json`.
