# Мова кліпів у датасетах

Перевіряє, якою мовою говорять у кліпах усіх розмічених датасетів (`common_voice`, `fleurs`, `opentts`, `tg_voices`, `ukr_dialects`, `cv10_test`). Логіка та сама, що відбирає українські кліпи VOA на кроці `pipeline.language`:

1. **Whisper** — як у `scripts/monitor/meta_collector.py`: faster-whisper `medium`, перші 120 с, VAD-фільтр. Потрібна лише мова, тому транскрипт не декодується.
2. **MMS** — як у `scripts/monitor/predict_languages.py`: `pipeline.mms_model` (`facebook/mms-lid-256`), перші 300 с, батчі за тривалістю, fp16 за `pipeline.mms_fp16`.
3. **Рішення** — як у `scripts/monitor/split_language_meta.py`: кліп український, лише якщо Whisper каже `uk`, а найбільша ймовірність MMS — `ukr`.

Кліпи читаються через `fine-tune/speech_data.py`, тож це ті самі кліпи й ID, що в навчанні та оцінці. Нічого не відкидається: звіт лише показує, скільки кліпів моделі не чують як українські.

## Запуск

На сервері з GPU, з кореня репозиторію (на Vast додати `ASR_CONFIG=config.vast.yaml`):

```bash
env/bin/python analysis/language/detect_languages.py                      # усі кліпи
env/bin/python analysis/language/detect_languages.py --per-split 1000     # 1000 випадкових кліпів на спліт
env/bin/python analysis/language/detect_languages.py --datasets cv10_test fleurs --splits test
env/bin/python analysis/language/language_report.py                       # лише перебудувати звіт
```

1. GPU і CUDA-бібліотеки для faster-whisper беруться так само, як у `meta_collector.py`: `gpus` з конфігу і `.env`.
2. `--per-split N` бере ту саму випадкову вибірку, що й `analysis/label-judge/judge_labels.py` з тим самим N і `--seed`. Так мову можна зіставити з оцінками судді.
3. Результати дописуються в `predictions.csv` після кожної порції з `--chunk` кліпів (за замовчуванням 256). Перерваний запуск продовжує з місця зупинки. Кліпи з помилкою пробує знову.
4. Наприкінці `language_report.py` будує звіт; `--no-report` це вимикає.

## Файли

| Файл | Що там |
|---|---|
| `detect_languages.py` | Проганяє Whisper і MMS, пише `predictions.csv`. |
| `language_report.py` | Будує `report.md` і `plots/` з `predictions.csv`. Моделей і аудіо не потребує. |
| `predictions.csv` | Один рядок на кліп: `dataset`, `split`, `id`, `seconds`; Whisper: `whisper_language`, `whisper_probability`, `whisper_uk`, `whisper_ru`; MMS: `mms_language`, `mms_probability`, `mms_ukr`, `mms_rus`, `mms_top` (топ-3); `ukrainian` (обидві моделі), `error`, `text`. |
| `report.md` | Звіт: таблиця за сплітами, графіки, приклади кліпів, які жодна модель не чує українськими, помилки. |
| `plots/agreement.png` | Частка кліпів, які обидві моделі, лише одна або жодна вважають українськими. |
| `plots/other_whisper.png`, `plots/other_mms.png` | Які інші мови чує кожна модель, у відсотках від спліту. |
| `plots/confidence.png` | Розподіл меншої з двох ймовірностей української. |
| `plots/duration.png` | Частка не українських хоча б за однією моделлю залежно від тривалості кліпу. |

## Як читати

1. **Розбіжність моделей** («лише Whisper» або «лише MMS») ще не означає іншої мови. На коротких кліпах і діалектах визначення мови менш надійне, тому варто дивитися графік тривалості.
2. **«Жодна»** — кліпи, які варто прослухати першими: інша мова, суржик, тиша або шум.
3. Ймовірності Whisper `whisper_uk` і `whisper_ru` є, лише якщо встановлена версія faster-whisper повертає `all_language_probs`. Інакше відома лише ймовірність визначеної мови.
