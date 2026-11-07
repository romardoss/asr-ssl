# Common Voice (українська)

Mozilla Common Voice — прочитане мовлення: волонтери читають короткі речення, інші волонтери голосують, чи запис правильний.

| | |
|---|---|
| Версія | 27.0, мова `uk` |
| Ліцензія | CC0-1.0 |
| Джерело | <https://mozilladatacollective.com/datasets/cmu5wgebl00d0nq07ktip1i6c> |
| Обсяг | 121,37 год, з них 108,75 год з мітками (`datasets/info.yaml`) |
| Аудіо | `<paths.common_voice>/clips/*.mp3`, лише на сервері |
| Спліти | `raw/*.tsv` у git, шлях задає `paths.common_voice_splits` |

Код читає спліти в `fine-tune/common_voice.py` (`SPLIT_FILES`), а `fine-tune/speech_data.py` додає їх до `data.train`, `data.dev` і `data.test` з `config.yaml`.

## Файли в `raw/`

| Файл | Роль у коді | Рядків | Дикторів | Звідки |
|---|---|---|---|---|
| `train_validated.tsv` | `train` | 62 460 | 376 | зібрано в проєкті, див. нижче |
| `dev.tsv` | `dev` | 10 955 | 140 | офіційний `dev.tsv` |
| `test.tsv` | `test` | 10 956 | 938 | офіційний `test.tsv` |
| `test_cleaned.tsv` | не використовується в навчанні й оцінці | 5 703 | 503 | зібрано в проєкті, див. нижче |
| `unlabelled.tsv` | `unlabelled` | 9 787 | 696 | офіційний `other.tsv` |

Колонки однакові в усіх файлах, бо це рядки з TSV Common Voice: `client_id`, `path`, `sentence_id`, `sentence`, `sentence_domain`, `up_votes`, `down_votes`, `age`, `gender`, `accents`, `variant`, `locale`, `segment`. Код бере лише `path` (ім'я файлу в `clips/`) і `sentence`.

TSV читаються з `quoting=csv.QUOTE_NONE`: один рядок — один кліп, а лапки в реченні є частиною тексту.

`unlabelled.tsv` має тексти, хоча назва каже інше. Це `other.tsv`: записи, які ще не набрали достатньо голосів або були відхилені. У навчанні він не використовується.

## `train_validated.tsv`

Офіційний `train.tsv` Common Voice значно менший за обсяг доступних валідованих даних. Common Voice будує спліти так, щоб диктори й речення не перетиналися між train, dev і test. Через це з train випадає все, що записали диктори dev і test, і все, де звучать речення з dev і test.

`train_validated.tsv` збирає всі валідовані записи, крім dev і test:

1. взято `validated.tsv` (усі записи, що пройшли голосування);
2. прибрано кліпи `test.tsv`, так вийшло 73 415 рядків;
3. скриптом `datasets/leak-clean/clean_common_voice_train.py` прибрано 10 955 кліпів `dev.tsv`, так вийшло 62 460 рядків.

Диктори й речення, спільні з dev і test, **свідомо лишено** в train. Інакше лишилося б лише 38 849 рядків і 74 диктори замість 376: різноманіття голосів упало б уп'ятеро.

Ціна цього рішення — частина `test.tsv` перетинається з train:
1. 3 932 тестові кліпи записали диктори, які є в train;
2. 5 001 тестовий кліп має речення, що є в train (після нормалізації тексту, як у WER).

Чи завищує це WER, перевірено за допомогою `test_cleaned.tsv`.

## `test_cleaned.tsv`

Це частина `test.tsv`, у якій ні диктора (`client_id`), ні нормалізованого речення немає в `train_validated.tsv`: 5 703 кліпи від 503 дикторів. Нормалізація та сама, що в `normalize_text` у `fine-tune/fine-tune.py`: нижній регістр, без пунктуації, `’` замінено на `'`.

Файл пише той самий `clean_common_voice_train.py`. З dev у `test_cleaned.tsv` немає ні дикторів, ні речень, бо офіційні dev і test Common Voice між собою не перетинаються. Тому він чистий і для моделей, які навчалися на старому train разом із кліпами dev.

`test_cleaned.tsv` **не входить** у навчання, `evaluate.py` і `combined_*`. Це закритий експеримент. Скрипт `datasets/leak-clean/score_common_voice_cleaned.py` порівнює на ньому WER з повним test. Нічого заново не розпізнається: кліпи `test_cleaned.tsv` є підмножиною test, тож гіпотези беруться з `predictions.csv` кожного прогону.

### Результат (2026-11-07)

WER, %:

| Прогін | test | test_cleaned | різниця |
|---|---|---|---|
| original-whisper-medium | 20.92 | 20.61 | −0.31 |
| common-voice-only-baseline | 17.40 | 17.39 | −0.01 |
| whisper-uk-epoch00-01 | 17.64 | 17.37 | −0.27 |
| whisper-uk-epoch01-01 | 17.54 | 17.26 | −0.28 |
| student_whisper-commonvoice-based-epoch01-01 | 18.02 | 17.88 | −0.14 |
| student_whisper-commonvoice-based-mark5-epoch01-01 | 17.56 | 17.67 | +0.12 |

Висновок: спільні з train диктори й речення WER помітно не завищують.
1. Оригінальний Whisper ніколи не бачив цього train, але на чистій частині й у нього WER нижчий на 0.31. Отже, сама ця частина трохи легша.
2. У дотренованих моделей різниця така сама або менша.
3. CER на чистій частині, навпаки, на 0.2–0.4 вищий у всіх моделей.

## Як перезібрати й перевірити

З кореня репозиторію:

```bash
# прибрати кліпи dev і test з train і перезаписати test_cleaned.tsv
python datasets/leak-clean/clean_common_voice_train.py --dry-run
python datasets/leak-clean/clean_common_voice_train.py

# порівняти WER на test і test_cleaned для всіх прогонів в evaluation/
python datasets/leak-clean/score_common_voice_cleaned.py
python datasets/leak-clean/score_common_voice_cleaned.py evaluation/baseline --csv cleaned_scores.csv
```

Повторний запуск `clean_common_voice_train.py` нічого не прибирає з train і пише той самий `test_cleaned.tsv`. Якщо `train_validated.tsv` змінився, скрипт треба перезапустити: інакше `test_cleaned.tsv` застаріє.
