# Перевірка датасетів на повтори й витоки

Ця тека для перевірки того, чи не повторюються дані всередині датасетів і між ними. Найважливіше питання — чи не потрапили кліпи, тексти або диктори з dev і test у train.

Чому це важливо: у Common Voice такий витік уже знайдено.

1. Усі 10 955 кліпів `common-voice/raw/dev.tsv` є в `train_validated.tsv`.
2. У 4 992 з 10 956 тестових кліпів (45,6 %) речення трапляється в train.
3. 3 932 тестові кліпи (36 %) записали диктори, які є в train.

Ці цифри стосуються початкового `train_validated.tsv`. Скрипт `clean_common_voice_train.py` з цієї теки прибрав з train лише кліпи dev і test: лишилося 62 460 рядків з 73 415 і 376 дикторів. Речення й дикторів, спільних із dev і test, свідомо лишено в train. Мета `train_validated` — навчатися на всіх валідованих даних, а офіційний `train.tsv` з такими правилами значно менший.

Через це частина test досі має дикторів (3 932 кліпи) або речення (5 001 кліп після нормалізації) з train, і WER на `common_voice` трохи оптимістичний. Тому той самий скрипт пише `common-voice/raw/test_cleaned.tsv`: 5 703 тестові кліпи від 503 дикторів, у яких ні диктора, ні нормалізованого речення немає в train. Він не бере участі в навчанні й оцінці. Скрипт `score_common_voice_cleaned.py` окремо порівнює на ньому WER з повним test за наявними `predictions.csv`. Різниця виявилася в межах ±0.3 п.п., тобто витік WER помітно не завищує. Подробиці й таблиця — в `datasets/common-voice/README.md`.

Попередня версія скрипта прибирала з train ще й спільні речення та дикторів (лишалося 38 849 рядків і 74 диктори). `MANIFEST_SCRIPTS_REVIEW.md` описує саме її.

Інші датасети ще не перевірялися. Аудіо лежить лише на сервері (`paths.*` у `config.yaml`), тому все нижче потрібно запускати там, з кореня репозиторію.

---

## Як запустити перевірку

Одна команда на сервері, з кореня репозиторію (на Vast додати `ASR_CONFIG=config.vast.yaml`):

```bash
env/bin/python datasets/leak-clean/check_leaks.py
env/bin/python datasets/leak-clean/check_leaks.py --datasets common_voice cv10_test   # лише частина датасетів
```

Скрипт проходить три етапи, а потім пише **`output/leak-clean/report.md`**. Це головний результат: його можна читати у VS Code чи будь-якому переглядачі Markdown, графіки вбудовано. Тека `output/` ігнорується git.

Шляхи беруться з конфігу (`config.yaml` або файл з `ASR_CONFIG`). Тексти порівнюються після нормалізації: регістр, пунктуація й варіанти апострофа не враховуються. Нормалізація та сама, що й у WER (`fine-tune/fine-tune.py`), і задана в `leak_common.py`. Датасет, якого немає на диску, не зупиняє запуск: він потрапляє в розділ «Не перевірено» звіту.

### Що є у звіті

1. **Головне.** Скільки кліпів перевірено, які датасети не знайдено, найбільший витік тексту в dev/test, кількість повторів, аудіо без метаданих.
2. **Що перевірено.** Кількість кліпів у кожному спліті та список того, що не вдалося перевірити, з шляхами.
3. **Тексти dev і test, що є в train.** Для кожного dev і test: скільки кліпів мають текст із будь-якого train — усі тексти й тексти від 4 слів, — і з яких train-сплітів ці тексти. Нижче наведено приклади найчастіших таких текстів. Короткі фрази на кшталт «так» чи «добрий день» збігаються майже неминуче, тому головна колонка — «від 4 слів».
4. **Повтори текстів.** Скільки текстів повторюється між датасетами, між сплітами одного датасету і всередині спліту, з прикладами текстів, спільних для різних датасетів.
5. **Аудіо й метадані без пари.** Аудіо без метаданих, метадані без аудіо, аудіо без тексту, однакові імена в архіві OpenTTS.
6. **Графіки** і **перелік файлів**.

### Файли в `output/leak-clean/`

| Файл | Що там |
|---|---|
| `report.md` | Звіт, описаний вище. |
| `metadata.parquet` | Один рядок на кліп: датасет, спліт, ID, ім'я аудіо, точне місце (`location`: файл, рядок parquet/Arrow або член zip-архіву), текст і нормалізований текст. |
| `coverage.csv` | Усі перевірки аудіо й метаданих, включно з тими, що нічого не знайшли (`count = 0`). |
| `duplicates.csv` | Кожен кліп, текст якого трапляється більше одного разу: номер групи, тип повтору (`within_split`, `between_splits`, `between_datasets`), місце й усі спліти з цим текстом. |
| `units.csv` | Кількість кліпів у кожному спліті. |
| `plots/overlap_matrix.png` | Скільки різних текстів має кожна пара сплітів (по діагоналі — повтори всередині спліту). |
| `plots/eval_leaks.png` | Частка кожного dev і test, текст якої є в train: усі тексти й тексти від 4 слів. |
| `plots/coverage.png` | Аудіо й метадані без пари. Не створюється, якщо нічого не знайдено. |

### Етапи окремо

`check_leaks.py` викликає три скрипти. Їх можна запускати й окремо, наприклад щоб перебудувати графіки без повторного збору метаданих:

1. **`collect_metadata.py`** збирає метадані всіх датасетів (`common_voice`, `fleurs`, `opentts`, `tg_voices`, `ukr_dialects`, `cv10_test`, `voa`) у `metadata.parquet` і пише `coverage.csv`. Аудіо не декодується, перевіряється лише його наявність. Для VOA текстом слайсу вважається псевдомітка з `student.labels`. Тому без `labels.csv` вчителя VOA позначиться як не знайдений.
2. **`find_overlaps.py`** шукає повтори текстів між train, dev і test і пише `duplicates.csv` та `units.csv`. Параметр `--min-words 4` відсіює короткі фрази. Звіт цього не потребує, бо сам показує обидва варіанти.
3. **`plot_overlaps.py`** будує графіки з цих CSV.

### Окремо для Common Voice

1. `clean_common_voice_train.py` прибирає з `train_validated.tsv` кліпи dev і test і пише `test_cleaned.tsv`.
2. `score_common_voice_cleaned.py` порівнює WER на test і `test_cleaned.tsv` за наявними `predictions.csv`.

Обидва описано в `datasets/common-voice/README.md`.

### Результат локального прогону (2026-11-07)

Локально є лише TSV Common Voice, без аудіо. Тому решта датасетів і аудіо потрапили в «Не перевірено».

1. `common_voice/dev`: 2 984 з 10 955 кліпів (27,2 %) мають текст від 4 слів, що є в train.
2. `common_voice/test`: 4 408 з 10 956 кліпів (40,2 %) мають такий текст.

Обидва витоки очікувані: `train_validated.tsv` свідомо зберігає спільні речення (див. початок README). Перевірку решти датасетів ще треба запустити на сервері (`todo.md`, пункт 6).

### Обмеження

Скрипти порівнюють лише нормалізований текст. Вони не перевіряють:

1. хеш аудіо, ім'я файлу й диктора. Для цього є ручна перевірка в розділах 4–5 нижче. Вона потрібна, зокрема, щоб звірити `cv10_test` із train Common Voice (`todo.md`, пункт 3);
2. чи містять псевдомітки VOA речення з dev і test Common Voice як підрядок (розділ 3, крок 6). Повний збіг тексту довгого слайсу VOA з реченням майже неможливий, тож звичайне порівняння тут нічого не покаже.

---

## 1. Що вважати повтором

Повтори бувають різного рівня, від найочевиднішого до найтоншого:

1. **Той самий файл.** Збігається `path` або ID кліпу.
2. **Ті самі байти аудіо.** Збігається SHA-256 закодованого файлу. Ловить копії, перейменовані файли й той самий кліп у двох датасетах.
3. **Той самий звук після перекодування.** Байти різні (інший кодек чи бітрейт), а запис той самий. Ловиться акустичним відбитком, наприклад `fpcalc` / Chromaprint. Це необов'язковий і дорожчий крок.
4. **Той самий текст.** Збігається нормалізований текст. Для прочитаного мовлення (Common Voice, FLEURS, OpenTTS) це означає, що модель могла запам'ятати речення.
5. **Той самий диктор.** Збігається ідентифікатор диктора. Модель звикає до голосу, і WER стає оптимістичним.

Рівні 1–3 означають, що той самий запис є і в train, і в оцінці. Такі кліпи потрібно прибирати обов'язково. Рівні 4–5 — м'якший витік. Офіційні спліти Common Voice його не допускають, тож для порівняння з опублікованими результатами його теж варто прибирати.

---

## 2. Що де лежить і які ключі є

Формати взято з `fine-tune/speech_data.py` і `fine-tune/common_voice.py`. Назви колонок, позначені «перевірити», варто підтвердити через `pq.read_schema(...)` або `batch.schema`.

| Датасет (`config.yaml`) | Спліти | Де лежить | Текст | Диктор | Аудіо |
|---|---|---|---|---|---|
| `common_voice` | train / dev / test | `datasets/common-voice/raw/{train_validated,dev,test}.tsv` і `paths.common_voice/clips/` | `sentence` | `client_id` | файл `path` |
| `fleurs` | train / validation (= dev) / test | `paths.fleurs/<split>/` (Arrow, список файлів у `state.json`) | `transcription` | немає (`id` означає речення, а не кліп) | `audio.bytes` |
| `opentts` | train | `paths.opentts/voices/<voice>/train.jsonl` і `data/<voice>.zip` | `text` | голос `<voice>` | член zip-архіву |
| `tg_voices` | train | `paths.tg_voices/data/train-*.parquet` | `transcription` | перевірити | `audio.bytes` |
| `ukr_dialects` | train / validation (= dev) / test | `paths.ukr_dialects/data/{train,validation,test}-*.parquet` | `text` | перевірити | `audio.bytes` |
| `cv10_test` | test (файли називаються `train-*`) | `paths.cv10_test/data/train-*.parquet` | `transcription` | немає | `audio.bytes`, `audio.path` (перевірити, чи це оригінальне ім'я `common_voice_uk_*.mp3`) |
| VOA (student) | train | `paths`/`student.voa` `/data/train-*.parquet`, мітки в `output/pseudo-labelling/<run>/labels.csv` | `text` (псевдомітка) | немає | `audio.bytes` усього файлу |

Пари, які варто перевірити насамперед:

1. `common_voice` train проти `common_voice` dev і test: знайдено витік, див. вище.
2. `common_voice` train проти `cv10_test`. Тестові кліпи Common Voice 10 майже напевно ввійшли до `validated` пізніших версій, тобто й до `train_validated.tsv` версії 27. Якщо це так, `cv10_test` повністю або частково є train-даними.
3. Усі train-набори (`opentts`, `tg_voices`, `fleurs`, `ukr_dialects`) проти всіх dev і test за текстом. Наприклад, OpenTTS і TG Voices можуть озвучувати ті самі речення, що є в Common Voice.
4. `ukr_dialects` і `fleurs` всередині себе: train проти validation і test.
5. dev проти test для кожного датасету.
6. Псевдомітки VOA проти Common Voice dev і test, бо student оцінюється саме на них. Повних збігів тексту тут майже не буде, бо слайси VOA довгі. Корисніше шукати, чи містить текст VOA тестове речення (див. крок 6).

---

## 3. Порядок перевірки

1. **Зібрати маніфест.** Пройти всі датасети й записати для кожного кліпу один рядок у `datasets/leak-clean/manifest.parquet` з такими полями:
   - `dataset`, `split`, `id`;
   - `text` і `text_norm`;
   - `speaker`;
   - `audio_sha256`, `audio_name`.

   Аудіо в маніфест не копіюється, зберігається лише хеш. Код див. у розділі 4.
2. **Повтори всередині кожного спліту.** Це однакові `audio_sha256` або `text_norm` у межах однієї пари `(dataset, split)`. Сам по собі такий повтор не є витоком, але він дублює дані й зсуває метрики.
3. **Витоки між сплітами.** Для пар `dev → train`, `test → train` і `test → dev` порахувати, скільки кліпів цільового спліту мають збіг у еталонному. Рахувати окремо за кожним ключем і з розбивкою за датасетом-джерелом. Код див. у розділі 5.
4. **Акустичні дублікати (необов'язково).** Для кліпів, які пройшли кроки 2–3 без збігу за хешем, обчислити Chromaprint (`fpcalc -raw`) і шукати близькі відбитки між train і test. Це потрібно, якщо є підозра, що датасет — перекодована копія іншого. Найімовірніший кандидат — `cv10_test` проти Common Voice.
5. **Короткі тексти.** Збіг тексту на 1–3 слова (наприклад, «так», «добрий день») майже неминучий і не означає витоку. Звітувати збіги тексту окремо для речень із щонайменше 4 слів.
6. **Псевдомітки VOA.** Для речень Common Voice dev і test завдовжки щонайменше 6 слів перевірити, чи трапляються вони як підрядок у нормалізованих текстах `labels.csv`.
7. **Записати рішення.** Скласти `datasets/leak-clean/drop_from_train.csv` з колонками `dataset`, `id`, `reason` (`audio`, `name`, `text`, `speaker`). Потім навчити `fine-tune/speech_data.py` і `fine-tune/common_voice.py` відкидати ці рядки з train.
8. **Захист на майбутнє.** Додати в `load_configured_splits` перевірку, яка зупиняє запуск з помилкою, якщо train перетинається з dev або test за `id`, `audio_sha256` чи `audio_name`.

---

## 4. Збір маніфесту

Запускати на сервері з кореня репозиторію (`env/bin/python collect_manifest.py`, назва файлу довільна). Шляхи беруться з `config.yaml` тим самим кодом, що й у навчанні. Нормалізація тексту та сама, що й для WER у `fine-tune/fine-tune.py`, плюс зведення всіх варіантів апострофа до `'`.

```python
import csv
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

sys.path.insert(0, "fine-tune")
from settings import load_settings

cfg = load_settings()
OUT = Path("datasets/leak-clean/manifest.parquet")
LABELS = Path("output/pseudo-labelling/model-whisper-uk-epoch00-01/labels.csv")  # потрібний запуск

PUNCT = re.compile(r"[^\w\s']+")
SPACES = re.compile(r"\s+")


def norm(text) -> str:
    text = "" if text is None else str(text).lower()
    for mark in ("’", "ʼ", "‘", "`"):
        text = text.replace(mark, "'")
    return SPACES.sub(" ", PUNCT.sub(" ", text)).strip()


rows = []


def add(dataset, split, item_id, text, audio=None, speaker=None, audio_name=None):
    rows.append(
        {
            "dataset": dataset,
            "split": split,
            "id": str(item_id),
            "text": text,
            "text_norm": norm(text),
            "speaker": None if speaker is None else str(speaker),
            "audio_sha256": hashlib.sha256(audio).hexdigest() if audio else None,
            "audio_name": None if not audio_name else Path(str(audio_name)).name,
        }
    )


# Common Voice: ті самі TSV, що читає fine-tune/common_voice.py
clips = cfg.common_voice / "clips"
for split, name in {"train": "train_validated.tsv", "dev": "dev.tsv", "test": "test.tsv"}.items():
    frame = pd.read_csv(f"datasets/common-voice/raw/{name}", sep="\t", dtype=str, quoting=csv.QUOTE_NONE)
    for row in frame.itertuples(index=False):
        clip = clips / row.path
        audio = clip.read_bytes() if clip.is_file() else None
        add("common_voice", split, row.path, row.sentence, audio, row.client_id, row.path)

# FLEURS: Arrow-потоки, перелічені в state.json
for folder, split in (("train", "train"), ("validation", "dev"), ("test", "test")):
    directory = cfg.fleurs / folder
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    for item in state["_data_files"]:
        for batch in ipc.open_stream(directory / item["filename"]):
            for record in batch.to_pylist():
                audio = record.get("audio") or {}
                item_id = f"{record.get('id')}/{audio.get('path')}"
                add("fleurs", split, item_id, record.get("transcription"), audio.get("bytes"), None, audio.get("path"))

# OpenTTS: jsonl + zip на кожен голос
for voice_dir in sorted(path for path in (cfg.opentts / "voices").iterdir() if path.is_dir()):
    voice = voice_dir.name
    with zipfile.ZipFile(cfg.opentts / "data" / f"{voice}.zip") as archive:
        members = {Path(name).name: name for name in archive.namelist() if not name.endswith("/")}
        for line in (voice_dir / "train.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            name = Path(str(record.get("audio") or "")).name
            member = members.get(name)
            audio = archive.read(member) if member else None
            add("opentts", "train", f"{voice}/{name}", record.get("text"), audio, voice, name)


# Parquet-датасети: TG Voices, Ukrainian Dialects, CV10 test
def parquet(dataset, root, folder, split, text_column, speaker_column=None):
    for path in sorted((root / "data").glob(f"{folder}-*.parquet")):
        parquet_file = pq.ParquetFile(path)
        columns = [text_column, "audio"] + ([speaker_column] if speaker_column else [])
        for group in range(parquet_file.num_row_groups):
            records = parquet_file.read_row_group(group, columns=columns).to_pylist()
            for offset, record in enumerate(records):
                audio = record.get("audio") or {}
                speaker = record.get(speaker_column) if speaker_column else None
                add(
                    dataset,
                    split,
                    f"{path.name}:{group}:{offset}",
                    record.get(text_column),
                    audio.get("bytes"),
                    speaker,
                    audio.get("path"),
                )


parquet("tg_voices", cfg.tg_voices, "train", "train", "transcription")
for folder, split in (("train", "train"), ("validation", "dev"), ("test", "test")):
    parquet("ukr_dialects", cfg.ukr_dialects, folder, split, "text")
parquet("cv10_test", cfg.cv10_test, "train", "test", "transcription")

# VOA: лише тексти псевдоміток (аудіо тут — слайси довгих файлів)
if LABELS.is_file():
    for row in pd.read_csv(LABELS, dtype=str, keep_default_na=False).itertuples(index=False):
        add("voa", "train", row.ID, row.text, None, None, None)

OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_parquet(OUT, index=False)
print(f"{len(rows)} rows -> {OUT}")
```

Примітки до коду:

1. Якщо у `tg_voices` або `ukr_dialects` є колонка диктора, передайте її назву як `speaker_column`. Перед цим перевірте схему: `pq.read_schema(next((root / "data").glob("*.parquet")))`.
2. `audio_name` має сенс лише там, де збережено оригінальне ім'я файлу. Для Common Voice і `cv10_test` це `common_voice_uk_*.mp3`. Для інших parquet-датасетів ім'я може бути порожнім або однаковим, тож у розділі 5 збіги за `audio_name` рахуються лише для імен `common_voice_*`.
3. Маніфест займає кілька десятків мегабайт. Комітити його в git не потрібно.

---

## 5. Аналіз маніфесту

```python
import pandas as pd

manifest = pd.read_parquet("datasets/leak-clean/manifest.parquet")
manifest = manifest[manifest["text_norm"] != ""]
manifest.loc[~manifest["audio_name"].fillna("").str.startswith("common_voice_"), "audio_name"] = None
manifest["words"] = manifest["text_norm"].str.split().str.len()

# 1. Повтори всередині спліту
for key in ("id", "audio_sha256", "text_norm"):
    data = manifest.dropna(subset=[key])
    repeated = data[data.duplicated(["dataset", "split", key], keep=False)]
    print(f"\nповтори за {key}:")
    print(repeated.groupby(["dataset", "split"]).size().to_string())


# 2. Витоки між сплітами: скільки кліпів цільового спліту мають збіг в еталонному
def leaks(target_role, reference_role, key, same_dataset=False, min_words=1):
    target = manifest[(manifest["split"] == target_role) & (manifest["words"] >= min_words)].dropna(subset=[key])
    reference = manifest[manifest["split"] == reference_role].dropna(subset=[key])
    hits = target.merge(
        reference[["dataset", key]].drop_duplicates(),
        on=key,
        suffixes=("", "_reference"),
    )
    if same_dataset:
        hits = hits[hits["dataset"] == hits["dataset_reference"]]
    counts = hits.groupby(["dataset", "dataset_reference"])["id"].nunique()
    totals = target.groupby("dataset")["id"].nunique()
    return counts.rename("clips").reset_index().assign(
        total=lambda frame: frame["dataset"].map(totals),
        share=lambda frame: (100 * frame["clips"] / frame["total"]).round(1),
    )


checks = (
    ("audio_sha256", False, 1),  # той самий запис
    ("audio_name", False, 1),    # той самий файл Common Voice
    ("text_norm", False, 4),     # те саме речення (від 4 слів)
    ("speaker", True, 1),        # той самий диктор у межах датасету
)
for target_role, reference_role in (("dev", "train"), ("test", "train"), ("test", "dev")):
    for key, same_dataset, min_words in checks:
        table = leaks(target_role, reference_role, key, same_dataset, min_words)
        if not table.empty:
            print(f"\n{target_role} -> {reference_role}, ключ {key}:")
            print(table.to_string(index=False))
```

Як читати результат: рядок `dataset=cv10_test, dataset_reference=common_voice, clips=3204, share=100.0` для ключа `audio_sha256` або `audio_name` означає, що весь `cv10_test` уже є в train Common Voice.

---

## 6. Що робити зі знайденим

1. **Прибирати з train, а не з test.** Тест має лишатися незмінним, щоб результати були порівнянні з опублікованими й з попередніми запусками.
2. **Збіги за `audio_sha256` і `audio_name`** прибирати завжди: це той самий запис.
3. **Збіги за диктором** у Common Voice лишати в train і перевіряти скриптом `score_common_voice_cleaned.py` (див. початок README). Офіційні спліти CV розділені за дикторами, але таке розділення зменшує train майже вдвічі.
4. **Збіги за текстом** для прочитаного мовлення прибирати, якщо тексти довші за 3 слова. Якщо втрата train-даних завелика, залишити їх, але окремо звітувати WER на підмножині test без збігів.
5. **Якщо цілий набір виявився train-даними** (наприклад, `cv10_test`), або прибрати відповідні кліпи з train, або вилучити цей набір з `data.test` і явно позначити це в результатах.
6. **Після очищення** переоцінити всі моделі в `evaluation/`, включно з `baseline/`, бо попередні числа зсунуті.
