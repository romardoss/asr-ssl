# Аудит репозиторію asr_ssl: помилки, неточності, вразливості

Дата: 2026-11-03. Це список знахідок. Код змінено лише в одному місці: `fine-tune/settings.py` і `pseudo-labelling/settings.py` тепер вимагають рівно одну GPU. Помилки, які виникали лише при двох GPU, через це прибрано зі списку.

Пункти 1–2, 3, 7 і 8 перевірено на даних, що є в репозиторії: TSV Common Voice, `ukrainian_30s.csv`, `label_marks.csv`, `evaluation/.../config.yaml`. Решту знайдено читанням коду.

---

## A. Критичне: метрики або навчання неправильні

1. **Dev Common Voice повністю входить у train.**
   - Де: `datasets/common-voice/raw/dev.tsv`, `datasets/common-voice/raw/train_validated.tsv`
   - Факт: усі 10 955 кліпів `dev.tsv` є в `train_validated.tsv` (з 73 415 рядків train).
   - Наслідок: у `fine-tune/fine-tune.py` dev-WER (`val_wer`) рахується на кліпах, на яких модель навчається. Він не показує, наскільки модель узагальнює, і за ним не можна вибирати чекпоінт. Для student це не проблема, бо student навчається на VOA.
   - **Як вирішити:**
     1. Перебудувати `train_validated.tsv` як `validated − dev − test` за колонкою `path`. Або фільтрувати в `fine-tune/common_voice.py` (`load_split`): для `train` відкидати шляхи, які є в `dev.tsv` і `test.tsv`.
     2. Додати в `load_configured_splits` (`fine-tune/speech_data.py`) перевірку, яка зупиняє запуск з помилкою, якщо train перетинається з dev або test за `example_id`.
     3. Після виправлення перезапустити fine-tune: усі наявні значення `val_*` недійсні.
   - **Статус: виправлено в даних.** `datasets/leak-clean/clean_common_voice_train.py` прибрав з train кліпи dev. Перевірки на перетин у коді завантаження ще немає, а моделі ще не перенавчені.

2. **Тест Common Voice перетинається з train за текстом і за дикторами.**
   - Де: `datasets/common-voice/raw/test.tsv` порівняно з `train_validated.tsv`
   - Факт: у 4 992 з 10 956 тестових кліпів (45,6 %) речення трапляється в train. 3 932 тестові кліпи (36 %) записали диктори, які є в train. Між dev і test перетинів немає.
   - Наслідок: test-WER на `common_voice` занижений, бо модель запам'ятовує речення й голоси. В офіційних спліттах CV диктори й речення не перетинаються.
   - **Як вирішити:**
     1. Викинути з train усі кліпи, у яких `sentence` або `client_id` трапляється в test чи dev. Простіший варіант — взяти офіційний `train.tsv` CV, де такого перетину вже немає. Обидва варіанти зменшують train, тож варто порахувати, скільки годин втрачається.
     2. Додати перевірку на перетин дикторів і речень у тому самому місці, що й у пункті 1.
     3. Якщо train змінювати не можна, окремо звітувати WER на підмножині test без перетинів, щоб мати чесне число.
     4. Переоцінити базові моделі в `evaluation/baseline/`, бо порівняння з ними теж зсунуте.
   - **Статус: виправлено в даних.** `datasets/leak-clean/clean_common_voice_train.py` прибрав з train усі речення й дикторів, що є в dev і test. Train зменшився з 73 415 до 38 849 рядків, а дикторів лишилося 74 з 435. 129 рядків train збігаються з dev або test лише після нормалізації (регістр, пунктуація), їх не прибирали. Моделі ще не перенавчені.

3. **Чекпоінти зберігаються не на тих відсотках епохи.**
   - Де: `fine-tune/checkpoints.py:176`, `range(5, 101, int(self.fraction * 100))`
   - Факт: цикл завжди стартує з 5. Для student (`checkpoint_every: 0.25`) збереження йдуть на 5, 30, 55 і 80 %, а не на 25, 50, 75 і 100 %. Це перевірено.
   - Додатково:
     1. при `fraction < 0.01` крок дорівнює 0, і `range` падає з `ValueError`;
     2. через похибку float `int(0.29*100)` дає 28;
     3. 100 % збігається з епохальним збереженням, тож той самий стан пишеться двічі.
   - **Як вирішити:**
     1. У `on_train_epoch_start` рахувати крок як `step = round(self.fraction * 100)` і перебирати `range(step, 100, step)`. Верхня межа 100 не входить, бо кінець епохи вже зберігає `on_train_epoch_end`.
     2. У `fine-tune/settings.py` (`_fraction`) перевіряти, що `checkpoint_every * 100` є цілим числом не менше 1, тобто значення кратне 0,01. Інакше зупиняти запуск з помилкою.

4. **`precision: bf16-true` при `learning_rate: 1e-5`.**
   - Де: `config.yaml:35` (training), `config.yaml:96` (student)
   - Суть: у режимі `bf16-true` ваги й стан AdamW зберігаються в bf16, точність якого лише близько 3 значущих цифр. Оновлення завбільшки близько 1e-5 від ваги здебільшого округлюються до нуля, і decoder майже не вчиться.
   - **Як вирішити:**
     1. У `config.yaml` змінити `training.precision` і `student.precision` на `bf16-mixed`. Тоді master-ваги й стан AdamW лишаються у fp32, а обчислення йдуть у bf16.
     2. Пам'яті стане потрібно більше, бо decoder і optimizer зберігатимуться у fp32. Можливо, доведеться зменшити `batch_size.train`. Перевірити на короткому запуску з `limit_train_batches`.
     3. За бажання додати в `settings.py` попередження або помилку для `*-true` під час навчання.
     4. Чекпоінти, навчені в `bf16-true`, варто порівняти з новим запуском: їхня якість, імовірно, занижена.

---

## B. Пайплайн і цілісність даних

5. **`label_ready` не перевіряє маркер `complete`.**
   - Де: `pipeline.py:219`
   - Суть: якщо `label.py` впав посередині, `labels.csv` уже існує і не порожній. При `label: skip_if_present` пайплайн вважає labelling завершеним, і student мовчки навчається на частині даних.
   - Те саме стосується `student.labels: latest`: `find_labelling_file` у `fine-tune/settings.py` теж не дивиться на `complete`.

6. **`labels: latest` і `marks: latest` можуть вказати на різні запуски.**
   - Де: `fine-tune/settings.py` (`find_labelling_file`, `_stored_path`)
   - Суть: кожен файл шукається окремо як найновіший запуск, де він є. Якщо в новому запуску є `labels.csv`, але ще немає `label_marks.csv`, оцінки візьмуться зі старого запуску. ID збігаються, тож перевірка в `load_slices` не спрацює, а тексти будуть оцінені не ті.
   - Натомість `pipeline.py:435` бере `labels.parent / "label_marks.csv"`, тобто логіка непослідовна.

7. **`accepted_mark: [0,1,2,3,4,5]` вимикає фільтр.**
   - Де: `config.yaml:81`
   - Суть: при `filter_marks: true` приймаються всі оцінки. У навчання йдуть 4 771 мітка з оцінкою 0 і 4 996 з оцінкою 1, а відкидаються лише рядки без оцінки.
   - Попередній експеримент `evaluation/student_whisper-commonvoice-based-mark5-epoch01-01/config.yaml` використовував `[5]`. Варто перевірити, чи це навмисно.

8. **Дуже короткі VAD-сегменти.**
   - Де: `pseudo-labelling/vad-chunking/chunk.py:89-98` (`chunk_spans`)
   - Факт: у `ukrainian_30s.csv` 1 487 сегментів коротші за 0,5 с і 2 975 коротші за 1 с, найкоротший має 5 мс.
   - Причина: залишок `(cursor, duration)` після останнього розрізу, або межа VAD одразу після курсора.
   - Наслідок: на такому аудіо Whisper вигадує текст, і ці мітки потрапляють у навчання. Мінімальної довжини сегмента немає, хвіст не зливається з попереднім шматком.

9. **Прогрес VAD-шарду не зберігається по ходу роботи.**
   - Де: `pseudo-labelling/vad-chunking/chunk.py:252`
   - Суть: `spans_for_shard` повертає словник лише після обробки всього шарду, і тільки тоді все дописується в `.jsonl`. Падіння посередині втрачає всю роботу над шардом. Docstring стверджує, що прогрес пишеться поступово.

10. **Оркестратор вантажить CPU на 100 %.**
    - Де: `pseudo-labelling/vad-chunking/chunk_orchestrator.py:93`
    - Суть: `sleep` виконується лише коли `pending and len(running) >= max_processes`. Коли черга шардів порожня, а процеси ще працюють, цикл крутиться без паузи.
    - Додатково: дочірні процеси мають `start_new_session=True`. При Ctrl+C оркестратора вони не вбиваються, бо немає `try/finally: stop(running)`.

11. **Обірваний рядок CSV після падіння.**
    - Де: `pseudo-labelling/label.py:65-69` і `:180` (`finished_ids`, `append_labels`), `fine-tune/fine-tune-student.py:161`
    - Суть: якщо процес упав під час запису, останній рядок `labels.csv` може лишитися обрізаним, з неповним ID або без поля `text`.
      1. При відновленні сегмент перемічається, а битий рядок лишається у файлі.
      2. `row["text"].strip()` у student падає з `AttributeError` на `None`.

12. **Оцінка з marks конвертується в число без перевірки.**
    - Де: `fine-tune/fine-tune-student.py:143` (`int(row["mark"])`), `pseudo-labelling/llm-as-a-judge/plot_marks.py` (`latest_marks`)
    - Суть: будь-яке пусте або нечислове значення, наприклад з обірваного рядка, зупиняє навчання.

13. **Методологія pseudo-labelling.**
    - Де: `pseudo-labelling/llm-as-a-judge/judge.py`, `pseudo-labelling/label.py:120-129`
    - Суть:
      1. LLM-суддя бачить лише текст і не чує аудіо. Вигадана, але граматично правильна транскрипція отримує оцінку 5.
      2. `generate` викликається без захисту від повторів (`no_repeat_ngram_size`, compression-ratio, `max_new_tokens`), тож зациклені вигадки потрапляють у мітки.
      3. Тексти міток відправляються в LLM як є, тож можлива prompt-injection з транскрипції. Ризик низький.

14. **Розбіжність налаштувань між етапами.**
    - Де: `pseudo-labelling/settings.py`, `evaluation/evaluate.py`
    - Суть:
      1. Labelling і evaluate беруть верхній `gpus`, а не `student.gpus`.
      2. `evaluate.py` при оцінці student-чекпоінта бере базу з `model.id`, а не зі `student.base`. Зараз це однакові значення, але якщо їх розвести, ваги не завантажаться.

---

## C. Надійність і продуктивність

15. **Суддя ловить не всі мережеві помилки.**
    - Де: `pseudo-labelling/llm-as-a-judge/judge.py:172,191`
    - Суть: перехоплюються лише `URLError`, `TimeoutError`, `KeyError` і `JSONDecodeError`. Не перехоплюються:
      1. `ConnectionResetError` / `http.client.RemoteDisconnected` (це `OSError`, не `URLError`);
      2. `http.client.IncompleteRead`;
      3. `ssl.SSLError`;
      4. `IndexError` при `"choices": []`;
      5. `TypeError`, якщо `choices` дорівнює `null`.
    - Будь-яка з них прокидається з `future.result()` і зупиняє весь запуск разом із запитами, що ще в дорозі.

16. **`evaluate.py` індексує всі train-датасети.**
    - Де: `evaluation/evaluate.py:493` (`ft.load_splits()["test"]`)
    - Суть: щоб отримати test, будуються всі спліти train/dev/test, зокрема OpenTTS-zip, TG Voices і FLEURS train. Оцінка повільна і падає, якщо будь-якого train-датасету немає на диску.
    - Так само `fine-tune-student.py:599` викликає `load_splits()` з `common_voice.py`, який перевіряє існування файлів для всіх чотирьох сплітів CV, включно з 73 тис. train-кліпів.
    - Ще `evaluate.py:186`: `num_workers=0`, тож аудіо декодується в головному потоці.

17. **Пошкоджений JSON метаданих тихо замінюється порожнім.**
    - Де: `scripts/monitor/meta_collector.py:115-119` і `:321`
    - Суть: при `JSONDecodeError` скрипт починає з порожнього списку, і наступний `save_meta` затирає години результатів Whisper. Безпечніше зупинитися з помилкою або зробити бекап.
    - Продуктивність: весь JSON (~15 МБ, 389 тис. рядків) перезаписується після кожного кліпу, тож кількість записаних даних зростає квадратично.

18. **Неправильна назва змінної середовища.**
    - Де: `.env.example:1`, `HF-TOKEN=`
    - Суть: бібліотеки Hugging Face читають `HF_TOKEN`. Дефіс у назві до того ж неприйнятний для shell. Токен ніколи не використовується.

19. **Залежності.**
    - Де: `requirements.txt`
    - Суть:
      1. `av` (PyAV) імпортується в `fine-tune/speech_data.py` і `fine-tune/fine-tune-student.py`, але в `requirements.txt` його немає. Він встановлюється лише як залежність `faster-whisper`.
      2. Коментар у `scripts/monitor/meta_collector.py:194` каже, що `to_ndarray` у PyAV несумісний з NumPy 2, хоча навчальні скрипти використовують саме його, а закріплено `numpy==2.5.3`. Варто закріпити сумісну версію `av`.
      3. `torch`, `transformers`, `lightning`, `jiwer` і `silero-vad` не закріплені. `torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch` (`fine-tune/fine-tune.py:393`, `fine-tune/fine-tune-student.py:575`) є лише в нових версіях torch і дасть `AttributeError` на старих.
      4. Системна залежність `ffmpeg`/`ffprobe` ніде не задокументована.

20. **Відновлення з чекпоінта посеред епохи.**
    - Де: `fine-tune/checkpoints.py` у поєднанні з `trainer.fit(ckpt_path=...)`
    - Суть: семплери (`RandomSampler`, `RowGroupSampler`) не мають стану. Після відновлення посеред епохи вона проходиться заново:
      1. частина даних бачиться двічі;
      2. лінійний scheduler виходить за `estimated_stepping_batches`, і LR стає 0 до кінця;
      3. порядок перемішування відрізняється від початкового.
    - Також `fine-tune.py` з `resume: true` бере найновіший каталог навіть із завершеним запуском: `fit` нічого не робить, лише перезапускає test.

21. **Дуже повільне читання при перемішаних даних.**
    - Де: `fine-tune/speech_data.py:121-125` (`_ArrowPart._batch`), `:221` (`_ParquetPart._audio_bytes`)
    - Суть: при `shuffle=True` кеш на один батч майже ніколи не влучає.
      1. Для FLEURS кожен приклад перечитує Arrow-потік від початку до потрібного батчу.
      2. Для parquet кожен приклад читає й декодує всю row group з аудіо.
    - Схожа проблема в student (`VoaSliceDataset`): DataLoader роздає батчі воркерам по черзі, тож кожен воркер читає ту саму row group і декодує той самий довгий файл повторно.

22. **Обробка аудіо в `decode_audio`.**
    - Де: `fine-tune/speech_data.py` (`decode_audio`), `fine-tune/fine-tune-student.py:116-133` (`load_waveform`)
    - Суть:
      1. Для interleaved-форматів (наприклад, стерео WAV `s16`) `frame.to_ndarray()` повертає форму `(1, N*channels)`. Канали тоді не усереднюються, а склеюються, і аудіо стає вдвічі довшим.
      2. Для `uint8` нормалізація без зсуву на 128 неправильна.
    - Для mono MP3 Common Voice це не проблема, але для інших джерел ризик є.

23. **Watcher графіків нестабільний.**
    - Де: `fine-tune/plot_metrics.py:354-369`
    - Суть:
      1. `pd.read_csv` на файлі, який Lightning саме переписує, може впасти з `ParserError` і зупинити watcher.
      2. Якщо watcher запущено до старту навчання, він одразу завершується, бо `current == seen == None`.
      3. Пошук процесу за підрядком `"fine-tune/fine-tune.py"` не спрацює, якщо скрипт запущено з іншого каталогу.

24. **Відсутні аудіо в Common Voice відкидаються мовчки.**
    - Де: `fine-tune/common_voice.py:85-89`
    - Суть: такі кліпи викидаються лише з `print`. Тестовий набір може непомітно відрізнятися між машинами, і результати стають непорівнянними. Крім того, `path.open()` у `:66` не закривається.

---

## D. Безпека

25. **Передбачуваний тимчасовий файл у `/tmp`.**
    - Де: `scripts/monitor/meta_collector.py:298`, `Path("/tmp/voa_current").with_suffix(suffix)`
    - Суть: на спільному сервері інший користувач може заздалегідь створити `/tmp/voa_current.mp3` як симлінк, і `write_bytes` перезапише файл, на який він вказує (CWE-377). Паралельні запуски теж перетирають один одному файл.
    - Рекомендація: `tempfile.NamedTemporaryFile` / `mkstemp`.

26. **`torch.load(..., weights_only=False)`.**
    - Де: `fine-tune/fine-tune-student.py:384`, `evaluation/evaluate.py:130`, `pseudo-labelling/label.py` (`load_teacher`)
    - Суть: unpickle може виконати довільний код з чекпоінта. Для власних файлів це прийнятно, для чекпоінтів зі сторонніх джерел небезпечно. Для читання лише ваг достатньо `weights_only=True`.

27. **Секрети.**
    - Суть: `.env` правильно додано в `.gitignore`. Водночас `output/` теж у `.gitignore`, але `output/pseudo-labelling/.../label_marks.csv` уже відстежується git, тож ігнор для нього не діє. Це неузгодженість, а не витік.

---

## E. Дрібні неточності й документація

28. **Docstring-и й коментарі розходяться з кодом.**
    1. `fine-tune/fine-tune.py:3` і `config.yaml:50` кажуть, що train і dev «concatenated», хоча dev використовується окремо як валідація.
    2. `scripts/monitor/meta_collector.py:5` пише «first 300 seconds», а `LANGUAGE_SECONDS = 120`.
    3. Docstring-и в `scripts/monitor/*.py` посилаються на неіснуючу теку `src/` і на `notebooks/` (насправді `archive/notebooks/`).
    4. `fine-tune/plot_metrics.py` у прикладах і довідці згадує `student_model-whisper-uk`, а в конфігу student називається `student_whisper-commonvoice-based`.

29. **Шляхи за замовчуванням прописані в коді, а не взяті з конфігу.**
    - Де: `pseudo-labelling/llm-as-a-judge/judge.py:26`, `pseudo-labelling/llm-as-a-judge/plot_marks.py` (`DEFAULT_MARKS`), `pseudo-labelling/vad-chunking/chunk.py` і `scripts/monitor/voa_dataset.py` (`/mnt/data/rberehun/voa-ukr`)
    - Суть: `python main.py judge` без `--labels` бере фіксований `model-whisper-uk-epoch00-01/labels.csv`, а не `student.labels: latest`. Модель судді (`pipeline.judge_model`) береться з конфігу лише в `pipeline.py`, а `main.py judge` читає її тільки з `.env`.

30. **Чекпоінти займають багато місця.**
    - Де: `fine-tune/checkpoints.py` (`BatchFractionCheckpoint`)
    - Суть: при `checkpoint_every: 0.05` повний чекпоінт із станом optimizer пишеться близько 20 разів за епоху, без ротації старих. Симлінк `last.ckpt` оновлюється неатомарно: спершу `unlink`, потім `symlink_to`.

31. **Обмеження в settings.**
    - Де: `fine-tune/settings.py`
    - Суть:
      1. `_limit` пропускає дробові значення більші за 1, наприклад `1.5`, які Lightning відхилить.
      2. `gradient_clip` має бути більшим за 0, тож вимкнути clipping неможливо.

32. **Загальний шум у результатах.**
    - Де: `fine-tune/fine-tune.py`, `fine-tune/fine-tune-student.py`
    - Суть:
      1. `results.json` перезаписується для кожного запуску в тому самому `output/fine-tune/<name>/`, хоча метрики `CSVLogger` йдуть у нові `version_N`.
      2. Після відновлення метрики одного навчання розкидані по кількох `version_N`.
      3. Student не викликає `write_plots`, на відміну від fine-tune.

---

## Пріоритет виправлень

1. Пункти 1–2: прибрати витік даних між train і dev/test Common Voice, бо він впливає на всі вже отримані результати.
2. Пункти 3–4: виправити відсотки чекпоінтів і перейти з `bf16-true` на `bf16-mixed`.
3. Пункти 5–8: перевіряти `complete`, брати оцінки з того самого запуску, що й мітки, перевірити `accepted_mark`, відкидати короткі сегменти.
4. Пункти 15, 17, 18, 19 і 25: надійність, залежності й безпечний тимчасовий файл.
