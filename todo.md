# TODO: після очищення train Common Voice

`train_validated.tsv` (`validated` мінус `test`) очищено скриптом `datasets/leak-clean/clean_common_voice_train.py`. З train прибрано лише кліпи dev і test: лишилося 62 460 рядків з 73 415 і 376 дикторів. Спільні з dev і test речення й диктори свідомо лишено, щоб навчатися на всіх валідованих даних. Що це не завищує WER, перевірено окремим скриптом `datasets/leak-clean/score_common_voice_cleaned.py` на `test_cleaned.tsv` (5 703 кліпи без дикторів і нормалізованих речень з train): різниця з повним test у межах ±0.3 п.п. Опис — у `datasets/common-voice/README.md`. Нижче те, що ще потрібно зробити.

1. [x] **129 рядків, що збігаються лише після нормалізації.** Пункт втратив сенс, бо речення з train більше не прибираються. Нормалізоване порівняння тепер використовується для відбору `test_cleaned.tsv`.

2. [ ] **Перенавчити моделі й оцінити їх заново.**
   - Суть: усі чекпоінти й результати в `evaluation/` отримано на train із витоком.
   - Що зробити:
     1. перенавчити fine-tune (`python main.py fine-tune`) на очищеному train;
     2. перезібрати псевдомітки вчителем, якщо вчитель зміниться, і перенавчити student;
     3. заново оцінити всі моделі, включно з `evaluation/baseline/`, і перебудувати `model_comparison.png` (`evaluation/batch-compare.py`).
   - Врахувати: Common Voice train зменшився з 73 415 до 62 460 рядків, тож кількість кроків за епоху й розклад LR трохи зміняться.
   - Після переоцінки прогнати `datasets/leak-clean/score_common_voice_cleaned.py`, щоб переконатися, що висновок про спільні диктори й речення не змінився.

3. [ ] **Додати перевірку на перетин сплітів у коді завантаження.**
   - Суть: зараз `fine-tune/common_voice.py` і `fine-tune/speech_data.py` приймають будь-які спліти без перевірки, тому витік і не було помічено.
   - Що зробити: у `load_configured_splits` (або в `load_split`) зупиняти запуск з помилкою, якщо train перетинається з dev або test за кліпом (`path` або ID).
   - Спільні речення й диктори train з dev і test помилкою не вважати: їх лишено свідомо.

4. [ ] **Перевірити `cv10_test` проти train Common Voice.**
   - Суть: тестові кліпи Common Voice 10, імовірно, увійшли до `validated` пізніших релізів, а отже й до train Common Voice 27. Тоді `cv10_test` перевіряє модель на даних, на яких вона навчалася.
   - Що зробити: на сервері зібрати маніфест і проаналізувати його за `README.md` у цій теці (розділи 4–5), щоб звірити `cv10_test` із train за хешем аудіо, назвою файлу й текстом. Якщо перетин є, прибрати ці кліпи з train або вилучити `cv10_test` з `data.test`.

5. [ ] **Перейменувати два останні запуски student (ablation за оцінками LLM-судді).**
   - Суть: обидва запуски мають однакову назву `student_whisper-commonvoice-based`, тож за шляхом не видно, з якими оцінками навчалася модель. У `batch-compare.py` легенда береться з назви теки, тому на графіку їх теж не розрізнити.
   - Які запуски:
     1. **усі оцінки** (`accepted_mark: [0,1,2,3,4,5]`, весь датасет): чекпоінти в `_01`, оцінка в `evaluation/student_whisper-commonvoice-based-epoch01-01/` (видно з її `config.yaml` і `meta.yaml`);
     2. **лише оцінка 3** (`accepted_mark: [3]`, теперішній `config.yaml`): чекпоінти, імовірно, в `_02`. Оцінки цього запуску в репозиторії немає. На сервері вона, якщо вже є, лежить у `evaluation/student_whisper-commonvoice-based-epoch01-02/`. Перед перейменуванням звірити: `grep accepted_mark /mnt/data/rberehun/checkpoints/student_whisper-commonvoice-based_0*/config.yaml`.
   - Що перейменувати (схема `<student.name>-<фільтр оцінок>`, як у вже наявного `-mark5`):

     | Що | Зараз | Нова назва |
     |---|---|---|
     | чекпоінти, усі оцінки | `/mnt/data/rberehun/checkpoints/student_whisper-commonvoice-based_01/` | `/mnt/data/rberehun/checkpoints/student_whisper-commonvoice-based-allmarks_01/` |
     | чекпоінти, оцінка 3 | `/mnt/data/rberehun/checkpoints/student_whisper-commonvoice-based_02/` | `/mnt/data/rberehun/checkpoints/student_whisper-commonvoice-based-mark3_01/` |
     | оцінка, усі оцінки | `evaluation/student_whisper-commonvoice-based-epoch01-01/` | `evaluation/student_whisper-commonvoice-based-allmarks-epoch01-01/` |
     | оцінка, оцінка 3 (на сервері) | `evaluation/student_whisper-commonvoice-based-epoch01-02/` | `evaluation/student_whisper-commonvoice-based-mark3-epoch01-01/` |
     | метрики навчання (на сервері) | `output/fine-tune/student_whisper-commonvoice-based/version_N/` обох запусків | `output/fine-tune/student_whisper-commonvoice-based-allmarks/` і `...-mark3/` |
     | для послідовності: старий запуск з оцінкою 5 | `/mnt/data/rberehun/checkpoints/student_whisper-commonvoice-based/` (без номера) | `/mnt/data/rberehun/checkpoints/student_whisper-commonvoice-based-mark5_01/` |

   - Як перейменувати, щоб нічого не зламати:
     1. Теку чекпоінтів називати саме `<назва>_NN`. Код (`_numbered_dirs` у `fine-tune/checkpoints.py`) знаходить запуски лише за шаблоном `<student.name>_(\d+)`. Інакше `resume` і `latest_student_checkpoint` теку не побачать.
     2. Файли `.ckpt` усередині теж перейменувати (`student_whisper-commonvoice-based-epoch01.ckpt` → `student_whisper-commonvoice-based-mark3-epoch01.ckpt`), бо `evaluate.py` будує назву теки оцінки з назви файлу чекпоінта (`model_slug`). Саме тому обидві оцінки й отримали однакову назву. Після цього заново створити симлінк `last.ckpt`, бо він вказує на стару назву файлу.
     3. Оновити `checkpoint_dir` у `meta.yaml` теки оцінки й теки чекпоінтів.
     4. Метрики в `output/fine-tune/.../version_N/` розкласти за запусками. Звіряти за `config.yaml` у кожній `version_N`. `results.json` у цій теці перезаписується кожним запуском (`REVIEW_FINDINGS.md`, пункт 32), тож зараз там лише результат останнього.
     5. Перебудувати `model_comparison.png` (`evaluation/batch-compare.py`).
   - Щоб не повторилося: перед кожним прогоном ablation змінювати `student.name` у `config.yaml` (наприклад, `student_whisper-commonvoice-based-mark3`). Або додати в `fine-tune/fine-tune-student.py` автоматичний суфікс з `accepted_mark` (`-allmarks`, `-mark3`, `-mark4-5`), коли `filter_marks: true`.

6. [ ] **Пройти відкриті пункти аудиту `REVIEW_FINDINGS.md`.**
   - Суть: аудит кодової бази вже зроблено 2026-11-03, повторювати його не треба. Частину знахідок з того часу виправлено (наприклад, `HF_TOKEN` у `.env.example`, `av` у `requirements.txt`), але не всі.
   - Досі відкриті критичні пункти (перевірено 2026-11-06):
     1. пункт 3: `fine-tune/checkpoints.py:176` усе ще `range(5, 101, ...)`, тож чекпоінти student зберігаються на 5, 30, 55 і 80 % епохи;
     2. пункт 4: `config.yaml` усе ще має `precision: bf16-true` для `training` і `student`. Це стосується й обох запусків ablation: висновки з порівняння оцінок можуть бути спотворені, якщо decoder майже не вчиться.
   - Решту пунктів (5–32) звірити з поточним кодом і позначити в `REVIEW_FINDINGS.md`, що виправлено.

7. [ ] **Запустити перевірку датасетів на витік на сервері.**
   - Стан: скрипти `datasets/leak-clean/` готові до запуску й закомічені разом із `project_config.py`, від якого вони залежать. 2026-11-06 їх прогнано:
     1. локально на TSV Common Voice: результат збігся з описаним у `datasets/leak-clean/README.md` (19 кліпів dev і 32 кліпи test мають текст із train);
     2. на синтетичних даних для всіх інших датасетів (FLEURS, OpenTTS, TG Voices, Ukrainian Dialects, CV10 test, VOA): усі гілки коду відпрацювали, проблеми покриття знайдено правильно, графіки побудовано.
   - Що зробити: на сервері, з кореня репозиторію, після `git pull`:
     ```bash
     env/bin/python datasets/leak-clean/collect_metadata.py
     env/bin/python datasets/leak-clean/find_overlaps.py --min-words 4
     env/bin/python datasets/leak-clean/plot_overlaps.py
     ```
     На Vast додати `ASR_CONFIG=config.vast.yaml`. Для VOA потрібен `labels.csv` вчителя в `output/pseudo-labelling/`, інакше VOA лише позначиться в `coverage.csv` як не знайдений.
   - Обмеження: скрипти порівнюють лише нормалізований текст. Вони не перевіряють:
     1. хеш аудіо, назву файлу й диктора. Тому для пункту 4 (`cv10_test` проти train Common Voice) усе одно потрібен код з розділів 4–5 `datasets/leak-clean/README.md`. Він лежить лише в README, тож його треба зберегти у файл. Шлях `datasets/common-voice/raw` і `LABELS` у ньому прописані вручну;
     2. чи містять псевдомітки VOA речення з dev і test Common Voice як підрядок (README, розділ 3, крок 6). Повний збіг тексту слайсу VOA з реченням майже неможливий, тож ця перевірка зараз нічого не покаже.
