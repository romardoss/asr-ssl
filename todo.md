# TODO

Контекст: `train_validated.tsv` тепер містить усі валідовані записи Common Voice, крім кліпів dev і test (62 460 рядків). Спільні з dev і test речення й диктори лишено свідомо. Див. `datasets/common-voice/README.md`.

1. [ ] **Перенавчити й переоцінити моделі.** Усі наявні моделі вчилися на train, що містив кліпи dev.
   1. `python main.py fine-tune --note "..."` на новому train. Кроків за епоху стане менше, тож зміниться й розклад LR.
   2. Якщо вчитель змінився, перезібрати псевдомітки й перенавчити student.
   3. Переоцінити `evaluation/baseline/`, перебудувати `model_comparison.png` (`evaluation/batch-compare.py`) і повторно прогнати `datasets/leak-clean/score_common_voice_cleaned.py`.

2. [ ] **Перевірка перетину сплітів у завантажувачі.** `load_configured_splits` або `load_split` має зупиняти запуск, якщо train перетинається з dev чи test за кліпом (`path`/ID). Спільні речення й диктори помилкою не вважати.

3. [ ] **`cv10_test` проти train Common Voice.** Кліпи CV10 test, імовірно, є у `validated` CV27, а отже в train. Звірити за хешем аудіо, іменем файлу й текстом. Код є лише в розділах 4–5 `datasets/leak-clean/README.md`, його треба зберегти у файл. Якщо перетин є, прибрати ці кліпи з train або вилучити `cv10_test` з `data.test`.

4. [ ] **Перейменувати запуски student з ablation за оцінками судді.** Усі мають назву `student_whisper-commonvoice-based`, тож теки оцінок і легенда `batch-compare.py` їх не розрізняють. У `results.csv` вони вже розрізняються за колонкою `accepted_marks`.

   | Запуск | Чекпоінти зараз → нова назва | Оцінка зараз → нова назва |
   |---|---|---|
   | усі оцінки `[0..5]` | `checkpoints/student_whisper-commonvoice-based_01` → `…-allmarks_01` | `evaluation/student_whisper-commonvoice-based-epoch01-01` → `…-allmarks-epoch01-01` |
   | лише 3 (`_02`, звірити: `grep accepted_mark …_0*/config.yaml`) | `…_02` → `…-mark3_01` | на сервері `…-epoch01-02` → `…-mark3-epoch01-01` |
   | лише 5 | `…based` (без номера) → `…-mark5_01` | вже `…-mark5-epoch01-01` |

   Команди для сервера, з кореня репозиторію, після `git pull`:

   1. Звірити, який запуск з якими оцінками, і що є в `evaluation/`. Якщо `_01` чи `_02` не збігаються з таблицею, поміняти аргументи в кроці 2.
      ```bash
      grep -H accepted_mark /mnt/data/rberehun/checkpoints/student_whisper-commonvoice-based{,_0*}/config.yaml
      ls -d evaluation/student_whisper-commonvoice-based*
      ```
   2. Зберегти як `/tmp/rename_students.sh` і запустити `bash /tmp/rename_students.sh`. Скрипт перейменовує теку чекпоінтів у `<назва>_01` (інакше `_numbered_dirs` її не знайде), файли `.ckpt` і `last.ckpt`, теку оцінки, а також шляхи й назви в `meta.yaml`, `config.yaml` і `results.json`. Перевірено на макеті тек.
      ```bash
      set -euo pipefail
      CK=/mnt/data/rberehun/checkpoints
      OLD=student_whisper-commonvoice-based

      # $1 стара тека чекпоінтів, $2 нова назва моделі, $3 стара тека оцінки в evaluation/
      rename_run() {
        local src=$CK/$1 new=$2 dst=$CK/$2_01 eval_old=evaluation/$3 eval_new=evaluation/$2-epoch01-01
        [ -d "$src" ] || { echo "немає $src"; return; }
        local link; link=$(readlink "$src/last.ckpt" || true)
        mv "$src" "$dst"
        for f in "$dst"/$OLD-*.ckpt; do mv "$f" "$dst/$new${f##*/$OLD}"; done
        [ -n "$link" ] && ln -sfn "$new${link#$OLD}" "$dst/last.ckpt"
        [ -d "$eval_old" ] && [ "$eval_old" != "$eval_new" ] && mv "$eval_old" "$eval_new"
        for file in "$dst/meta.yaml" "$dst/config.yaml" "$eval_new/meta.yaml" "$eval_new/config.yaml" "$eval_new/results.json"; do
          [ -f "$file" ] && sed -i -e "s#$src\([/\"]\|\$\)#$dst\1#g" -e "s#$OLD-epoch#$new-epoch#g" \
            -e "s#^model: $OLD\$#model: $new#" -e "s#^  name: $OLD\$#  name: $new#" "$file"
        done
        echo "$src -> $dst"
      }

      rename_run ${OLD}_01 ${OLD}-allmarks ${OLD}-epoch01-01
      rename_run ${OLD}_02 ${OLD}-mark3    ${OLD}-epoch01-02
      rename_run ${OLD}    ${OLD}-mark5    ${OLD}-mark5-epoch01-01
      ```
   3. Розкласти метрики навчання за запусками. Для кожної `version_N` вивести `accepted_mark`, а потім перенести відповідно:
      ```bash
      for v in output/fine-tune/student_whisper-commonvoice-based/version_*; do echo "$v: $(grep -h accepted_mark $v/config.yaml)"; done
      # приклад для version_3 з [3]:
      mkdir -p output/fine-tune/student_whisper-commonvoice-based-mark3
      mv output/fine-tune/student_whisper-commonvoice-based/version_3 output/fine-tune/student_whisper-commonvoice-based-mark3/
      ```
   4. Перебудувати таблицю й графік і закомітити перейменовані теки оцінок:
      ```bash
      env/bin/python evaluation/results_table.py
      env/bin/python evaluation/batch-compare.py evaluation
      git add -A evaluation && git commit -m "rename student ablation runs"
      ```

   Щоб не повторилося: перед кожним ablation змінювати `student.name` або додати в `fine-tune-student.py` автоматичний суфікс з `accepted_mark`.

5. [ ] **Відкриті пункти аудиту `REVIEW_FINDINGS.md` (2026-11-03).**
   1. Пункт 3: `fine-tune/checkpoints.py` досі має `range(5, 101, ...)`, тому чекпоінти student зберігаються на 5, 30, 55 і 80 % епохи.
   2. Пункт 4: `precision: bf16-true` у `training` і `student`. Якщо декодер майже не вчиться, висновки ablation можуть бути спотворені.
   3. Пункти 5–32 звірити з кодом і позначити виправлені.

6. [ ] **Перевірка датасетів на витоки на сервері.** Виконати `env/bin/python datasets/leak-clean/check_leaks.py`, на Vast — з `ASR_CONFIG=config.vast.yaml`, і прочитати `output/leak-clean/report.md`. Для VOA потрібен `labels.csv` вчителя. Скрипт порівнює лише текст. Тому для хешу, імені файлу й диктора потрібен пункт 3, а пошук речень CV dev/test усередині псевдоміток VOA (README, розділ 3, крок 6) ще не реалізовано.
