"""Перевірка LaTeX-джерел магістерської дисертації на відповідність docs/diploma/rules.md.

Використання:
    python check_tex.py main.tex [--max 300]

Розгортає \\input/\\include, аналізує текст регулярними виразами й виводить
нумерований список порушень з ID правил. Верстку (висячі заголовки, розміщення
плавучих об'єктів) перевіряй у PDF.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

APPENDIX_OK = set("АБВГДЕЖКЛМНПРСТУФХЦШЩЮЯ")


def expand(path: Path, seen=None) -> list[tuple[str, int, str]]:
    """Повертає [(файл, рядок, текст)] з розгорнутими \\input/\\include, без коментарів."""
    seen = seen or set()
    path = path if path.suffix else path.with_suffix(".tex")
    if path in seen or not path.exists():
        return []
    seen.add(path)
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = re.sub(r"(?<!\\)%.*", "", line)
        m = re.search(r"\\(?:input|include|subfile)\{([^}]+)\}", line)
        if m:
            out.extend(expand(path.parent / m.group(1), seen))
            continue
        out.append((path.name, n, line))
    return out


def run(main: Path, maxn: int) -> int:
    lines = expand(main)
    issues: list[tuple[str, str, str]] = []
    add = lambda rule, where, msg: issues.append((rule, where, msg))
    full = "\n".join(l for _, _, l in lines)

    # ---- преамбула ----
    m = re.search(r"\\documentclass\[([^\]]*)\]\{(\w+)\}", full)
    if m:
        opts, cls = m.group(1), m.group(2)
        if "14pt" not in opts:
            add("F-2.1", "преамбула", f"\\documentclass без 14pt ({opts})")
        if "a4paper" not in opts:
            add("F-1.1", "преамбула", "\\documentclass без a4paper")
        if cls not in ("extreport", "extbook"):
            add("F-2.1", "преамбула", f"клас {cls}: для 14pt і розділів потрібен extreport")
    if "kpi-thesis" not in full:
        add("—", "преамбула", "не підключено kpi-thesis.sty — правила F-1…F-13 доведеться задавати вручну")
        g = re.search(r"\\usepackage\[([^\]]*)\]\{geometry\}", full)
        if not g or not all(x in g.group(1).replace(" ", "") for x in ("left=30mm", "right=10mm", "top=20mm", "bottom=20mm")):
            add("F-1.4", "преамбула", "поля не 30/10/20/20 мм")
    for bad, rule, msg in ((r"\\usepackage\{times\}|\\usepackage\{mathptmx\}", "F-2.1", "pdfLaTeX-шрифти — використовуйте XeLaTeX + Times New Roman (kpi-thesis)"),
                           (r"\\onehalfspacing|\\doublespacing", "F-2.2", "ручний інтервал — kpi-thesis вже задає 1,5 (Word)"),
                           (r"\\setlength\{\\parindent\}\{(?!1\.25cm)", "F-2.3", "абзацний відступ ≠ 1,25 см"),
                           (r"\\setlength\{\\parskip\}\{(?!0pt)", "F-2.3", "ненульовий \\parskip"),
                           (r"\\section\*?\{[^}]*\}\s*\\section", "F-5.10", "два заголовки поспіль")):
        for mm in re.finditer(bad, full):
            add(rule, "преамбула/текст", msg)
            break

    # ---- прохід по рядках ----
    order = []
    chapter, chap_sections, chap_has_concl, chap_preamble = None, {}, {}, {}
    after_heading = None          # 'chapter' | 'section'
    last_block = None             # 'text' | 'float' | 'equation' | 'list'
    in_env = []
    labels = {}                   # label -> (kind, where, order)
    refs = []                     # (label, order)
    eq_labels = []
    numbered_eq_nolabel = []
    bib = []
    cites = []
    in_app = False
    seq = 0
    pending_end = None

    def where(f, n):
        return f"{f}:{n}"

    for f, n, line in lines:
        seq += 1
        s = line.strip()
        if not s:
            continue
        w = where(f, n)

        # заголовки
        mh = re.match(r"\\(chapterconclusions|chapter|subsection|section|structuralnotoc|structural)\b(\*?)(?:\{(.*)\})?", s)
        if mh:
            cmd, star, title = mh.group(1), mh.group(2), (mh.group(3) or "")
            if cmd in ("chapter", "structural", "structuralnotoc", "section", "subsection", "chapterconclusions"):
                if last_block in ("float", "equation") and pending_end:
                    add("F-3.6", pending_end, "розділ/підрозділ закінчується рисунком/таблицею/формулою")
            if title.rstrip().endswith("."):
                add("F-5.7", w, f"крапка в кінці заголовка: «{title}»")
            if re.match(r"^\d", title):
                add("F-5.12", w, f"номер у тексті заголовка — нумерацію ставить LaTeX: «{title}»")
            if cmd in ("chapter", "structural", "structuralnotoc"):
                if chapter is not None and not in_app:
                    if not chap_has_concl.get(chapter):
                        add("F-3.4", f"розділ {chapter}", "немає \\chapterconclusions")
                    if chap_sections.get(chapter) == 1:
                        add("F-3.5", f"розділ {chapter}", "лише один підрозділ")
                if cmd == "chapter" and not star:
                    if in_app:
                        chapter = None
                    else:
                        chapter = (chapter or 0) + 1 if isinstance(chapter, int) or chapter is None else 1
                        chapter = len([o for o in order if o == "РОЗДІЛ"]) + 1
                        chap_sections[chapter] = 0
                        chap_has_concl[chapter] = False
                        chap_preamble[chapter] = False
                        order.append("РОЗДІЛ")
                    after_heading = "chapter"
                else:
                    t = title.upper()
                    key = next((k for k in ("РЕФЕРАТ", "ABSTRACT", "ПЕРЕЛІК УМОВНИХ", "ВСТУП", "ВИСНОВКИ", "ПЕРЕЛІК ПОСИЛАНЬ") if t.startswith(k)), None)
                    if key:
                        order.append(key)
                    chapter = None
                    after_heading = "structural"
            elif cmd == "section" and not star:
                if chapter is not None:
                    chap_sections[chapter] += 1
                    if after_heading == "chapter":
                        add("F-3.4", w, f"розділ {chapter} без преамбули перед першим підрозділом")
                after_heading = "section"
            elif cmd == "chapterconclusions":
                if chapter is not None:
                    chap_has_concl[chapter] = True
                after_heading = "section"
            pending_end = w
            last_block = "heading"
            continue
        if s.startswith("\\tableofcontents"):
            order.append("ЗМІСТ")
            continue
        if s.startswith("\\kpiappendix") or s.startswith("\\appendix"):
            if chapter is not None and not chap_has_concl.get(chapter):
                add("F-3.4", f"розділ {chapter}", "немає \\chapterconclusions")
            in_app = True
            chapter = None
            order.append("ДОДАТОК")
            continue

        if "\\begin{thebibliography}" in s and "ПЕРЕЛІК ПОСИЛАНЬ" not in order:
            if chapter is not None and not chap_has_concl.get(chapter):
                add("F-3.4", f"розділ {chapter}", "немає \\chapterconclusions")
            chapter = None
            order.append("ПЕРЕЛІК ПОСИЛАНЬ")
        if in_env and in_env[-1] in ("lstlisting", "verbatim", "minted") and not s.startswith("\\end{"):
            continue

        # оточення
        for mb in re.finditer(r"\\begin\{(\w+\*?)\}", s):
            env = mb.group(1)
            in_env.append(env)
            if env in ("figure", "table", "longtable", "figure*", "table*"):
                last_block = "float"
            elif env in ("equation", "align", "gather", "multline"):
                last_block = "equation"
                numbered_eq_nolabel.append([w, False, env])
            elif env in ("equation*", "align*", "gather*", "multline*"):
                last_block = "equation"
            elif env in ("itemize", "enumerate", "inlinelist"):
                last_block = "list"
        if numbered_eq_nolabel and "\\label" in s and in_env and in_env[-1] in ("equation", "align", "gather", "multline"):
            numbered_eq_nolabel[-1][1] = True
        for me in re.finditer(r"\\end\{(\w+\*?)\}", s):
            if in_env:
                in_env.pop()
        inside = set(in_env)

        # підписи
        for mc in re.finditer(r"\\caption(?:\[[^\]]*\])?\{(.*?)\}(?:\\label|$|\s)", s):
            cap = mc.group(1).strip()
            if cap.endswith("."):
                add("F-8.3/F-9.2", w, f"крапка в кінці підпису: «{cap}»")
            if re.match(r"^(Рис|Рисунок|Таблиця|Табл)", cap):
                add("F-8.3", w, "слово «Рисунок/Таблиця» і номер додає \\caption — не пишіть їх у тексті підпису")
        if "\\caption" in s and inside & {"table", "table*"}:
            pass

        # мітки
        for ml in re.finditer(r"\\label\{([^}]+)\}", s):
            lab = ml.group(1)
            kind = "fig" if inside & {"figure", "figure*"} else "tab" if inside & {"table", "table*", "longtable"} else \
                   "eq" if inside & {"equation", "align", "gather", "multline"} else "other"
            labels[lab] = (kind, w, seq)
            if kind == "eq":
                eq_labels.append(lab)
        for mr in re.finditer(r"\\(?:ref|eqref|autoref|cref)\{([^}]+)\}", s):
            for lab in mr.group(1).split(","):
                refs.append((lab.strip(), seq))

        # бібліографія
        mbib = re.match(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}(.*)", s)
        if mbib:
            bib.append((mbib.group(1), w, mbib.group(2)))
            txt = mbib.group(2)
            if "URL" in txt and "дата звернення" not in txt and "DOI" not in txt:
                add("F-12.5", w, "URL без «(дата звернення: ДД.ММ.РРРР)»")
            continue
        for mc in re.finditer(r"\\cite\{([^}]+)\}", s):
            for k in mc.group(1).split(","):
                cites.append((k.strip(), seq))

        # текстові правила
        if not s.startswith("\\") or re.match(r"\\(item|textbf|textit|emph|noindent)\b", s):
            plain = re.sub(r"\$[^$]*\$", "", s)
            if chapter is not None and after_heading == "chapter":
                chap_preamble[chapter] = True
                after_heading = None
            elif after_heading in ("section", "structural"):
                after_heading = None
            if last_block != "list" or not s.startswith("\\item"):
                last_block = "text" if not inside & {"figure", "table", "equation", "align"} else last_block
            if re.search(r"\b(рис|табл)\.\s*\\(ref|eqref)", plain) and not re.search(r"\b(рис|табл)\.~\\", plain):
                add("F-2.6", w, "після «рис.»/«табл.» потрібна тильда: рис.~\\ref{…}")
            if re.search(r"(?<![-~])\s-\s", plain):
                add("F-2.5", w, "дефіс як тире — пишіть «~--» (en dash) з пробілами")
            if re.search(r"---", plain):
                add("F-2.5", w, "«---» (довге тире) — у шаблоні коротке «--»")
            if re.search(r'(?<!\\)"[^"]+"', plain) or "``" in plain:
                add("F-2.5", w, "прямі/англійські лапки — пишіть «ялинки»")
            if re.search(r"\bдан(ий|ого|ому|им|ій|а|у|і|их)\s+(розділ|підрозділ|робот|метод|алгоритм|модел|підхід|етап)", plain, re.I):
                add("F-2.8", w, "«даний» → «цей»")
            if re.search(r"\bнаступн(і|ий|а|е|их|им)\s+(крок|етап|пункт|вимог|метод|задач|завдан|результат)", plain, re.I):
                add("F-2.8", w, "«наступні» → «такі»")
            if re.match(r"^де\s*:", plain) or "\\begin{where}:" in s:
                add("F-10.4", w, "після «де» двокрапку не ставлять")
            if re.search(r"\\cite\{[^}]+\}\s*\\cite\{", s):
                add("F-11.1", w, "кілька \\cite поспіль — об'єднайте: \\cite{a,b}")
            if re.search(r"\(\s*Рис\.|\(\s*рисунок\s+\\ref|\(\s*Fig", plain):
                add("F-11.2", w, "посилання у формі (рис.~\\ref{…})")

        # елементи списків
        mi = re.match(r"\\item\s+(.*)", s)
        if mi and in_env:
            body = mi.group(1).strip()
            env = next((e for e in reversed(in_env) if e in ("itemize", "enumerate", "inlinelist")), None)
            if env == "itemize" and body and not re.search(r"[;.]\s*$", body):
                add("F-7.2", w, f"пункт маркованого списку без «;»/«.»: «{body[:50]}»")
            if env == "itemize" and body[:1].isupper() and not re.match(r"^[A-ZА-ЯІЇЄҐ]{2,}", body):
                add("F-7.2", w, f"пункт маркованого списку з великої літери: «{body[:50]}»")
            if env == "enumerate" and body and (body[:1].islower() or not body.rstrip().endswith((".", "\\dots"))):
                add("F-7.1", w, f"пункт нумерованого списку: з великої літери і з крапкою: «{body[:50]}»")

        if in_app:
            ma = re.match(r"\\chapter\{", s)

    # кінець документа
    if chapter is not None and not chap_has_concl.get(chapter):
        add("F-3.4", f"розділ {chapter}", "немає \\chapterconclusions")

    # порядок
    ORDER = ["РЕФЕРАТ", "ABSTRACT", "ЗМІСТ", "ПЕРЕЛІК УМОВНИХ", "ВСТУП", "РОЗДІЛ", "ВИСНОВКИ", "ПЕРЕЛІК ПОСИЛАНЬ", "ДОДАТОК"]
    if bib:
        order.append("ПЕРЕЛІК ПОСИЛАНЬ") if "ПЕРЕЛІК ПОСИЛАНЬ" not in order else None
    pos = [ORDER.index(o) for o in order if o in ORDER]
    for a, b in zip(pos, pos[1:]):
        if b < a:
            add("F-3.1", "структура", f"порушено порядок: {ORDER[b]} після {ORDER[a]}")
            break
    for must in ("РЕФЕРАТ", "ABSTRACT", "ЗМІСТ", "ВСТУП", "ВИСНОВКИ"):
        if must not in order:
            add("F-3.1", "структура", f"немає «{must}»")

    # рисунки/таблиці/формули: посилання до об'єкта
    first_ref = {}
    for lab, sq in refs:
        first_ref.setdefault(lab, sq)
    for lab, (kind, w, sq) in labels.items():
        if kind in ("fig", "tab"):
            r = first_ref.get(lab)
            if r is None:
                add("F-8.1/F-9.1", w, f"на {lab} немає посилання в тексті")
            elif r > sq:
                add("F-8.1/F-9.1", w, f"перше посилання на {lab} стоїть після об'єкта")
        if kind == "eq" and lab not in first_ref:
            add("F-10.2", w, f"формула {lab} нумерована, але не згадується — використайте equation*")
    for w, has_label, env in numbered_eq_nolabel:
        if not has_label:
            add("F-10.2", w, f"нумерована {env} без \\label — номер без посилання; використайте {env}*")

    # джерела
    keys = [k for k, _, _ in bib]
    cited_first = []
    for k, _ in cites:
        if k not in cited_first:
            cited_first.append(k)
    for k in cited_first:
        if keys and k not in keys:
            add("F-11.3", "посилання", f"\\cite{{{k}}} немає в thebibliography")
    for k, w, _ in bib:
        if k not in cited_first:
            add("F-11.3", w, f"джерело {k} не процитоване")
    in_order = [k for k in cited_first if k in keys]
    if in_order and in_order != [k for k in keys if k in in_order]:
        add("F-12.2", "перелік посилань", "порядок \\bibitem не збігається з порядком першої згадки")

    if not issues:
        print("Порушень не знайдено.")
        return 0
    print(f"Знайдено порушень: {len(issues)}\n")
    for i, (rule, w, msg) in enumerate(issues[:maxn], 1):
        print(f"{i}. [{rule}] {w}: {msg}")
    if len(issues) > maxn:
        print(f"… ще {len(issues) - maxn}")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("--max", type=int, default=300)
    a = ap.parse_args()
    sys.exit(run(Path(a.tex), a.max))
