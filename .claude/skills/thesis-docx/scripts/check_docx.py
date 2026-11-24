"""Перевірка .docx магістерської дисертації на відповідність docs/diploma/rules.md.

Використання:
    python check_docx.py thesis.docx [--max 300]

Виводить нумерований список порушень з ID правила (F-x.y), номером абзацу і фрагментом.
Перевіряє лише те, що можна визначити зі структури файлу; верстку (розриви сторінок,
висячі заголовки) треба додатково дивитися в PDF.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

STRUCTURAL = ["РЕФЕРАТ", "ABSTRACT", "ЗМІСТ", "ПЕРЕЛІК УМОВНИХ ПОЗНАЧЕНЬ", "ВСТУП", "ВИСНОВКИ",
              "ПЕРЕЛІК ПОСИЛАНЬ"]
ORDER = ["РЕФЕРАТ", "ABSTRACT", "ЗМІСТ", "ПЕРЕЛІК УМОВНИХ", "ВСТУП", "РОЗДІЛ", "ВИСНОВКИ",
         "ПЕРЕЛІК ПОСИЛАНЬ", "ДОДАТОК"]
APPENDIX_OK = set("АБВГДЕЖКЛМНПРСТУФХЦШЩЮЯ")
EN = "–"

RE_CHAPTER = re.compile(r"^РОЗДІЛ\s+(\d+)\b")
RE_SEC = re.compile(r"^(\d+)\.(\d+)\.?\s+\S")
RE_SUBSEC = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.?\s+\S")
RE_FIG = re.compile(r"^Рисунок\s+([А-Я]|\d+)\.(\d+)\s*(.)\s*(.*)$")
RE_TAB = re.compile(r"^(?:Таблиця)\s+([А-Я]|\d+)\.(\d+)\s*(.)\s*(.*)$")
RE_FORMULA_NUM = re.compile(r"\(((?:\d+|[А-Я])\.\d+)\)\s*$")
RE_CITE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)(?:,\s*с\.\s*[\d–-]+)?\]")


class Ctx:
    def __init__(self, doc):
        self.doc = doc
        self.issues: list[tuple[str, str, str]] = []

    def add(self, rule, where, msg):
        self.issues.append((rule, where, msg))


def iter_block(doc):
    body = doc.element.body
    for el in body.iterchildren():
        if el.tag == qn("w:p"):
            yield Paragraph(el, doc)
        elif el.tag == qn("w:tbl"):
            yield Table(el, doc)


def style_chain(style):
    while style is not None:
        yield style
        style = style.base_style


def eff_pf(p, attr):
    v = getattr(p.paragraph_format, attr)
    if v is not None:
        return v
    for st in style_chain(p.style):
        v = getattr(st.paragraph_format, attr)
        if v is not None:
            return v
    return None


def eff_run(p, r, attr):
    v = getattr(r.font, attr)
    if v is not None:
        return v
    if r.style is not None:
        for st in style_chain(r.style):
            v = getattr(st.font, attr)
            if v is not None:
                return v
    for st in style_chain(p.style):
        v = getattr(st.font, attr)
        if v is not None:
            return v
    if attr == "name":
        rpr = p.part.document.styles.element.find(qn("w:docDefaults"))
        if rpr is not None:
            f = rpr.find(".//" + qn("w:rFonts"))
            if f is not None:
                return f.get(qn("w:ascii"))
    return None


def is_list(p):
    ppr = p._p.pPr
    return ppr is not None and ppr.numPr is not None


def has_math(p):
    return p._p.find(".//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath") is not None


def has_drawing(p):
    return bool(p._p.findall(".//" + qn("w:drawing"))) or bool(p._p.findall(".//" + qn("w:pict")))


def short(t, n=70):
    t = t.replace("\n", " ").replace("\t", " ")
    return t if len(t) <= n else t[:n] + "…"


def check_page(ctx: Ctx):
    secs = ctx.doc.sections
    for i, s in enumerate(secs):
        if round(s.page_width.mm) != 210 or round(s.page_height.mm) != 297:
            ctx.add("F-1.1", f"розділ Word {i+1}", f"формат {s.page_width.mm:.0f}×{s.page_height.mm:.0f} мм ≠ A4")
        exp_r = 15 if (i == 0 and len(secs) > 1) else 10
        got = (round(s.left_margin.mm), round(s.right_margin.mm), round(s.top_margin.mm), round(s.bottom_margin.mm))
        if got != (30, exp_r, 20, 20):
            ctx.add("F-1.3/F-1.4", f"розділ Word {i+1}", f"поля Л/П/В/Н = {got}, очікується (30, {exp_r}, 20, 20)")
    last = secs[-1]
    hx = last.header._element.xml
    if "PAGE" not in hx:
        ctx.add("F-4.1", "колонтитул", "у верхньому колонтитулі основного розділу немає поля PAGE")
    elif 'w:jc w:val="right"' not in hx:
        ctx.add("F-4.1", "колонтитул", "номер сторінки не вирівняно праворуч")
    if "PAGE" in last.footer._element.xml:
        ctx.add("F-4.1", "колонтитул", "номер сторінки внизу — має бути у правому верхньому куті")
    if len(secs) > 1 and "PAGE" in secs[0].header._element.xml and not secs[0].header.is_linked_to_previous:
        ctx.add("F-4.2", "титул", "на титулі/завданні друкується номер сторінки")


def main_text_checks(ctx: Ctx, idx, p, text):
    al = eff_pf(p, "alignment")
    if al not in (WD_ALIGN_PARAGRAPH.JUSTIFY, None) and al != 3:
        ctx.add("F-2.3", f"абз. {idx}", f"вирівнювання {al}, має бути за шириною: «{short(text)}»")
    fi = eff_pf(p, "first_line_indent")
    if fi is None or abs(fi.cm - 1.25) > 0.05:
        if not is_list(p):
            ctx.add("F-2.3", f"абз. {idx}", f"абзацний відступ {fi.cm if fi else 0:.2f} см ≠ 1,25: «{short(text)}»")
    ls, rule = eff_pf(p, "line_spacing"), eff_pf(p, "line_spacing_rule")
    if not (rule == WD_LINE_SPACING.ONE_POINT_FIVE or (isinstance(ls, float) and abs(ls - 1.5) < 0.01)):
        ctx.add("F-2.2", f"абз. {idx}", f"міжрядковий інтервал {ls} ≠ 1,5: «{short(text)}»")
    for attr in ("space_before", "space_after"):
        v = eff_pf(p, attr)
        if v is not None and v.pt > 0.5:
            ctx.add("F-2.3", f"абз. {idx}", f"{attr} = {v.pt:.0f} pt ≠ 0: «{short(text)}»")
    for r in p.runs:
        if not r.text.strip():
            continue
        name, size = eff_run(p, r, "name"), eff_run(p, r, "size")
        if name and name != "Times New Roman":
            ctx.add("F-2.1", f"абз. {idx}", f"шрифт {name}: «{short(r.text, 40)}»"); break
        if size is not None and abs(size.pt - 14) > 0.1:
            ctx.add("F-2.1", f"абз. {idx}", f"кегль {size.pt} ≠ 14: «{short(r.text, 40)}»"); break


def text_style_checks(ctx: Ctx, idx, text):
    if re.search(r"\bдан(ий|ого|ому|им|ій|а|у|і|их)\s+(розділ|підрозділ|робот|метод|алгоритм|модел|підхід|етап)", text, re.I):
        ctx.add("F-2.8", f"абз. {idx}", f"«даний» → «цей»: «{short(text)}»")
    if re.search(r"\bнаступн(і|ий|а|е|их|им)\s+(крок|етап|пункт|вимог|метод|задач|завдан|результат)", text, re.I):
        ctx.add("F-2.8", f"абз. {idx}", f"«наступні» → «такі»: «{short(text)}»")
    if re.search(r"\s-\s", text):
        ctx.add("F-2.5", f"абз. {idx}", f"дефіс замість тире «–»: «{short(text)}»")
    if re.search(r"\s—\s", text):
        ctx.add("F-2.5", f"абз. {idx}", f"довге тире «—», у шаблоні коротке «–»: «{short(text)}»")
    if re.search(r'"[^"]+"', text):
        ctx.add("F-2.5", f"абз. {idx}", f"прямі лапки → «ялинки»: «{short(text)}»")
    if re.search(r"\b(рис|табл)\. (?=\d)", text):
        ctx.add("F-2.6", f"абз. {idx}", "звичайний пробіл після «рис.»/«табл.» — потрібен нерозривний")


def run(path, maxn):
    doc = Document(path)
    ctx = Ctx(doc)
    check_page(ctx)

    blocks = list(iter_block(doc))
    started = False
    order_seen: list[str] = []
    chapter = None
    chapter_has_conclusion: dict[int, bool] = {}
    chapter_preamble: dict[int, bool] = {}
    sections_per: dict[str, list[str]] = defaultdict(list)
    fig_nums, tab_nums, formula_nums = [], [], []
    text_all = []
    refs_section, in_refs = [], False
    prev_text, prev_block = "", None
    last_kind = None   # "heading" | "text" | "figure" | "table" | "formula"
    pending_section_end: list[tuple[str, str]] = []
    abstract_counts = None
    keywords_line = None
    cur_struct = None
    in_where = False

    for idx, b in enumerate(blocks):
        if isinstance(b, Table):
            if started:
                if not RE_TAB.match(prev_text) and not prev_text.startswith(("Продовження таблиці", "Кінець таблиці")):
                    if not cur_struct == "ЗМІСТ":
                        ctx.add("F-9.2", f"табл. після абз. {idx-1}", f"над таблицею немає назви «Таблиця N.M – …» (перед нею: «{short(prev_text, 40)}»)")
                last_kind = "table"
            prev_block = b
            continue
        p: Paragraph = b
        text = p.text.strip()
        if not started:
            if text.upper() == "РЕФЕРАТ":
                started = True
            else:
                continue
        if not text:
            if has_drawing(p):
                last_kind = "figure"
            prev_block = p
            continue
        upper = text.upper()

        # рядки змісту (ручного або поля TOC): «Назва<tab>12»
        if cur_struct == "ЗМІСТ" and re.search(r"[\t ]\d+$", p.text.rstrip()):
            prev_text = text
            continue

        # ---- заголовки ----
        bold = all((eff_run(p, r, "bold") or False) for r in p.runs if r.text.strip())
        italic = all((eff_run(p, r, "italic") or False) for r in p.runs if r.text.strip())
        al = eff_pf(p, "alignment")
        heading_look = text == upper or al == WD_ALIGN_PARAGRAPH.CENTER or (p.style is not None and p.style.name.startswith("Heading"))
        struct = next((s for s in STRUCTURAL if upper.startswith(s) and len(text) <= len(s) + 45 and heading_look
                       and not text.endswith((",", ";")) and not re.match(r"(?i)^висновки до розділу", text)), None)
        m_ch, m_sub, m_sec = RE_CHAPTER.match(text), RE_SUBSEC.match(text), RE_SEC.match(text)
        is_app = re.match(r"^ДОДАТОК\s+(\S)", text)

        # легенда рисунка між зображенням і підписом (F-8.5) — не основний текст
        nxt = next((x.text.strip() for x in blocks[idx + 1: idx + 5] if isinstance(x, Paragraph) and x.text.strip()), "")
        if al == WD_ALIGN_PARAGRAPH.CENTER and RE_FIG.match(nxt) and not RE_FIG.match(text):
            prev_text, prev_block = text, p
            continue

        if struct or m_ch or is_app:
            kind = "РОЗДІЛ" if m_ch else ("ДОДАТОК" if is_app else struct)
            if pending_section_end and last_kind in ("figure", "table", "formula"):
                ctx.add("F-3.6", pending_section_end[-1][0], "розділ/підрозділ закінчується рисунком/таблицею/формулою")
            if chapter is not None and not chapter_has_conclusion.get(chapter):
                ctx.add("F-3.4", f"розділ {chapter}", "немає підрозділу «Висновки до розділу N»")
            chapter = None
            order_seen.append("ПЕРЕЛІК УМОВНИХ" if kind and kind.startswith("ПЕРЕЛІК УМОВНИХ") else kind)
            cur_struct = kind
            in_refs = kind == "ПЕРЕЛІК ПОСИЛАНЬ"
            if text != upper:
                ctx.add("F-5.1", f"абз. {idx}", f"заголовок не великими літерами: «{short(text)}»")
            if not bold:
                ctx.add("F-5.1", f"абз. {idx}", f"заголовок не напівжирний: «{short(text)}»")
            if al != WD_ALIGN_PARAGRAPH.CENTER:
                ctx.add("F-5.1", f"абз. {idx}", f"заголовок не по центру: «{short(text)}»")
            if text.endswith("."):
                ctx.add("F-5.7", f"абз. {idx}", f"крапка в кінці заголовка: «{short(text)}»")
            if m_ch:
                chapter = int(m_ch.group(1))
                chapter_has_conclusion[chapter] = False
                chapter_preamble[chapter] = False
            if is_app and is_app.group(1) not in APPENDIX_OK:
                ctx.add("F-13.2", f"абз. {idx}", f"заборонена літера додатка «{is_app.group(1)}»")
            pending_section_end = [(f"абз. {idx}", text)]
            last_kind = "heading"
            prev_text, prev_block = text, p
            continue

        if cur_struct == "ЗМІСТ":
            prev_text = text
            continue

        is_heading_like = (m_sub or m_sec) and len(text) < 160 and not text.endswith((";", ":")) and \
            (bold or italic or (p.style is not None and p.style.name.startswith("Heading")))
        if is_heading_like:
            if pending_section_end and last_kind in ("figure", "table", "formula"):
                ctx.add("F-3.6", pending_section_end[-1][0], "підрозділ закінчується рисунком/таблицею/формулою")
            if re.match(r"^\d+(\.\d+)+\.\s", text):
                ctx.add("F-5.7", f"абз. {idx}", f"крапка після номера: «{short(text)}»")
            if text.endswith("."):
                ctx.add("F-5.7", f"абз. {idx}", f"крапка в кінці заголовка: «{short(text)}»")
            if m_sub:
                if not italic:
                    ctx.add("F-5.4", f"абз. {idx}", f"пункт не курсивом: «{short(text)}»")
                key = f"{m_sub.group(1)}.{m_sub.group(2)}"
                sections_per[key].append(text)
            else:
                if not bold:
                    ctx.add("F-5.3", f"абз. {idx}", f"підрозділ не напівжирний: «{short(text)}»")
                sections_per[f"ch{m_sec.group(1)}"].append(text)
                if chapter is not None and int(m_sec.group(1)) != chapter:
                    ctx.add("F-5.12", f"абз. {idx}", f"номер підрозділу не відповідає розділу {chapter}: «{short(text)}»")
                if chapter is not None and not chapter_preamble.get(chapter) and m_sec.group(2) == "1":
                    ctx.add("F-3.4", f"абз. {idx}", f"розділ {chapter} не має преамбули перед 1-м підрозділом")
            if al == WD_ALIGN_PARAGRAPH.CENTER:
                ctx.add("F-5.3", f"абз. {idx}", f"підрозділ/пункт по центру — має бути з абзацного відступу: «{short(text)}»")
            pending_section_end.append((f"абз. {idx}", text))
            last_kind = "heading"
            prev_text, prev_block = text, p
            continue

        if re.match(r"^Висновки до розділу\s+\d+$", text):
            n = int(text.split()[-1])
            if chapter is not None:
                chapter_has_conclusion[chapter] = True
                if n != chapter:
                    ctx.add("F-5.5", f"абз. {idx}", f"«{text}» у розділі {chapter}")
            if not bold:
                ctx.add("F-5.5", f"абз. {idx}", "«Висновки до розділу N» не напівжирний")
            pending_section_end.append((f"абз. {idx}", text))
            last_kind = "heading"
            prev_text, prev_block = text, p
            continue

        # ---- підписи ----
        mf, mt = RE_FIG.match(text), RE_TAB.match(text)
        if re.match(r"^(Рис\.|Рис\s|Fig)", text):
            ctx.add("F-8.3", f"абз. {idx}", f"підпис має починатися з «Рисунок N.M – »: «{short(text)}»")
        if mf:
            num = f"{mf.group(1)}.{mf.group(2)}"
            fig_nums.append((num, idx))
            if mf.group(3) != EN:
                ctx.add("F-8.3", f"абз. {idx}", f"після номера рисунка має бути « – » (U+2013), а є «{mf.group(3)}»")
            if text.endswith("."):
                ctx.add("F-8.3", f"абз. {idx}", f"крапка в кінці підпису рисунка: «{short(text)}»")
            if al != WD_ALIGN_PARAGRAPH.CENTER:
                ctx.add("F-8.3", f"абз. {idx}", "підпис рисунка не по центру")
            if p.runs and not (eff_run(p, p.runs[0], "bold") or False):
                ctx.add("F-8.3", f"абз. {idx}", "«Рисунок N.M» не напівжирним")
            if last_kind != "figure" and not has_drawing(prev_block) if prev_block is not None else True:
                pass
            if chapter is not None and mf.group(1).isdigit() and int(mf.group(1)) != chapter:
                ctx.add("F-8.4", f"абз. {idx}", f"рисунок {num} у розділі {chapter}")
            last_kind = "figure"
            prev_text, prev_block = text, p
            continue
        if mt:
            num = f"{mt.group(1)}.{mt.group(2)}"
            tab_nums.append((num, idx))
            if mt.group(3) != EN:
                ctx.add("F-9.2", f"абз. {idx}", f"після номера таблиці має бути « – » (U+2013), а є «{mt.group(3)}»")
            if text.endswith("."):
                ctx.add("F-9.2", f"абз. {idx}", "крапка в кінці назви таблиці")
            if al == WD_ALIGN_PARAGRAPH.CENTER:
                ctx.add("F-9.2", f"абз. {idx}", "назва таблиці по центру — має бути ліворуч з абзацного відступу")
            if p.runs and not (eff_run(p, p.runs[0], "bold") or False):
                ctx.add("F-9.2", f"абз. {idx}", "«Таблиця N.M» не напівжирним")
            if chapter is not None and mt.group(1).isdigit() and int(mt.group(1)) != chapter:
                ctx.add("F-9.8", f"абз. {idx}", f"таблиця {num} у розділі {chapter}")
            prev_text, prev_block = text, p
            continue
        if text.startswith(("Продовження таблиці", "Кінець таблиці")):
            prev_text, prev_block = text, p
            continue

        # ---- формули ----
        mfn = RE_FORMULA_NUM.search(text)
        if (has_math(p) or "\t" in p.text) and mfn and len(text) < 200:
            formula_nums.append((mfn.group(1), idx))
            if chapter is not None and mfn.group(1).split(".")[0].isdigit() and int(mfn.group(1).split(".")[0]) != chapter:
                ctx.add("F-10.2", f"абз. {idx}", f"формула ({mfn.group(1)}) у розділі {chapter}")
            last_kind = "formula"
            prev_text, prev_block = text, p
            continue
        if re.match(r"^де\b", text) or (in_where and prev_text.endswith(";")):
            in_where = not text.endswith(".")
            if re.match(r"^де\s*:", text):
                ctx.add("F-10.4", f"абз. {idx}", "після «де» не ставлять двокрапку")
            fi = eff_pf(p, "first_line_indent")
            if fi is not None and fi.cm > 0.1:
                ctx.add("F-10.4", f"абз. {idx}", "«де» має бути без абзацного відступу")
            text_style_checks(ctx, idx, text)
            text_all.append((idx, text))
            last_kind = "text"
            prev_text, prev_block = text, p
            continue
        in_where = False
        if re.match(r"^де\s*:", text):
            ctx.add("F-10.4", f"абз. {idx}", "після «де» не ставлять двокрапку")
        if re.match(r"^де\b", text) and eff_pf(p, "first_line_indent") and eff_pf(p, "first_line_indent").cm > 0.1:
            ctx.add("F-10.4", f"абз. {idx}", "«де» має бути без абзацного відступу")

        # ---- перелік посилань ----
        if in_refs:
            refs_section.append(text)
            if not re.search(r"(URL:|DOI|doi\.org|С\.\s*\d|P\.\s*\d|с\.|p\.|arXiv)", text):
                ctx.add("F-12.5", f"абз. {idx}", f"джерело без URL/DOI/сторінок: «{short(text, 60)}»")
            if "URL" in text and "дата звернення" not in text and "DOI" not in text and "arXiv" not in text:
                ctx.add("F-12.5", f"абз. {idx}", "для URL немає «(дата звернення: ДД.ММ.РРРР)»")
            prev_text, prev_block = text, p
            continue

        # ---- звичайний текст ----
        if cur_struct in ("РЕФЕРАТ", "ABSTRACT"):
            mcount = re.match(r"^(Магістерська дисертація|Master'?s thesis):\s*(.*)", text)
            if mcount and cur_struct == "РЕФЕРАТ":
                abstract_counts = mcount.group(2)
            if text == upper and len(text) > 15 and "," in text:
                keywords_line = text
                if len([k for k in text.rstrip(".").split(",") if k.strip()]) > 10:
                    ctx.add("F-15.3", f"абз. {idx}", "понад 10 ключових слів")
                if not text.endswith("."):
                    ctx.add("F-15.3", f"абз. {idx}", "ключові слова мають закінчуватися крапкою")

        if chapter is not None and last_kind == "heading" and pending_section_end and RE_CHAPTER.match(pending_section_end[-1][1]):
            chapter_preamble[chapter] = True

        # списки
        if is_list(p) or re.match(r"^([–—•-]|\d+[.)])\s", text) or re.match(r"^([–—•-]|\d+[.)])\t", p.text):
            marker = re.match(r"^\s*([–—•-]|\d+[.)])", p.text)
            mk = marker.group(1) if marker else ""
            if mk in ("—", "•", "-"):
                ctx.add("F-7.2", f"абз. {idx}", f"маркер «{mk}» — у шаблоні тире «–»")
            if mk.endswith(")") or mk in ("–", "—", "•", "-"):
                if not re.search(r"[;.]$", text):
                    ctx.add("F-7.2", f"абз. {idx}", f"пункт маркованого списку має закінчуватися «;» або «.»: «{short(text)}»")
            elif mk.endswith("."):
                body = text[len(mk):].strip()
                if body and body[0].islower():
                    ctx.add("F-7.1", f"абз. {idx}", f"пункт нумерованого списку з малої літери: «{short(text)}»")
                if not text.endswith("."):
                    ctx.add("F-7.1", f"абз. {idx}", f"пункт нумерованого списку без крапки в кінці: «{short(text)}»")
        else:
            main_text_checks(ctx, idx, p, text)
        text_style_checks(ctx, idx, text)
        text_all.append((idx, text))
        last_kind = "text"
        prev_text, prev_block = text, p

    if chapter is not None and not chapter_has_conclusion.get(chapter):
        ctx.add("F-3.4", f"розділ {chapter}", "немає «Висновки до розділу N»")

    # ---- порядок структурних елементів ----
    pos = [ORDER.index(k) for k in order_seen if k in ORDER]
    for a, b in zip(pos, pos[1:]):
        if b < a:
            ctx.add("F-3.1", "структура", f"порушено порядок: {ORDER[b]} після {ORDER[a]}")
            break
    for must in ("РЕФЕРАТ", "ABSTRACT", "ЗМІСТ", "ВСТУП", "ВИСНОВКИ", "ПЕРЕЛІК ПОСИЛАНЬ"):
        if must not in order_seen:
            ctx.add("F-3.1", "структура", f"немає структурного елемента «{must}»")

    # ---- одиничні підрозділи/пункти ----
    for key, lst in sections_per.items():
        if len(lst) == 1:
            ctx.add("F-3.5", "структура", f"єдиний підрозділ/пункт у {key.replace('ch', 'розділі ')}: «{short(lst[0])}»")

    # ---- посилання на рисунки/таблиці/формули ----
    joined = [(i, t) for i, t in text_all]
    def first_ref(pattern):
        for i, t in joined:
            if re.search(pattern, t):
                return i
        return None
    for num, i in fig_nums:
        r = first_ref(r"рис\.\s?" + re.escape(num) + r"(?!\d)")
        if r is None:
            ctx.add("F-8.1", f"рис. {num}", "немає посилання в тексті")
        elif r > i:
            ctx.add("F-8.1", f"рис. {num}", "перше посилання стоїть після рисунка")
    for num, i in tab_nums:
        r = first_ref(r"табл\.\s?" + re.escape(num) + r"(?!\d)|таблиц\w+\s" + re.escape(num) + r"(?!\d)")
        if r is None:
            ctx.add("F-9.1", f"табл. {num}", "немає посилання в тексті")
        elif r > i:
            ctx.add("F-9.1", f"табл. {num}", "перше посилання стоїть після таблиці")
    for num, i in formula_nums:
        if first_ref(r"\(" + re.escape(num) + r"\)") is None:
            ctx.add("F-10.2", f"формула ({num})", "пронумерована, але немає посилання — зняти номер")
    for kind, nums in (("рисунків", fig_nums), ("таблиць", tab_nums), ("формул", formula_nums)):
        seen = set()
        for num, _ in nums:
            if num in seen:
                ctx.add("F-8.4", kind, f"номер {num} повторюється")
            seen.add(num)

    # ---- джерела ----
    cited_order = []
    for i, t in joined:
        for m in RE_CITE.finditer(t):
            for part in re.split(r"\s*,\s*", m.group(1)):
                if re.search(r"[–-]", part):
                    a, b = map(int, re.split(r"\s*[–-]\s*", part))
                    nums = range(a, b + 1)
                else:
                    nums = [int(part)]
                for n in nums:
                    if n not in cited_order:
                        cited_order.append(n)
    nref = len(refs_section)
    if nref:
        for n in cited_order:
            if n > nref:
                ctx.add("F-11.3", "посилання", f"[{n}] є в тексті, але в переліку лише {nref} джерел")
        missing = [n for n in range(1, nref + 1) if n not in cited_order]
        if missing:
            ctx.add("F-11.3", "перелік посилань", f"не процитовані в тексті: {missing[:30]}")
        firsts = [n for n in cited_order if n <= nref]
        if firsts and firsts != sorted(firsts):
            bad = next(n for k, n in enumerate(firsts) if n != k + 1) if firsts[:1] == [1] else firsts[0]
            ctx.add("F-12.2", "перелік посилань", f"порядок не за першою згадкою (перше відхилення: [{bad}])")

    # ---- числа в рефераті ----
    if abstract_counts:
        def num_before(word):
            m = re.search(r"(\d+)\s*" + word, abstract_counts)
            return int(m.group(1)) if m else None
        real_fig = len({n for n, _ in fig_nums})
        real_tab = len({n for n, _ in tab_nums})
        for word, real, rule in (("рис", real_fig, "рисунків"), ("табл", real_tab, "таблиць"), ("посилан", nref, "джерел")):
            stated = num_before(word)
            if stated is not None and real and stated != real:
                ctx.add("F-15.2", "РЕФЕРАТ", f"вказано {stated} {rule}, фактично {real}")

    # ---- вивід ----
    if not ctx.issues:
        print("Порушень не знайдено.")
        return 0
    print(f"Знайдено порушень: {len(ctx.issues)}\n")
    for k, (rule, where, msg) in enumerate(ctx.issues[:maxn], 1):
        print(f"{k}. [{rule}] {where}: {msg}")
    if len(ctx.issues) > maxn:
        print(f"… ще {len(ctx.issues) - maxn}")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("docx")
    ap.add_argument("--max", type=int, default=300)
    a = ap.parse_args()
    sys.exit(run(a.docx, a.max))
