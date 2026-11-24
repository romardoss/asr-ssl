"""Збирач .docx магістерської дисертації за правилами docs/diploma/rules.md.

Бере шаблон кафедри (титул + завдання, поля, колонтитул з номером сторінки),
видаляє демонстраційний вміст від РЕФЕРАТУ і дає функції, що одразу
застосовують пряме форматування з rules.md (ID правил у коментарях).

Залежності: python-docx; для формул з LaTeX — math2docx (необов'язково).

Приклад:
    from thesis_docx import ThesisDoc
    t = ThesisDoc()
    t.structural("РЕФЕРАТ")
    t.para("Магістерська дисертація: 98 с., 25 рис., 12 табл., 45 посилань, 2 додатки.")
    t.toc()
    t.structural("ВСТУП"); t.para("Текст [1].")
    t.chapter(1, "Огляд методів")
    t.para("Преамбула розділу.")
    t.section("1.1", "Назва підрозділу"); t.para("Текст (рис. 1.1).")
    t.figure("fig.png", "1.1", "Архітектура моделі", width_cm=15)
    t.chapter_conclusions(1); t.para("Висновки.")
    t.save("thesis.docx")
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[4]
TEMPLATE = ROOT / "docs/diploma/templates/83561_oppm_smai_md_2026.docx"

FONT = "Times New Roman"
SIZE = Pt(14)            # F-2.1
INDENT = Cm(1.25)        # F-2.3
DASH = "–"          # F-2.5
NBSP = " "          # F-2.6
APPENDIX_LETTERS = "АБВГДЕЖКЛМНПРСТУФХЦШЩЮЯ"  # F-13.2


def _set_font(run, size=SIZE, bold=None, italic=None, font=FONT):
    run.font.name = font
    run.font.size = size
    rpr = run._r.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(a), font)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def nbsp(text: str) -> str:
    """F-2.6: нерозривні пробіли після «рис.», «табл.», «с.», «Таблиця», «Рисунок», ініціалів."""
    text = re.sub(r"\b(рис\.|табл\.|с\.|С\.|Т\.|№|Рисунок|Таблиця|розділ[іу]?|додат(?:ок|ку|ки))\s+(?=[\dА-ЯA-Z])",
                  lambda m: m.group(1) + NBSP, text)
    text = re.sub(r"\b([А-ЯҐЄІЇA-Z]\.)\s+(?=[А-ЯҐЄІЇA-Z])", lambda m: m.group(1) + NBSP, text)
    return text


class ThesisDoc:
    def __init__(self, template: str | Path = TEMPLATE, keep_front: bool = True):
        self.doc = Document(str(template))
        self.body = self.doc.element.body
        self._clear_after_front(keep_front)
        self._fix_styles()
        self._first_in_section2 = True

    # ---------- службове ----------
    def _clear_after_front(self, keep_front: bool):
        """Залишає титул+завдання (розділ Word №1) і sectPr розділу №2, решту видаляє."""
        children = list(self.body)
        final_sect = children[-1] if children[-1].tag == qn("w:sectPr") else None
        # абзац, що завершує розділ №1, містить w:pPr/w:sectPr
        split = None
        for i, el in enumerate(children):
            if el.tag == qn("w:p") and el.find(qn("w:pPr") + "/" + qn("w:sectPr")) is not None:
                split = i
                break
        start = (split + 1) if (keep_front and split is not None) else 0
        for el in children[start:]:
            if el is not final_sect:
                self.body.remove(el)
        if not keep_front and split is not None:
            pass  # sectPr розділу 1 видалено разом з абзацом; лишився фінальний (розділ 2)

    def _fix_styles(self):
        st = self.doc.styles
        n = st["Normal"]
        n.font.name, n.font.size = FONT, SIZE
        pf = n.paragraph_format
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE        # F-2.2
        pf.first_line_indent = INDENT
        pf.space_before = pf.space_after = Pt(0)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        try:
            st["footnote text"].font.size = Pt(12)                    # F-2.9
        except KeyError:
            pass

    def _p(self, text="", align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=INDENT, bold=None, italic=None,
           size=SIZE, spacing=WD_LINE_SPACING.ONE_POINT_FIVE, keep_next=False, style=None):
        p = self.doc.add_paragraph(style=style)
        pf = p.paragraph_format
        p.alignment = align
        pf.first_line_indent = indent
        pf.left_indent = Cm(0)
        pf.space_before = pf.space_after = Pt(0)
        pf.line_spacing_rule = spacing
        pf.widow_control = True
        if keep_next:
            pf.keep_with_next = True
        if text:
            self._runs(p, text, bold=bold, italic=italic, size=size)
        return p

    def _runs(self, p, text, bold=None, italic=None, size=SIZE):
        """Підтримує **жирний** і *курсив* у тексті."""
        for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", nbsp(text)):
            if not part:
                continue
            b, i, t = bold, italic, part
            if part.startswith("**"):
                b, t = True, part[2:-2]
            elif part.startswith("*"):
                i, t = True, part[1:-1]
            _set_font(p.add_run(t), size=size, bold=b, italic=i)

    def blank(self, n=1):
        for _ in range(n):
            self._p(indent=Cm(0))

    def page_break(self):
        if self._first_in_section2:
            self._first_in_section2 = False   # перший елемент розділу 2 уже на новій сторінці
            return
        p = self._p(indent=Cm(0))
        p.add_run().add_break(WD_BREAK.PAGE)

    def _heading_style(self, p, level):
        """Прив'язка до Heading N, щоб працював автоматичний зміст (F-6.4)."""
        name = {1: "Heading 1", 2: "Heading 2", 3: "Heading 3"}[level]
        try:
            p.style = self.doc.styles[name]
        except KeyError:
            return
        pf = p.paragraph_format
        pf.space_before = pf.space_after = Pt(0)
        pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        pf.keep_with_next = True
        for r in p.runs:
            r.font.color.rgb = None

    # ---------- заголовки (F-5.*) ----------
    def structural(self, title: str):
        """РЕФЕРАТ, ABSTRACT, ЗМІСТ, ВСТУП, ВИСНОВКИ, ПЕРЕЛІК ПОСИЛАНЬ (F-5.1, F-3.2)."""
        self.page_break()
        p = self._p(title.upper(), align=WD_ALIGN_PARAGRAPH.CENTER, indent=Cm(0), bold=True, keep_next=True)
        if title.upper() != "ЗМІСТ":
            self._heading_style(p, 1)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        self.blank(1)                                                   # F-5.10
        return p

    def chapter(self, n: int, title: str):
        """РОЗДІЛ N НАЗВА (F-5.2). Після нього одразу преамбула (F-3.4)."""
        self.page_break()
        p = self._p(f"РОЗДІЛ {n} {title.upper()}", align=WD_ALIGN_PARAGRAPH.CENTER, indent=Cm(0),
                    bold=True, keep_next=True)
        self._heading_style(p, 1)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        self.blank(1)
        self._last_heading = "chapter"
        return p

    def _sub(self, text, italic, level, blank_before):
        if blank_before:
            self.blank(blank_before)
        p = self._p(text, align=WD_ALIGN_PARAGRAPH.LEFT, indent=INDENT,
                    bold=not italic, italic=italic, keep_next=True)
        self._heading_style(p, level)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.first_line_indent = INDENT
        self.blank(1)
        return p

    def section(self, num: str, title: str):
        """1.1 Назва — напівжирний, з відступу 1,25 (F-5.3, F-5.7)."""
        return self._sub(f"{num} {title}", italic=False, level=2, blank_before=1)

    def subsection(self, num: str, title: str):
        """1.1.1 Назва — курсив (F-5.4)."""
        return self._sub(f"{num} {title}", italic=True, level=3, blank_before=1)

    def chapter_conclusions(self, n: int):
        return self._sub(f"Висновки до розділу {n}", italic=False, level=2, blank_before=1)

    def appendix(self, letter: str, title: str):
        assert letter in APPENDIX_LETTERS, f"F-13.2: літера {letter} заборонена"
        self.page_break()
        p = self._p(f"ДОДАТОК {letter}", align=WD_ALIGN_PARAGRAPH.CENTER, indent=Cm(0), bold=True, keep_next=True)
        self._heading_style(p, 1)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        # назву додатка — в тому ж абзаці через розрив рядка, щоб зміст показав її
        r = p.add_run(); r.add_break()
        _set_font(r, bold=True)
        _set_font(p.add_run(title.upper()), bold=True)
        self.blank(1)
        return p

    # ---------- текст ----------
    def para(self, text: str, **kw):
        """Абзац: TNR 14, 1,5, відступ 1,25, за шириною (F-2.1–F-2.3)."""
        return self._p(text, **kw)

    def numbered_list(self, items: list[str]):
        """F-7.1: перед списком речення з крапкою; пункти з великої, з крапкою."""
        for i, it in enumerate(items, 1):
            it = it[0].upper() + it[1:]
            if not it.endswith("."):
                it += "."
            self._list_item(f"{i}.", it)

    def bullet_list(self, items: list[str]):
        """F-7.2: перед списком двокрапка; «–», з малої, «;», останній «.»."""
        for i, it in enumerate(items):
            it = it.rstrip(".;") + ("." if i == len(items) - 1 else ";")
            self._list_item(DASH, it)

    def inline_list(self, items: list[str]):
        """F-7.3: 1), 2) з малої, «;», останній «.»."""
        for i, it in enumerate(items):
            it = it.rstrip(".;") + ("." if i == len(items) - 1 else ";")
            self._list_item(f"{i + 1})", it)

    def _list_item(self, marker, text):
        p = self._p(indent=INDENT)
        pf = p.paragraph_format
        pf.tab_stops.add_tab_stop(INDENT + Cm(0.75))
        self._runs(p, f"{marker}\t{text}")
        return p

    # ---------- рисунки (F-8.*) ----------
    def figure(self, path: str | Path, num: str, caption: str, width_cm: float = 15.0,
               legend: list[str] | None = None, source: str | None = None, ai_note: str | None = None):
        self.blank(1)                                                       # F-8.7
        p = self._p(indent=Cm(0), align=WD_ALIGN_PARAGRAPH.CENTER, keep_next=True,
                    spacing=WD_LINE_SPACING.SINGLE)
        p.add_run().add_picture(str(path), width=Cm(min(width_cm, 16.5)))   # F-8.2
        for line in legend or []:                                           # F-8.5
            self._p(line, indent=Cm(0), align=WD_ALIGN_PARAGRAPH.CENTER, size=Pt(12),
                    spacing=WD_LINE_SPACING.SINGLE, keep_next=True)
        cap = self._p(indent=Cm(0), align=WD_ALIGN_PARAGRAPH.CENTER)
        _set_font(cap.add_run(f"Рисунок{NBSP}{num}"), bold=True)            # F-8.3
        _set_font(cap.add_run(f" {DASH} {nbsp(caption).rstrip('.')}"))
        if source:
            _set_font(cap.add_run(f" [{source}]"))                          # F-8.6
        if ai_note:
            self.footnote(cap, ai_note)                                     # F-16.3
        self.blank(1)
        return cap

    # ---------- таблиці (F-9.*) ----------
    def table(self, rows: list[list[str]], num: str, caption: str, font_pt: int = 12,
              header_rows: int = 1, col_widths_cm: list[float] | None = None,
              numeric_cols: set[int] | None = None, notes: list[str] | None = None):
        self.blank(1)                                                       # F-9.3
        cap = self._p(indent=INDENT, align=WD_ALIGN_PARAGRAPH.LEFT, keep_next=True)
        _set_font(cap.add_run(f"Таблиця{NBSP}{num}"), bold=True)            # F-9.2
        _set_font(cap.add_run(f" {DASH} {nbsp(caption).rstrip('.')}"))
        ncol = max(len(r) for r in rows)
        t = self.doc.add_table(rows=len(rows), cols=ncol)
        t.style = self.doc.styles["Table Grid"]                             # F-9.4 усі межі
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        numeric_cols = numeric_cols or set()
        for i, row in enumerate(rows):
            tr = t.rows[i]._tr
            if i < header_rows:                                             # F-9.7 повтор шапки
                trPr = tr.get_or_add_trPr()
                h = OxmlElement("w:tblHeader"); h.set(qn("w:val"), "true"); trPr.append(h)
            cant = OxmlElement("w:cantSplit"); tr.get_or_add_trPr().append(cant)
            for j in range(ncol):
                cell = t.cell(i, j)
                txt = row[j] if j < len(row) else ""
                cp = cell.paragraphs[0]
                cp.paragraph_format.first_line_indent = Cm(0)
                cp.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE   # F-9.5
                cp.paragraph_format.space_before = cp.paragraph_format.space_after = Pt(0)
                if i < header_rows or j in numeric_cols:
                    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    cp.alignment = WD_ALIGN_PARAGRAPH.LEFT
                _set_font(cp.add_run(nbsp(txt or DASH)), size=Pt(font_pt))  # F-9.6
                if col_widths_cm:
                    cell.width = Cm(col_widths_cm[j])
        for line in notes or []:                                            # F-9.9
            self._p(line, size=Pt(12), spacing=WD_LINE_SPACING.SINGLE)
        self.blank(1)
        return t

    # ---------- формули (F-10.*) ----------
    def formula(self, latex: str, num: str | None = None, trailing: str = ","):
        """Формула по центру (таб 8,5 см), номер праворуч (таб 17 см). trailing: ',' перед «де», інакше '.'."""
        p = self._p(indent=Cm(0), align=WD_ALIGN_PARAGRAPH.LEFT)
        ts = p.paragraph_format.tab_stops
        ts.add_tab_stop(Cm(8.5), WD_TAB_ALIGNMENT.CENTER)
        ts.add_tab_stop(Cm(17), WD_TAB_ALIGNMENT.RIGHT)
        _set_font(p.add_run("\t"))
        try:
            import math2docx
            math2docx.add_math(p, latex)                                    # F-10.6 OMML
        except Exception:
            _set_font(p.add_run(latex), italic=True)
        _set_font(p.add_run(trailing))                                      # F-10.3
        if num:                                                             # F-10.2
            _set_font(p.add_run(f"\t({num})"))
        return p

    def where(self, items: list[tuple[str, str]]):
        """F-10.4: «де» без двокрапки, без абзацного відступу, у порядку появи."""
        for k, (sym, desc) in enumerate(items):
            last = k == len(items) - 1
            p = self._p(indent=Cm(0), align=WD_ALIGN_PARAGRAPH.JUSTIFY)
            if k == 0:
                _set_font(p.add_run("де "))
            try:
                import math2docx
                math2docx.add_math(p, sym)
            except Exception:
                _set_font(p.add_run(sym), italic=True)
            _set_font(p.add_run(f" {DASH} {nbsp(desc).rstrip('.;')}{'.' if last else ';'}"))

    # ---------- виноски ----------
    def footnote(self, paragraph, text: str):
        """Проста виноска як надрядковий номер + абзац у кінці сторінки неможлива в python-docx;
        додаємо справжню виноску через footnotes part, якщо він є у шаблоні."""
        part = self.doc.part
        fn_part = None
        for rel in part.rels.values():
            if rel.reltype.endswith("/footnotes"):
                fn_part = rel.target_part
        if fn_part is None:
            _set_font(paragraph.add_run(f" ({text})"), size=Pt(12))
            return
        from docx.oxml import parse_xml
        root = parse_xml(fn_part.blob) if not hasattr(fn_part, "element") else fn_part.element
        ids = [int(f.get(qn("w:id"))) for f in root.findall(qn("w:footnote"))]
        new_id = max(ids + [0]) + 1
        W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        fn = parse_xml(
            f'<w:footnote xmlns:w="{W}" w:id="{new_id}"><w:p><w:pPr><w:pStyle w:val="FootnoteText"/></w:pPr>'
            f'<w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr><w:footnoteRef/></w:r>'
            f'<w:r><w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}"/><w:sz w:val="24"/></w:rPr>'
            f'<w:t xml:space="preserve"> {text}</w:t></w:r></w:p></w:footnote>')
        root.append(fn)
        if not hasattr(fn_part, "element"):
            from lxml import etree
            fn_part._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
        r = paragraph.add_run()
        rpr = r._r.get_or_add_rPr()
        va = OxmlElement("w:vertAlign"); va.set(qn("w:val"), "superscript"); rpr.append(va)
        ref = OxmlElement("w:footnoteReference"); ref.set(qn("w:id"), str(new_id)); r._r.append(ref)

    # ---------- зміст, перелік посилань ----------
    def toc(self):
        """F-6.*: поле TOC; оновити в Word (F9) або LibreOffice після збирання."""
        self.structural("ЗМІСТ")
        p = self._p(indent=Cm(0), align=WD_ALIGN_PARAGRAPH.LEFT)
        r = p.add_run()
        for tag, attr in (("w:fldChar", "begin"), ("w:instrText", None), ("w:fldChar", "separate")):
            el = OxmlElement(tag)
            if attr:
                el.set(qn("w:fldCharType"), attr)
            else:
                el.set(qn("xml:space"), "preserve"); el.text = 'TOC \\o "1-3" \\h \\z \\u'
            r._r.append(el)
        _set_font(p.add_run("Оновіть зміст (F9)."))
        r2 = p.add_run(); e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), "end"); r2._r.append(e)
        upd = OxmlElement("w:updateFields"); upd.set(qn("w:val"), "true")
        self.doc.settings.element.append(upd)

    def references(self, items: list[str]):
        """F-12.*: ПЕРЕЛІК ПОСИЛАНЬ, нумерація 1., 2., висячий відступ."""
        self.structural("ПЕРЕЛІК ПОСИЛАНЬ")
        for i, it in enumerate(items, 1):
            p = self._p(indent=INDENT, align=WD_ALIGN_PARAGRAPH.LEFT)
            p.paragraph_format.tab_stops.add_tab_stop(INDENT + Cm(0.75))
            self._runs(p, f"{i}.\t{it}")

    def save(self, path: str | Path):
        self.doc.save(str(path))
        return Path(path)


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "thesis_demo.docx"
    t = ThesisDoc()
    t.structural("РЕФЕРАТ")
    t.para("Магістерська дисертація: 98 с., 25 рис., 12 табл., 45 посилань, 2 додатки.")
    t.para("РОЗПІЗНАВАННЯ МОВЛЕННЯ, САМОНАВЧАННЯ, ПСЕВДОРОЗМІТКА.")
    t.toc()
    t.structural("ВСТУП")
    t.para("Актуальність теми зумовлена розвитком систем розпізнавання мовлення [1].")
    t.chapter(1, "Огляд методів")
    t.para("У цьому розділі розглянуто сучасні підходи.")
    t.section("1.1", "Моделі розпізнавання")
    t.para("Основні метрики наведено в табл. 1.1 і формулі (1.1).")
    t.table([["Модель", "WER, %"], ["Whisper", "12,3"], ["wav2vec 2.0", "14,1"]], "1.1", "Порівняння моделей",
            numeric_cols={1})
    t.formula(r"WER = \frac{S + D + I}{N}", "1.1", trailing=",")
    t.where([("S", "кількість замін"), ("D", "кількість видалень"), ("I", "кількість вставок"),
             ("N", "кількість слів еталона")])
    t.para("Етапи навчання такі.")
    t.numbered_list(["попереднє навчання", "донавчання"])
    t.para("Використано такі дані:")
    t.bullet_list(["Common Voice", "FLEURS"])
    t.chapter_conclusions(1)
    t.para("Обрано модель для подальших експериментів.")
    t.structural("ВИСНОВКИ")
    t.para("1. Проаналізовано методи.")
    t.references(["Radford A. et al. Robust Speech Recognition via Large-Scale Weak Supervision. "
                  "arXiv:2212.04356. 2022. DOI: 10.48550/arXiv.2212.04356."])
    t.appendix("А", "Лістинг програми")
    t.para("Код наведено нижче.")
    print(t.save(out))
