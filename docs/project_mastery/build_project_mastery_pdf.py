#!/usr/bin/env python3
"""
build_project_mastery_pdf.py
============================
PROJECT_MASTERY_GUIDE.md -> docs/PROJECT_MASTERY_GUIDE.pdf

Markdown kaynağı KANONİK; PDF ondan deterministik olarak üretilir. Salt-okunur
bir build'dir: yalnızca Markdown + figures/ okur, docs/PROJECT_MASTERY_GUIDE.pdf
yazar. Hiçbir bilimsel çıktı/dosya değiştirilmez.

Bağımlılık: fpdf2 (>=2.8), DejaVu TTF fontları (sistemde). WeasyPrint/pandoc/
LaTeX GEREKTİRMEZ (bu ortamda yalnızca saf-Python fpdf2 mevcuttur).

Çalıştırma (repo kökünden):
    python docs/project_mastery/build_project_mastery_pdf.py

Desteklenen Markdown alt kümesi (bu handbook'un kullandığı):
    # ## ### ####            başlıklar (## ve ### -> PDF bookmark + TOC)
    paragraf                 satır kaydırmalı; **kalın** ve `kod` inline
    - madde / 1. numaralı     listeler
    | tablo | ...            pipe tabloları (ilk satır başlık)
    ```kod bloğu```          fenced code (monospace, gri arka plan)
    > callout                blockquote callout kutuları
    ![alt](figures/x.png)    görseller
    ---                       yatay çizgi / sayfa ayracı
"""
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MD_PATH = HERE / "PROJECT_MASTERY_GUIDE.md"
OUT_PDF = ROOT / "docs" / "PROJECT_MASTERY_GUIDE.pdf"
FIG_DIR = HERE / "figures"

FONT_DIR = "/usr/share/fonts/truetype/dejavu"

# Renkler
CLR_TEXT = (17, 17, 17)
CLR_H1 = (26, 76, 122)
CLR_H2 = (26, 76, 122)
CLR_H3 = (60, 100, 70)
CLR_H4 = (120, 79, 163)
CLR_RULE = (200, 200, 200)
CLR_CODE_BG = (244, 246, 248)
CLR_CODE_TX = (30, 30, 30)
CLR_TABLE_HDR = (222, 231, 243)
CLR_TABLE_ALT = (247, 249, 251)
CLR_LINK = (26, 76, 122)

CALLOUT_STYLES = {
    "Neden önemli?": ((219, 235, 218), (60, 141, 64)),
    "Sık yapılan hata": ((251, 227, 227), (178, 59, 59)),
    "Leakage riski": ((251, 227, 227), (178, 59, 59)),
    "Claim sınırı": ((251, 227, 227), (178, 59, 59)),
    "Kodda nerede?": ((235, 240, 245), (45, 106, 159)),
    "Çıktıda nerede?": ((235, 240, 245), (45, 106, 159)),
    "Bunu kendin kontrol et": ((236, 227, 243), (122, 79, 163)),
    "Not": ((240, 240, 240), (120, 120, 120)),
    "Discrepancy": ((253, 240, 220), (181, 101, 29)),
}


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


class Guide(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 16, 18)
        self._register_fonts()
        self.commit = git_commit()
        self.gen_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        self.on_cover = False
        self.set_title("Uydu Termal Dijital İkiz — Proje Ustalık Rehberi")
        self.set_author("Proje sahibi")
        self.set_creator("build_project_mastery_pdf.py (fpdf2)")

    def _register_fonts(self):
        # DejaVu Sans has no oblique file in this environment; italic styles
        # reuse the upright faces (glyphs render correctly, just not slanted).
        self.add_font("DejaVu", "", f"{FONT_DIR}/DejaVuSans.ttf")
        self.add_font("DejaVu", "B", f"{FONT_DIR}/DejaVuSans-Bold.ttf")
        self.add_font("DejaVu", "I", f"{FONT_DIR}/DejaVuSans.ttf")
        self.add_font("DejaVu", "BI", f"{FONT_DIR}/DejaVuSans-Bold.ttf")
        self.add_font("Mono", "", f"{FONT_DIR}/DejaVuSansMono.ttf")
        self.add_font("Mono", "B", f"{FONT_DIR}/DejaVuSansMono-Bold.ttf")

    # --- footer with page number ---
    def footer(self):
        if self.on_cover:
            return
        self.set_y(-14)
        self.set_font("DejaVu", "", 7.5)
        self.set_text_color(130, 130, 130)
        self.cell(0, 6, f"Proje Ustalık Rehberi · commit {self.commit[:10]} · {self.gen_ts}",
                  align="L")
        self.cell(0, 6, f"Sayfa {self.page_no()}", align="R")
        self.set_text_color(*CLR_TEXT)


def inline_segments(text):
    """State-machine tokenizer: ** toggles bold, ` toggles code (independently).
    Returns list of (text, mode) where mode in {'', 'B', 'code'} — 'code' wins
    when a span is both bold and code (mono styling is more important there)."""
    segs = []
    buf = []
    bold = False
    code = False
    i = 0
    n = len(text)

    def flush():
        if buf:
            mode = "code" if code else ("B" if bold else "")
            segs.append(("".join(buf), mode))
            buf.clear()

    while i < n:
        if not code and text[i:i + 2] == "**":
            flush(); bold = not bold; i += 2; continue
        if text[i] == "`":
            flush(); code = not code; i += 1; continue
        buf.append(text[i]); i += 1
    flush()
    return segs


def write_inline(pdf, text, size=10, lh=5.4, color=CLR_TEXT):
    pdf.set_text_color(*color)
    for seg, mode in inline_segments(text):
        if mode == "code":
            pdf.set_font("Mono", "", size - 1.2)
            pdf.set_text_color(178, 59, 59)
        elif mode == "B":
            pdf.set_font("DejaVu", "B", size)
            pdf.set_text_color(*color)
        else:
            pdf.set_font("DejaVu", "", size)
            pdf.set_text_color(*color)
        pdf.write(lh, seg)
    pdf.ln(lh)
    pdf.set_text_color(*CLR_TEXT)


# --- Markdown parsing to blocks ---
def parse_blocks(md):
    lines = md.split("\n")
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.strip().startswith("```"):
            code = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i += 1
            i += 1
            blocks.append(("code", code)); continue
        if re.match(r"^\s*\|.*\|\s*$", line) and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            tbl = [line]
            i += 1
            while i < n and re.match(r"^\s*\|.*\|\s*$", lines[i]):
                tbl.append(lines[i]); i += 1
            blocks.append(("table", tbl)); continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            blocks.append(("h%d" % len(m.group(1)), m.group(2).strip())); i += 1; continue
        if re.match(r"^!\[.*\]\(.*\)\s*$", line):
            mm = re.match(r"^!\[(.*)\]\((.*)\)\s*$", line)
            blocks.append(("img", (mm.group(1), mm.group(2)))); i += 1; continue
        if line.strip() in ("---", "***", "___"):
            blocks.append(("hr", None)); i += 1; continue
        if line.strip().startswith(">"):
            quote = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip()); i += 1
            blocks.append(("quote", quote)); continue
        if re.match(r"^\s*[-*]\s+", line):
            items = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                indent = len(lines[i]) - len(lines[i].lstrip())
                items.append((indent, re.sub(r"^\s*[-*]\s+", "", lines[i]))); i += 1
            blocks.append(("ul", items)); continue
        if re.match(r"^\s*\d+\.\s+", line):
            items = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                items.append(re.sub(r"^\s*\d+\.\s+", "", lines[i])); i += 1
            blocks.append(("ol", items)); continue
        if line.strip() == "":
            i += 1; continue
        # paragraph (accumulate until blank)
        para = [line]
        i += 1
        while i < n and lines[i].strip() != "" and not re.match(r"^(#{1,6}\s|>|\s*[-*]\s|\s*\d+\.\s|\|)", lines[i]) and not lines[i].strip().startswith("```") and not re.match(r"^!\[", lines[i]) and lines[i].strip() not in ("---", "***", "___"):
            para.append(lines[i]); i += 1
        blocks.append(("p", " ".join(x.strip() for x in para)))
    return blocks


def render_table(pdf, tbl):
    rows = []
    for r in tbl:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        rows.append(cells)
    if len(rows) >= 2 and re.match(r"^[\s:-]+$", "".join(rows[1])):
        header = rows[0]; body = rows[2:]
    else:
        header = rows[0]; body = rows[1:]
    ncol = len(header)
    pdf.set_font("DejaVu", "", 8)
    epw = pdf.epw
    # heuristic column widths from max content length
    maxlen = [max((len(row[c]) if c < len(row) else 0) for row in [header] + body) for c in range(ncol)]
    tot = sum(maxlen) or 1
    widths = [max(0.06, ml / tot) * epw for ml in maxlen]
    # clamp
    scale = epw / sum(widths)
    widths = [w * scale for w in widths]
    with pdf.table(col_widths=[w / epw * 100 for w in widths], text_align="LEFT",
                   first_row_as_headings=True, headings_style=_fp_style(),
                   line_height=4.6, cell_fill_color=CLR_TABLE_ALT,
                   cell_fill_mode=_row_fill(), width=epw,
                   borders_layout="MINIMAL") as table:
        hr = table.row()
        for c in range(ncol):
            hr.cell(_strip_md(header[c]) if c < len(header) else "")
        for row in body:
            tr = table.row()
            for c in range(ncol):
                tr.cell(_strip_md(row[c]) if c < len(row) else "")
    pdf.ln(1.5)


def _strip_md(s):
    s = s.replace("**", "").replace("`", "")
    return s


from fpdf.fonts import FontFace
from fpdf.enums import TableCellFillMode


def _fp_style():
    return FontFace(emphasis="BOLD", color=(17, 17, 17), fill_color=CLR_TABLE_HDR)


def _row_fill():
    return TableCellFillMode.EVEN_ROWS


def render_blocks(pdf, blocks, toc_entries):
    for kind, payload in blocks:
        if kind == "h1":
            pdf.add_page()
            pdf.set_font("DejaVu", "B", 20)
            pdf.set_text_color(*CLR_H1)
            pdf.multi_cell(0, 10, _strip_md(payload), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_draw_color(*CLR_H1)
            pdf.set_line_width(0.6)
            y = pdf.get_y() + 1
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(5)
            pdf.set_text_color(*CLR_TEXT)
            pdf.start_section(_strip_md(payload), level=0)
        elif kind == "h2":
            if pdf.get_y() > pdf.h - 60:
                pdf.add_page()
            pdf.ln(3)
            pdf.set_font("DejaVu", "B", 15)
            pdf.set_text_color(*CLR_H2)
            pdf.multi_cell(0, 8, _strip_md(payload), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_draw_color(*CLR_RULE)
            pdf.set_line_width(0.3)
            y = pdf.get_y() + 0.5
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(2.5)
            pdf.set_text_color(*CLR_TEXT)
            pdf.start_section(_strip_md(payload), level=1)
        elif kind == "h3":
            if pdf.get_y() > pdf.h - 45:
                pdf.add_page()
            pdf.ln(2)
            pdf.set_font("DejaVu", "B", 12)
            pdf.set_text_color(*CLR_H3)
            pdf.multi_cell(0, 6.6, _strip_md(payload), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
            pdf.set_text_color(*CLR_TEXT)
            pdf.start_section(_strip_md(payload), level=2)
        elif kind == "h4":
            pdf.ln(1.5)
            pdf.set_font("DejaVu", "B", 10.5)
            pdf.set_text_color(*CLR_H4)
            pdf.multi_cell(0, 5.8, _strip_md(payload), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_text_color(*CLR_TEXT)
            pdf.ln(0.5)
        elif kind in ("h5", "h6"):
            pdf.set_font("DejaVu", "BI", 10)
            pdf.multi_cell(0, 5.5, _strip_md(payload), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        elif kind == "p":
            write_inline(pdf, payload, size=10, lh=5.4)
            pdf.ln(1.4)
        elif kind == "ul":
            for indent, item in payload:
                pad = 4 + (indent // 2)
                pdf.set_x(pdf.l_margin + pad)
                pdf.set_font("DejaVu", "B", 10)
                pdf.write(5.2, "•  ")
                write_inline(pdf, item, size=10, lh=5.2)
            pdf.ln(1.2)
        elif kind == "ol":
            for idx, item in enumerate(payload, 1):
                pdf.set_x(pdf.l_margin + 4)
                pdf.set_font("DejaVu", "B", 10)
                pdf.write(5.2, f"{idx}.  ")
                write_inline(pdf, item, size=10, lh=5.2)
            pdf.ln(1.2)
        elif kind == "code":
            render_code(pdf, payload)
        elif kind == "table":
            _ensure_space(pdf, 30)
            render_table(pdf, payload)
        elif kind == "quote":
            render_callout(pdf, payload)
        elif kind == "img":
            render_image(pdf, payload)
        elif kind == "hr":
            pdf.ln(1)
            pdf.set_draw_color(*CLR_RULE)
            pdf.set_line_width(0.2)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)


def _ensure_space(pdf, mm):
    if pdf.get_y() > pdf.h - pdf.b_margin - mm:
        pdf.add_page()


def render_code(pdf, code_lines):
    _ensure_space(pdf, 12 + 4 * min(len(code_lines), 6))
    pdf.set_font("Mono", "", 8)
    lh = 4.3
    pad = 2
    # wrap long lines
    epw = pdf.epw - 2 * pad
    wrapped = []
    for ln in code_lines:
        ln = ln.replace("\t", "    ")
        if pdf.get_string_width(ln) <= epw:
            wrapped.append(ln)
        else:
            cur = ""
            for ch in ln:
                if pdf.get_string_width(cur + ch) > epw:
                    wrapped.append(cur); cur = "    " + ch
                else:
                    cur += ch
            wrapped.append(cur)
    h = lh * len(wrapped) + 2 * pad
    x0, y0 = pdf.l_margin, pdf.get_y()
    pdf.set_fill_color(*CLR_CODE_BG)
    pdf.set_draw_color(220, 224, 228)
    pdf.rect(x0, y0, pdf.epw, h, style="DF")
    pdf.set_xy(x0 + pad, y0 + pad)
    pdf.set_text_color(*CLR_CODE_TX)
    for w in wrapped:
        pdf.set_x(x0 + pad)
        pdf.cell(0, lh, w, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*CLR_TEXT)
    pdf.ln(2.5)


def render_callout(pdf, quote_lines):
    text = "\n".join(quote_lines).strip()
    title = None
    m = re.match(r"^\*\*(.+?):\*\*\s*(.*)$", text, re.S)
    body = text
    if m:
        title = m.group(1).strip()
        body = m.group(2).strip()
    bg, bar = CALLOUT_STYLES.get(title, ((240, 240, 240), (120, 120, 120)))
    _ensure_space(pdf, 22)
    pdf.ln(1)
    x0, y0 = pdf.l_margin, pdf.get_y()
    # measure: render into temp by computing lines via multi_cell dry run
    pad = 3
    inner_w = pdf.epw - 2 * pad - 2
    # estimate height
    pdf.set_font("DejaVu", "", 9.2)
    start_y = pdf.get_y()
    # draw placeholder: we render text first at x with left bar, compute after
    pdf.set_xy(x0 + pad + 2, y0 + pad)
    if title:
        pdf.set_font("DejaVu", "B", 9.4)
        pdf.set_text_color(*bar)
        pdf.multi_cell(inner_w, 5.0, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(x0 + pad + 2)
    pdf.set_font("DejaVu", "", 9.2)
    pdf.set_text_color(*CLR_TEXT)
    # inline render body with wrapping using multi_cell (strip inline md markers but keep readable)
    for para in body.split("\n"):
        pdf.set_x(x0 + pad + 2)
        _multicell_inline(pdf, para, inner_w, 4.8)
    y1 = pdf.get_y() + pad
    # draw background + left bar behind (retro fill by re-drawing rect with transparency? fpdf no alpha easily)
    # Instead: draw rect first is impossible now; use a border box around
    pdf.set_draw_color(*bar)
    pdf.set_line_width(0.2)
    pdf.set_fill_color(*bg)
    # left accent bar
    pdf.rect(x0, start_y - pad + pad, 1.6, y1 - (start_y - pad + pad), style="F")
    pdf.set_draw_color(*[min(255, c + 30) for c in bar])
    pdf.rect(x0, y0, pdf.epw, y1 - y0, style="D")
    pdf.ln(3)
    pdf.set_text_color(*CLR_TEXT)


def _multicell_inline(pdf, text, w, lh):
    """multi_cell benzeri kaydırma ama **bold**/`code` inline destekli (basit)."""
    words = text.split(" ")
    x_start = pdf.get_x()
    space = pdf.get_string_width(" ")
    cur_w = 0
    for word in words:
        segs = inline_segments(word)
        ww = 0
        for s, mode in segs:
            if mode == "code":
                pdf.set_font("Mono", "", lh + 3.0)
            elif mode == "B":
                pdf.set_font("DejaVu", "B", 9.2)
            else:
                pdf.set_font("DejaVu", "", 9.2)
            ww += pdf.get_string_width(s)
        if cur_w + ww > w and cur_w > 0:
            pdf.ln(lh); pdf.set_x(x_start); cur_w = 0
        for s, mode in segs:
            if mode == "code":
                pdf.set_font("Mono", "", 8.0); pdf.set_text_color(178, 59, 59)
            elif mode == "B":
                pdf.set_font("DejaVu", "B", 9.2); pdf.set_text_color(*CLR_TEXT)
            else:
                pdf.set_font("DejaVu", "", 9.2); pdf.set_text_color(*CLR_TEXT)
            pdf.write(lh, s)
        pdf.write(lh, " ")
        cur_w += ww + space
    pdf.ln(lh)
    pdf.set_text_color(*CLR_TEXT)


def render_image(pdf, payload):
    alt, path = payload
    img_path = (HERE / path) if not os.path.isabs(path) else Path(path)
    if not img_path.exists():
        img_path2 = ROOT / path
        img_path = img_path2 if img_path2.exists() else img_path
    if not img_path.exists():
        write_inline(pdf, f"[GÖRSEL BULUNAMADI: {path}]", color=(178, 59, 59))
        return
    # scale to width, add caption
    from PIL import Image
    with Image.open(img_path) as im:
        iw, ih = im.size
    max_w = pdf.epw
    disp_w = max_w
    disp_h = disp_w * ih / iw
    max_h = pdf.h - pdf.t_margin - pdf.b_margin - 20
    if disp_h > max_h:
        disp_h = max_h
        disp_w = disp_h * iw / ih
    _ensure_space(pdf, disp_h + 8)
    x = pdf.l_margin + (pdf.epw - disp_w) / 2
    pdf.image(str(img_path), x=x, w=disp_w, h=disp_h)
    pdf.ln(1)
    if alt:
        pdf.set_font("DejaVu", "I", 8)
        pdf.set_text_color(110, 110, 110)
        pdf.multi_cell(0, 4.4, alt, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*CLR_TEXT)
    pdf.ln(3)


def render_cover(pdf):
    pdf.on_cover = True
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("DejaVu", "B", 26)
    pdf.set_text_color(*CLR_H1)
    pdf.multi_cell(0, 12, "Uydu Tabanlı Termal Dijital İkiz", align="C",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DejaVu", "B", 18)
    pdf.multi_cell(0, 10, "Proje Ustalık Rehberi", align="C",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_font("DejaVu", "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 7,
                   "AI olmadan projeye tam entelektüel ve teknik hâkimiyet için\n"
                   "kapsamlı, pedagojik ve izlenebilir el kitabı",
                   align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(20)
    pdf.image(str(FIG_DIR / "fig00_mental_model.png"), x=pdf.l_margin, w=pdf.epw)
    pdf.ln(6)
    pdf.set_font("Mono", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 6,
                   f"Repository commit : {pdf.commit}\n"
                   f"Üretim zamanı     : {pdf.gen_ts}\n"
                   f"repo              : satellite-thermal-digital-twin",
                   align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.on_cover = False


def render_toc(pdf, outline):
    start_page = pdf.page_no()
    pdf.set_font("DejaVu", "B", 18)
    pdf.set_text_color(*CLR_H1)
    pdf.cell(0, 12, "İçindekiler", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    for sect in outline:
        if sect.level > 1:
            continue
        lvl = sect.level
        pdf.set_x(pdf.l_margin + lvl * 6)
        if lvl == 0:
            pdf.set_font("DejaVu", "B", 10.5)
            pdf.set_text_color(*CLR_H2)
        else:
            pdf.set_font("DejaVu", "", 9.5)
            pdf.set_text_color(60, 60, 60)
        name = sect.name
        page = sect.page_number
        avail = pdf.w - pdf.r_margin - pdf.get_x() - 12
        # truncate name if needed
        while pdf.get_string_width(name) > avail - 8 and len(name) > 8:
            name = name[:-2]
        link = pdf.add_link(page=page)
        start_x = pdf.get_x()
        pdf.cell(pdf.get_string_width(name) + 1, 6, name, link=link)
        # dot leader
        dots_x = pdf.get_x()
        end_x = pdf.w - pdf.r_margin - 10
        pdf.set_text_color(180, 180, 180)
        ndots = max(0, int((end_x - dots_x) / pdf.get_string_width(".")))
        pdf.cell(end_x - dots_x, 6, "." * ndots)
        pdf.set_text_color(90, 90, 90)
        pdf.cell(10, 6, str(page), align="R", link=link,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(*CLR_TEXT)
    # Pad to fill exactly the reserved number of TOC pages (fpdf2 requires
    # the render function to span the reserved page count exactly).
    reserved = getattr(pdf, "_toc_reserved", 1)
    while pdf.page_no() < start_page + reserved - 1:
        pdf.add_page()


def main():
    if not MD_PATH.exists():
        print(f"HATA: Markdown kaynağı yok: {MD_PATH}", file=sys.stderr)
        return 2
    md = MD_PATH.read_text(encoding="utf-8")
    blocks = parse_blocks(md)
    pdf = Guide()

    render_cover(pdf)

    # TOC placeholder page(s): reserve enough pages for all level<=1 headings.
    n_toc_entries = sum(1 for k, _ in blocks if k in ("h1", "h2"))
    entries_per_page = 40
    toc_pages = max(1, (n_toc_entries + 8) // entries_per_page + 1)
    pdf._toc_reserved = toc_pages
    pdf.add_page()
    pdf.insert_toc_placeholder(render_toc, pages=toc_pages)

    render_blocks(pdf, blocks, None)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT_PDF))
    print(f"OK: {OUT_PDF}  ({OUT_PDF.stat().st_size/1024:.0f} KB, {pdf.page_no()} sayfa)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
