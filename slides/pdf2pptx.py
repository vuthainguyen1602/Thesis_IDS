#!/usr/bin/env python3
"""Convert a Beamer PDF deck into a .pptx (one full-bleed image per slide).

If a matching *_notes.pdf exists (beamer `show notes on second screen=right`),
the right half of each page is extracted as text and stored as speaker notes.

Usage: pdf2pptx.py slides/<deck>/main.pdf output/pdfs/<Name>.pptx [--dpi 200]
"""
import argparse, re, subprocess, sys, tempfile
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu


def page_info(pdf: Path):
    out = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True).stdout
    pages = w = h = None
    for line in out.splitlines():
        if line.startswith("Pages:"):
            pages = int(line.split()[1])
        elif line.startswith("Page size:"):
            parts = line.split()
            w, h = float(parts[2]), float(parts[4])
    return pages, w, h


NOTES_TOP_PT = 66  # pt from top: below the talk title, date, section line and first frame-title line
TITLE_PT = 56      # pt from top of the frame-title line (its second line can dip into the note body)


def notes_text(notes_pdf: Path, page: int, w: float, h: float) -> str:
    """Speaker notes from one beamer 'notes on second screen=right' page.

    The right half carries a header (talk title, date, section, frame title),
    a thumbnail of the slide, and then the note body.  Everything below
    NOTES_TOP_PT is body, except a wrapped second line of the frame title,
    which starts at the title indent rather than the body margin.
    """
    import html
    from collections import Counter
    xml = subprocess.run(
        ["pdftohtml", "-xml", "-i", "-f", str(page), "-l", str(page), "-stdout", str(notes_pdf)],
        check=True, capture_output=True, text=True).stdout
    pw = re.search(r'<page [^>]*width="([\d.]+)"', xml)
    if not pw:
        return ""
    scale = float(pw.group(1)) / w  # px per pt
    half, top_px, title_px = w / 2 * scale, NOTES_TOP_PT * scale, TITLE_PT * scale
    runs, title_left = [], None
    for m in re.finditer(r'<text top="(\d+)" left="(\d+)" width="(\d+)" height="(\d+)" font="\d+">(.*?)</text>', xml, re.S):
        top, left, width, height = (int(m[i]) for i in range(1, 5))
        body = m[5]
        text = html.unescape(re.sub(r"<[^>]+>", "", body)).strip()
        if not text or left < half:
            continue
        if title_px <= top < top_px and title_left is None:
            title_left = left  # first line of the frame title
        if top < top_px:
            continue
        runs.append((top, left, width, height, "<b>" in body, text))
    if not runs:
        return ""
    margin = Counter(r[1] for r in runs).most_common(1)[0][0]
    line_h = Counter(r[3] for r in runs).most_common(1)[0][0]
    if title_left is not None and abs(title_left - margin) > 4:
        runs = [r for r in runs if not (abs(r[1] - title_left) <= 2 and r[0] < top_px + 1.5 * line_h)]
    runs.sort(key=lambda r: (r[0], r[1]))

    # group runs into lines (tolerant of sub/superscripts), then lines into paragraphs
    lines, cur = [], []
    for r in runs:
        if cur and abs(r[0] - cur[0][0]) > 0.45 * line_h:
            lines.append(cur); cur = []
        cur.append(r)
    if cur:
        lines.append(cur)
    tops = [ln[0][0] for ln in lines]
    gaps = sorted(b - a for a, b in zip(tops, tops[1:]))
    pitch = gaps[len(gaps) // 2] if gaps else line_h  # median line pitch
    paras, buf, prev_top = [], [], None
    for ln in lines:
        ln.sort(key=lambda r: r[1])
        text, right = "", None
        for top, left, width, height, bold, t in ln:
            text += ("" if right is not None and left - right <= 2 else " ") + t
            right = left + width
        text = text.strip()
        new_para = prev_top is not None and (ln[0][0] - prev_top > 1.4 * pitch or (ln[0][4] and abs(ln[0][1] - margin) <= 2))
        if new_para and buf:
            paras.append(" ".join(buf)); buf = []
        buf.append(text); prev_top = ln[0][0]
    if buf:
        paras.append(" ".join(buf))
    return "\n\n".join(paras).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("pptx", type=Path)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-notes", action="store_true")
    ap.add_argument("--last", type=int, default=None,
                    help="export pages 1..LAST only (e.g. drop the beamer appendix)")
    a = ap.parse_args()

    pages, w, h = page_info(a.pdf)
    if a.last:
        pages = min(pages, a.last)
    notes_pdf = a.pdf.with_name(a.pdf.stem + "_notes.pdf")
    use_notes = notes_pdf.exists() and not a.no_notes
    if use_notes:
        npages, nw, nh = page_info(notes_pdf)
        if npages < pages:
            print(f"warning: {notes_pdf} has {npages} pages, deck has {pages}; skipping notes", file=sys.stderr)
            use_notes = False

    prs = Presentation()
    prs.slide_width = Emu(int(w * 12700))   # 1 pt = 12700 EMU
    prs.slide_height = Emu(int(h * 12700))
    blank = prs.slide_layouts[6]

    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-r", str(a.dpi), "-f", "1", "-l", str(pages), "-png", str(a.pdf), f"{td}/p"], check=True)
        pngs = sorted(Path(td).glob("p-*.png"))
        assert len(pngs) == pages, (len(pngs), pages)
        for i, png in enumerate(pngs, start=1):
            s = prs.slides.add_slide(blank)
            s.shapes.add_picture(str(png), 0, 0, width=prs.slide_width, height=prs.slide_height)
            if use_notes:
                t = notes_text(notes_pdf, i, nw, nh)
                if t:
                    s.notes_slide.notes_text_frame.text = t

    a.pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(a.pptx)
    print(f"{a.pptx}: {pages} slides, notes={'yes' if use_notes else 'no'}")


if __name__ == "__main__":
    main()
