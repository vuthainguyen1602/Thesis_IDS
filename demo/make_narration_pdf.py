#!/usr/bin/env python
"""Turn demo/LoiThuyetMinh.md into an A4 PDF in output/pdfs/ (Markdown -> HTML -> headless Chrome).

    python demo/make_narration_pdf.py
"""
import os

import markdown
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "demo", "LoiThuyetMinh.md")
OUT = os.path.join(ROOT, "output", "pdfs", "LuanVan_demo_LoiThuyetMinh.pdf")

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: "Times New Roman", "Arial Unicode MS", serif; font-size: 12pt; line-height: 1.45; color: #111; }
h1 { font-size: 20pt; color: #1f3b63; margin: 0 0 4pt; }
h2 { font-size: 15pt; color: #1f3b63; border-bottom: 1.5pt solid #1f3b63; padding-bottom: 3pt; margin-top: 18pt;
     break-before: page; }
h1 + p + p + table + p + ul + p + hr + h2 { break-before: auto; }
h3 { font-size: 12.5pt; color: #fff; background: #1f3b63; padding: 3pt 7pt; border-radius: 3pt; margin: 14pt 0 6pt;
     break-after: avoid; }
blockquote { margin: 6pt 0 8pt; padding: 6pt 10pt; background: #eef3fa; border-left: 4pt solid #2a78d6;
             font-size: 12.5pt; break-inside: avoid; }
blockquote p { margin: 0; }
code { font-family: Menlo, monospace; font-size: 9.5pt; background: #f2f2f2; padding: 0 2pt; border-radius: 2pt; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 11pt; break-inside: avoid; }
th, td { border: 0.6pt solid #999; padding: 3pt 7pt; text-align: left; }
th { background: #e6ebf2; }
hr { border: none; border-top: 0.6pt solid #bbb; margin: 12pt 0; }
ul, ol { margin: 4pt 0 6pt; }
p { margin: 4pt 0 6pt; }
"""


def main():
    html_body = markdown.markdown(open(SRC, encoding="utf-8").read(), extensions=["tables"])
    html = f"<!doctype html><html lang='vi'><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page()
        page.set_content(html, wait_until="load")
        page.pdf(path=OUT, format="A4", print_background=True,
                 display_header_footer=True, header_template="<span></span>",
                 footer_template="<div style='font-size:8pt;width:100%;text-align:center;color:#666'>"
                                 "Lời thuyết minh video demo · <span class='pageNumber'></span>/<span class='totalPages'></span></div>",
                 margin={"top": "18mm", "bottom": "18mm", "left": "17mm", "right": "17mm"})
        browser.close()
    print(f"[pdf] {OUT}")


if __name__ == "__main__":
    main()
