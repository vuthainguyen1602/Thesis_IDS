# Slides

Three Beamer decks built from the same results as the manuscripts and the thesis.

| Deck | Language | Length | Output PDF |
|---|---|---|---|
| `soict2026/` | English | 7 min (8 slides + 3 section pages + 5 backup) | `output/pdfs/SOICT2026_slides.pdf` |
| `fair2026/` | English | 7 min (8 slides + 3 section pages + 5 backup) | `output/pdfs/FAIR2026_slides.pdf` |
| `defense/` | Vietnamese | 15 min (16 slides + 5 section pages + 12 dự phòng) | `output/pdfs/LuanVan_BaoVe_slides.pdf` |

## Building

```bash
./build_slides.sh                 # all three decks
./build_slides.sh defense         # one deck
NOTES=1 ./build_slides.sh         # also build the *_notes.pdf versions
```

Each deck also has a `compile.sh` wrapper. Requires **XeLaTeX** (fontspec, for
Vietnamese diacritics) plus the `beamer`, `beamertheme-metropolis` and
`appendixnumberbeamer` packages.

`xelatex` under `-interaction=nonstopmode` exits 0 even when it bailed out on a
real error, so the script scans each log for `! ` and checks the page count
before publishing — a collapsed build fails loudly instead of shipping a
one-page PDF.

`common/preamble.tex` holds the shared theme: 16:9, metropolis, Okabe-Ito
accents matching the manuscript TikZ colours, and a Vietnamese-capable sans
font fallback chain (TeX Gyre Heros → Helvetica Neue → Arial → TeX Gyre Termes).

## Speaker notes

Every content frame carries a `\note{}` with what to say and what to emphasise.
`NOTES=1` builds a two-screen PDF (slide left, notes right) for presenter mode
in a PDF viewer that supports it. The plain PDF is unaffected.

## Layout guards

Two failure modes bite Beamer silently — neither raises an error, both only
show up in the rendered PDF:

- **A table wider than its column is not clipped**, it overprints the
  neighbouring column. Every `tabular` is therefore wrapped in `\fitwidth{...}`
  (see `common/preamble.tex`), which scales a box down only when it would
  otherwise overflow. Keep new tables wrapped the same way.
- **Content that overruns the text area runs into the footer**, so the frame
  number gets printed on top of the last line. After editing, check the log:
  `grep "Overfull \\\\vbox" <deck>/main.log` should show only the `\maketitle`
  entry (~15.6 pt, metropolis' own title block, harmless). Anything above
  ~11 pt on a content frame needs a line or two cut.

- **An image constrained only by `height` can still be wider than the text
  block** and stick out sideways. Every `\includegraphics` here passes both
  `height` and `width=\textwidth` with `keepaspectratio`.

Also note that a `tabular` leaves TeX in horizontal mode: text placed after it
without an explicit `\par` is typeset *beside* the table, not below it. And
Vietnamese offers few hyphenation points, so the preamble sets
`\emergencystretch` — without it a long run of words fails to break and
overflows the line.

## Slide numbering and backup navigation

Backup slides sit after `\appendix` and section pages are `noframenumbering`,
so the footer fraction counts only the talk itself (e.g. `5/8`) — neither
inflates the denominator. Section titles are Roman-numbered and upper-case in
all three decks.

The defense deck opens its backup section with a **clickable index** keyed by
the question a committee member is likely to ask, so the right slide is one
click away mid-answer. Each backup frame carries a `label=bk-*` and the index
reaches it with `\hyperlink`; if you add a backup slide, give it a label and a
row in that index.

## Timing

Each deck's header comment carries a per-slide timing plan. The paper decks are
built to the 7-minute slot requested; note the conference norms if the slot
changes:

- **SOICT 2025** ran oral presentations in 20-minute programme blocks
  (≈15 min talk + Q&A). To stretch a 7-minute deck, promote the backup slides
  on the gate operating curve, the model-selection bridge table and the
  measurement protocol into the main flow.
- **FAIR 2025** allowed a maximum of 15 minutes per paper plus 3–5 minutes of
  discussant comments, with the first author presenting.

## Figures

The decks pull PNGs straight from `results/` and `papers/soict2026/`, so they
stay in sync when the pipeline is re-run. Figures that may not exist yet are
wrapped in `\IfFileExists` and degrade to a labelled placeholder box instead of
failing the build.
