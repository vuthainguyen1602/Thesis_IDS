# Manuscript — Luận văn

Luận văn dùng `\documentclass{report}` (phù hợp độ dài), **tài liệu tham khảo** theo Springer `splncs04.bst` từ `Latex-Template-for-Springer/`.

## Cấu trúc

```
manuscript/
├── main.tex
├── references.bib
├── compile.sh
└── chapters/
    ├── 01_introduction.tex
    ├── 02_related_work.tex
    ├── 03_methodology.tex
    ├── 04_ml_experiments.tex
    ├── 05_edge_deployment.tex
    └── 06_conclusion.tex
```

## Biên dịch

```bash
./thesis/collect_results.sh
cd thesis/manuscript
./compile.sh
```

## Liên kết 2 bài báo

| Chương luận văn | Paper |
|-----------------|-------|
| 3--4 | FAIR'2026 (`papers/fair2026/manuscript/`) |
| 5 | SOICT 2026 (`papers/soict2026/manuscript/`) |

Viết luận văn trước, rút gọn thành 2 paper — tránh copy nguyên văn (self-plagiarism).
