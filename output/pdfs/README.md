# PDF outputs

Sau khi chạy `./build_pdfs.sh` hoặc từng `compile.sh`, file PDF nằm tại đây:

| File | Nội dung |
|------|----------|
| `FAIR2026_draft.pdf` | Bài báo FAIR'2026 |
| `SOICT2026_draft.pdf` | Bài báo SOICT 2026 |
| `JOURNAL2026_draft.pdf` | Bài báo journal 2026 |
| `thesis_draft.pdf` | Luận văn (outline 6 chương) |

Build tất cả:

```bash
./build_pdfs.sh
open output/pdfs/
```

Nguồn LaTeX: `papers/*/manuscript/main.tex`, `thesis/main.tex`
