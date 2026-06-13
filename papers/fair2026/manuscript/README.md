# Manuscript — FAIR'2026 (Springer LNCS)

Dựa trên `Latex-Template-for-Springer/` (llncs.cls + splncs04.bst).

## Files

| File | Mô tả |
|------|--------|
| `main.tex` | Bài báo tiếng Việt (~8--10 trang) |
| `references.bib` | Tài liệu tham khảo BibTeX |
| `compile.sh` | Biên dịch PDF |

## Biên dịch

```bash
# Thu hình trước (từ root repo)
./papers/fair2026/collect_results.sh

# Biên dịch
cd papers/fair2026/manuscript
./compile.sh
# → main.pdf
```

**Yêu cầu:** TeX Live với `xelatex`, `bibtex`, gói `booktabs`, `polyglossia`, `fontspec`.

```bash
# macOS — cài đặt đầy đủ hơn texlive-basic nếu thiếu gói
# brew install --cask mactex-no-gui
```

## TODO trước khi nộp FAIR

- [x] Tên tác giả, trường, email
- [ ] Hoàn thiện mục Related Work
- [ ] Cập nhật Bảng~\ref{tab:best-results} từ `tables/cross_method_summary.csv`
- [ ] Kiểm tra format theo hướng dẫn [FAIR'2026](https://fair.conf.vn/)
