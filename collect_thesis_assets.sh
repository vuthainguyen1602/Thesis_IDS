#!/bin/bash
# collect_thesis_assets.sh — run AFTER run_all.sh finishes.
#  (1) Copy auto-generated result figures from results/ -> thesis/img/ (renaming
#      to the names the thesis expects).
#  (2) Print every result CSV so you can read off the numbers for the \ph{} slots.
#
# Usage: ./collect_thesis_assets.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
RES="$ROOT/results"
IMG="$ROOT/thesis/img"
mkdir -p "$IMG"

echo "================================================================"
echo " (1) Copy result figures -> thesis/img/"
echo "================================================================"
# "dest_in_img : candidate source basenames" (searched recursively under results/)
MAP="
confusion_matrix.png:confusion_matrix.png confusion_matrices.png
shap_summary_beeswarm.png:shap_summary_beeswarm.png
shap_feature_importance_top30.png:shap_feature_importance_bar.png shap_feature_importance_top30.png
benchmark_comparison.png:benchmark_comparison.png cross_method_f1_comparison.png comparison.png
model_tradeoff_radar.png:model_tradeoff_radar.png model_tradeoff.png
"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  dest="${line%%:*}"; cands="${line#*:}"
  found=""
  for c in $cands; do
    f="$(find "$RES" -type f -name "$c" 2>/dev/null | head -1)"
    [ -n "$f" ] && { found="$f"; break; }
  done
  if [ -n "$found" ]; then
    cp "$found" "$IMG/$dest"
    echo "  OK    $dest   <-  ${found#$ROOT/}"
  else
    echo "  MISS  $dest   (chua thay trong results/ — kiem tra da chay ml_05/06/07/09 + benchmark chua)"
  fi
done <<< "$MAP"

echo
echo "  [Tu chup tay - khong script nao tao duoc] 5 anh he thong bien -> $IMG/ :"
echo "    grafana_dashboard.png  grafana_attacks.png  mailtrap_alert.png  slack_alert.png  pipeline_terminal.png"

echo
echo "================================================================"
echo " (2) Result CSVs — doc so de dien cac \\ph{} (Chuong 4)"
echo "================================================================"
found_csv=0
for csv in $(find "$RES" -type f -name "*.csv" 2>/dev/null | sort); do
  found_csv=1
  echo
  echo "----- ${csv#$ROOT/} -----"
  column -s, -t "$csv" 2>/dev/null | head -20 || head -20 "$csv"
done
[ "$found_csv" = 0 ] && echo "  (chua co CSV nao trong results/ — chay run_all.sh xong roi chay lai script nay)"

echo
echo "Xong. Kiem tra thesis/img/ va doi chieu so tu cac CSV o tren vao bang Chuong 4."
