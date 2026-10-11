#!/usr/bin/env bash
# Build the demo MP4 from a record_demo.sh output directory: a title card, the
# dashboard frames at their real timing, and Vietnamese captions placed by the
# logged events (events.txt). Captions and the title are drawn with Pillow
# (this ffmpeg build has no drawtext) and overlaid by ffmpeg. Writes
# <dir>/demo.mp4 and copies it to output/videos/LuanVan_demo.mp4.
#
#   ./demo/make_video.sh demo/out/<tag>
set -euo pipefail

DIR=${1:?record_demo output dir}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TITLE_SECONDS=${TITLE_SECONDS:-5}
cd "$DIR"

ev() { awk -v k="$1" '$2==k{print $1}' events.txt; }
rec=$(cat start.txt)
t_send=$(echo "$(ev send_start) - $rec" | bc)
t_end=$(echo "$(ev send_end) - $rec" | bc)
t_rec_end=$(echo "$(ev record_end) - $rec" | bc)
a=$(echo "$t_send + 30" | bc); b=$(echo "$t_send + 90" | bc)

python "$ROOT/demo/render_text.py" .

ffmpeg -loglevel error -y -f concat -safe 0 -i frames.txt \
  -i cap0.png -i cap1.png -i cap2.png -i cap3.png -i cap4.png -filter_complex "\
[0:v]fps=25,scale=1728:972,format=rgba,pad=1920:1080:96:0:color=0x1f3b63[v0];\
[v0][1:v]overlay=0:0:enable='between(t,0,$t_send)'[v1];\
[v1][2:v]overlay=0:0:enable='between(t,$t_send,$a)'[v2];\
[v2][3:v]overlay=0:0:enable='between(t,$a,$b)'[v3];\
[v3][4:v]overlay=0:0:enable='between(t,$b,$t_end)'[v4];\
[v4][5:v]overlay=0:0:enable='between(t,$t_end,$t_rec_end)',format=yuv420p[out]" \
  -map "[out]" -c:v libx264 -crf 20 -preset medium body.mp4
ffmpeg -loglevel error -y -loop 1 -framerate 25 -t "$TITLE_SECONDS" -i title.png \
  -vf format=yuv420p -c:v libx264 -crf 20 title.mp4
printf "file 'title.mp4'\nfile 'body.mp4'\n" > parts.txt
ffmpeg -loglevel error -y -f concat -safe 0 -i parts.txt -c copy demo.mp4
mkdir -p "$ROOT/output/videos"
cp demo.mp4 "$ROOT/output/videos/LuanVan_demo.mp4"
echo "[video] $DIR/demo.mp4 ($(ffprobe -v error -show_entries format=duration -of csv=p=0 demo.mp4 | cut -d. -f1) s)"
