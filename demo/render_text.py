#!/usr/bin/env python
"""Draw the demo's title card and caption overlays as 1920x1080 PNGs.

Captions are transparent full-frame PNGs whose text sits in the bar below the
scaled dashboard, so ffmpeg can overlay them at 0:0 without covering a panel. Text is Vietnamese, hence the Arial Unicode font.

    python demo/render_text.py <out-dir>
"""
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
BAR_TOP = 972  # dashboard is scaled to 1728x972 and padded; captions sit below it
FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"

TITLE = "Demo: Hệ thống phát hiện xâm nhập phân tán thời gian thực"
SUBTITLE = "Kafka + PySpark trên cụm Jetson Orin Nano Super  ·  Nguyễn Vũ Thái  ·  Luận văn Thạc sĩ"
CAPTIONS = [
    "Hai Jetson Orin Nano Super sẵn sàng: Jetson #1 chạy cổng lọc autoencoder, "
    "Jetson #2 chạy bộ phân loại PySpark",
    "Bắt đầu phát lại luồng mạng CICIDS2017 qua Kafka",
    "Cổng lọc bỏ qua phần lớn luồng lành tính; chỉ luồng nghi ngờ được chuyển sang bộ phân loại Spark",
    "Bảng điều khiển cập nhật trực tiếp: phán quyết, tấn công phát hiện, CPU và RAM từng nút",
    "Dừng phát lại: bộ phân loại xử lý nốt các luồng còn trong hàng đợi",
]


def wrap(draw, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def caption(text, path):
    """Caption for the bar below the dashboard (rows 972-1080 of the frame)."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 32)
    lines = wrap(d, text, font, W - 160)
    lh = 42
    top = BAR_TOP + (H - BAR_TOP - lh * len(lines)) / 2 - 4
    for i, line in enumerate(lines):
        tw = d.textlength(line, font=font)
        d.text(((W - tw) / 2, top + i * lh), line, font=font, fill=(255, 255, 255, 255))
    img.save(path)


def title(path):
    img = Image.new("RGB", (W, H), (31, 59, 99))
    d = ImageDraw.Draw(img)
    f1, f2 = ImageFont.truetype(FONT, 62), ImageFont.truetype(FONT, 34)
    for text, font, y, fill in ((TITLE, f1, H / 2 - 80, (255, 255, 255)),
                                (SUBTITLE, f2, H / 2 + 30, (220, 228, 240))):
        d.text(((W - d.textlength(text, font=font)) / 2, y), text, font=font, fill=fill)
    img.save(path)


def main():
    out = sys.argv[1]
    title(f"{out}/title.png")
    for i, text in enumerate(CAPTIONS):
        caption(text, f"{out}/cap{i}.png")


if __name__ == "__main__":
    main()
