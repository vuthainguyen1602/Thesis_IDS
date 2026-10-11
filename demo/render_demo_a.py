#!/usr/bin/env python
"""Render the phase-A (distributed training) demo from a run_session_a.py directory.

Same approach as render_demo.py: every command and output line is replayed
from the recorded transcript (long waits shortened), and the Spark UIs are
shown from the frames recorded during the run; the job UI is sped up and the
speed-up is stated on screen.

    python demo/render_demo_a.py demo/out/phaseA_<tag> [--out video.mp4]
"""
import argparse
import bisect
import json
import os

from PIL import Image, ImageDraw

import render_demo as rd

rd.PANES.update({
    "host": "Máy chủ (MacBook) · Spark Master",
    "j1": "Jetson #1 (192.168.1.81) · Spark Worker",
    "j2": "Jetson #2 (192.168.1.39) · Spark Worker + driver",
})
TITLES = {
    "1": "Bước 1 · Máy chủ: khởi động Spark Master (Spark Standalone)",
    "2": "Bước 2 · Hai Jetson: khởi động Spark Worker và đăng ký với Master",
    "3": "Bước 3 · Kiểm tra cụm: hai worker Jetson đang hoạt động",
    "master_ui": "Bước 3 · Spark Master UI: cụm gồm hai worker Jetson Orin Nano Super",
    "4": "Bước 4 · Gửi job huấn luyện Random Forest (SHAP Top-30) lên cụm",
    "app_ui": "Bước 4 · Spark UI trên driver: các job huấn luyện chạy trên executor của hai Jetson",
    "5": "Kết quả · Đánh giá mô hình trên toàn bộ tập kiểm tra CICIDS2017",
}


def title_card(r):
    img = Image.new("RGB", (rd.W, rd.H), rd.HEADER_BG)
    d = ImageDraw.Draw(img)
    f1, f2, f3 = rd.font(rd.UNI, 60), rd.font(rd.UNI, 34), rd.font(rd.UNI, 28)
    lines = [("Demo pha A: Huấn luyện mô hình phân tán trên cụm Spark", f1, rd.H / 2 - 120, (255, 255, 255)),
             ("Spark Standalone: Master trên máy chủ, Worker trên hai Jetson Orin Nano Super", f2,
              rd.H / 2 - 20, (220, 228, 240)),
             (f"{rd.AUTHOR}  ·  Luận văn Thạc sĩ", f2, rd.H / 2 + 40, (220, 228, 240)),
             ("Toàn bộ lệnh và kết quả trong video được ghi từ một lần chạy thật", f3, rd.H / 2 + 140,
              (180, 195, 215))]
    for text, f, y, col in lines:
        d.text(((rd.W - d.textlength(text, font=f)) / 2, y), text, font=f, fill=col)
    r.emit(img, 6)


def ui_scene(r, frames_dir, title, caption, speed=1.0, max_seconds=None):
    start = float(open(os.path.join(frames_dir, "start.txt")).read())
    stamps, files, acc = [], [], start
    for line in open(os.path.join(frames_dir, "frames.txt")):
        line = line.strip()
        if line.startswith("file "):
            files.append(line.split("'")[1])
        elif line.startswith("duration "):
            stamps.append(acc)
            acc += float(line.split()[1])
    files = files[:len(stamps)]
    total = (stamps[-1] - stamps[0]) / speed
    if max_seconds:
        total = min(total, max_seconds)
    cache, t = {}, 0.0
    while t <= total:
        i = max(0, bisect.bisect_right(stamps, stamps[0] + t * speed) - 1)
        if i not in cache:
            cache.clear()
            cache[i] = Image.open(os.path.join(frames_dir, files[i])).convert("RGB").resize((1575, 886))
        img = Image.new("RGB", (rd.W, rd.H), rd.HEADER_BG)
        d = ImageDraw.Draw(img)
        rd.header(d, title)
        img.paste(cache[i], ((rd.W - 1575) // 2, 96))
        tw = d.textlength(caption, font=rd.F_CAP)
        d.text(((rd.W - tw) / 2, 1012), caption, font=rd.F_CAP, fill=(255, 255, 255))
        r.emit(img)
        t += 1.0 / rd.FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--app-ui-seconds", type=float, default=30)
    args = ap.parse_args()
    tr = json.load(open(os.path.join(args.dir, "transcript.json")))
    ev = tr["events"]
    out = args.out or os.path.join(args.dir, "demo_phaseA.mp4")

    def between(a, b):
        ia = next(i for i, e in enumerate(ev) if e["kind"] == "step" and e["text"] == a)
        ib = next((i for i, e in enumerate(ev) if e["kind"] == "step" and e["text"] == b), len(ev))
        return [e for e in ev[ia + 1:ib] if e["kind"] in ("cmd", "out")]

    panes = {"host": rd.Pane("host", "thainguyenvu@MacBook ~/Thesis_IDS % "),
             "j1": rd.Pane("j1", "thainguyenvu@MacBook ~ % "),
             "j2": rd.Pane("j2", "thainguyenvu@MacBook ~ % ")}
    full = {"host": (40, 110, rd.W - 40, rd.H - 30)}
    two = {"j1": (40, 110, rd.W / 2 - 10, rd.H - 30), "j2": (rd.W / 2 + 10, 110, rd.W - 40, rd.H - 30)}
    r = rd.Renderer(out)
    title_card(r)

    s, end = rd.schedule(between("1", "2"))
    rd.terminal_scene(r, TITLES["1"], panes, full, s, end, 20, hold=2.5)
    s, end = rd.schedule(between("2", "3"), cap=0.8)
    rd.terminal_scene(r, TITLES["2"], panes, two, s, end, 18, hold=3)
    s, end = rd.schedule(between("3", "4"))
    rd.terminal_scene(r, TITLES["3"], panes, full, s, end, 20, hold=2.5)
    ui_scene(r, os.path.join(args.dir, "ui_master"), TITLES["master_ui"],
             "Spark Master UI (máy chủ, cổng 8080): hai worker Jetson ở trạng thái ALIVE, sẵn sàng nhận job",
             max_seconds=8)

    train = between("4", "__end__")
    cut = next((i for i, e in enumerate(train) if "TRAINING:" in e["text"]), len(train) - 1)
    first, rest = train[:cut + 3], train[cut + 3:]
    s, end = rd.schedule(first, cap=1.0)
    rd.terminal_scene(r, TITLES["4"], panes, full, s, end, 18, hold=2)
    app_dir = os.path.join(args.dir, "ui_app")
    if os.path.exists(os.path.join(app_dir, "frames.txt")):
        stamps = [float(x) for x in [open(os.path.join(app_dir, "start.txt")).read()]]
        dur = 0.0
        for line in open(os.path.join(app_dir, "frames.txt")):
            if line.startswith("duration "):
                dur += float(line.split()[1])
        speed = max(1.0, dur / args.app_ui_seconds)
        ui_scene(r, app_dir, TITLES["app_ui"],
                 f"Spark UI của job (driver trên Jetson #2, cổng 4040) · tua nhanh {speed:.0f}×",
                 speed=speed)
    s, end = rd.schedule(rest, cap=1.0)
    rd.terminal_scene(r, TITLES["5"], panes, full, s, end, 18, hold=8)
    r.close()
    print(f"[render] {out} ({r.frames / rd.FPS:.0f} s)")


if __name__ == "__main__":
    main()
