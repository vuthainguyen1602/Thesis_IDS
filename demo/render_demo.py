#!/usr/bin/env python
"""Render the demo video from a run_session.py directory.

Replays the recorded transcript as terminal scenes (every command and output
line is the real one, with long waits shortened), then shows the Grafana
dashboard recorded during the replay, then the shutdown statistics. Frames are
drawn with Pillow and piped to ffmpeg.

    python demo/render_demo.py demo/out/session_<tag> [--out video.mp4]
"""
import argparse
import bisect
import json
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 25
UNI = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
MONO = "/System/Library/Fonts/Menlo.ttc"
BG, HEADER_BG = (24, 26, 31), (31, 59, 99)
PANE_BG, PANE_BAR = (30, 30, 30), (55, 55, 58)
FG, DIM = (220, 220, 220), (150, 150, 150)
GREEN, YELLOW, RED, CYAN = (120, 210, 120), (230, 200, 90), (240, 110, 110), (110, 190, 230)
AUTHOR = "Bùi Văn Dũng"

TITLES = {
    "1": "Bước 1 · Máy chủ: hạ tầng Kafka, PostgreSQL, InfluxDB, Grafana đang chạy",
    "2": "Bước 2 · Kiểm tra hai Jetson Orin Nano Super và mô hình đã triển khai",
    "3": "Bước 3 · Jetson #1: khởi động cổng lọc bất thường (autoencoder)",
    "4": "Bước 4 · Jetson #2: khởi động bộ phân loại PySpark (Random Forest)",
    "5": "Bước 5 · Phát lại luồng CICIDS2017 qua Kafka — log trực tiếp trên hai Jetson",
    "grafana": "Kết quả · Bảng điều khiển Grafana theo dõi hệ thống theo thời gian thực",
    "6": "Bước 6 · Dừng hệ thống (Ctrl+C): thống kê cuối của từng nút",
}
GRAFANA_CAPTION = ("Thông lượng phán quyết, số tấn công phát hiện, độ trễ suy luận, CPU/RAM từng Jetson "
                   "và cảnh báo mới nhất được cập nhật trực tiếp")
PANES = {"host": "Máy chủ (MacBook) · Kafka, PostgreSQL, InfluxDB, Grafana",
         "j1": "Jetson #1 (192.168.1.81) · cổng lọc bất thường",
         "j2": "Jetson #2 (192.168.1.39) · bộ phân loại PySpark"}


def font(path, size):
    return ImageFont.truetype(path, size)


F_HEAD, F_CAP, F_BAR = font(UNI, 34), font(UNI, 30), font(UNI, 22)


class Pane:
    """Terminal state of one pane: finished lines plus a command being typed."""

    def __init__(self, key, prompt):
        self.key, self.prompt, self.lines = key, prompt, []

    def add(self, ev):
        if ev["kind"] == "cmd":
            self.lines.append(("cmd", self.prompt, ev["text"]))
            if ev["text"].startswith("ssh ") and " " not in ev["text"][4:].strip():
                self.prompt = "bvdung@ubuntu:~$ "
            if ev["text"].startswith("cd ~/Thesis_IDS/jetson"):
                self.prompt = "(venv) bvdung@ubuntu:~/Thesis_IDS/jetson$ "
        elif ev["kind"] == "out":
            self.lines.append(("out", "", ev["text"]))


def color_for(text):
    if "[ERR" in text or "Traceback" in text:
        return RED
    if "[WARN" in text or " WARN " in text:
        return YELLOW
    if "[OK]" in text or "READY" in text or "COMPLETE" in text:
        return GREEN
    if text.startswith("  [jetson") or "Sent " in text or "Total" in text or "Attack" in text:
        return CYAN
    return FG


def draw_pane(img, d, box, pane, typing=None, size=20):
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=PANE_BG)
    d.rectangle([x0, y0, x1, y0 + 34], fill=PANE_BAR)
    for i, c in enumerate(((237, 106, 94), (245, 191, 79), (98, 197, 84))):
        d.ellipse([x0 + 14 + i * 22, y0 + 11, x0 + 26 + i * 22, y0 + 23], fill=c)
    d.text((x0 + 90, y0 + 5), PANES[pane.key], font=F_BAR, fill=(230, 230, 230))
    f = font(MONO, size)
    cw = d.textlength("M", font=f)
    lh = int(size * 1.4)
    cols = max(20, int((x1 - x0 - 24) / cw))
    rows = int((y1 - y0 - 50) / lh)
    rendered = []  # (segments) per visual row; segment = (text, color)
    entries = list(pane.lines)
    if typing is not None:
        entries.append(("cmd", typing[0], typing[1] + "▌"))
    for kind, prompt, text in entries:
        if kind == "cmd":
            full = prompt + text
            for k in range(0, max(1, len(full)), cols):
                chunk = full[k:k + cols]
                if k == 0:
                    rendered.append([(chunk[:len(prompt)], GREEN), (chunk[len(prompt):], (255, 255, 255))])
                else:
                    rendered.append([(chunk, (255, 255, 255))])
        else:
            col = color_for(text)
            for k in range(0, max(1, len(text)), cols):
                rendered.append([(text[k:k + cols], col)])
    for r, segs in enumerate(rendered[-rows:]):
        x, y = x0 + 12, y0 + 44 + r * lh
        for text, col in segs:
            d.text((x, y), text, font=f, fill=col)
            x += d.textlength(text, font=f)


def header(d, text):
    d.rectangle([0, 0, W, 86], fill=HEADER_BG)
    d.text((40, 22), text, font=F_HEAD, fill=(255, 255, 255))


def schedule(events, cap=1.0, cps=45.0, realtime=False, lead=0.0):
    """Map transcript events to display times; returns [(t_show, t_type_start, ev)]."""
    out, cur, last = [], lead, None
    for ev in events:
        if realtime:
            t = ev["t"] - events[0]["t"] + lead
            typ = min(len(ev["text"]) / cps, 2.0) if ev["kind"] == "cmd" else 0.0
            out.append((t, t - typ, ev))
            continue
        if last is not None:
            cur += min(max(ev["t"] - last, 0.0), cap)
        if ev["kind"] == "cmd":
            typ = min(len(ev["text"]) / cps, 2.5)
            out.append((cur + typ, cur, ev))
            cur += typ + 0.35
        else:
            out.append((cur, cur, ev))
        last = ev["t"]
    return out, (out[-1][0] if out else lead)


class Renderer:
    def __init__(self, path):
        self.ff = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20",
             "-preset", "medium", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
        self.frames = 0

    def emit(self, img, seconds=None):
        data = img.convert("RGB").tobytes()
        n = 1 if seconds is None else max(1, int(round(seconds * FPS)))
        for _ in range(n):
            self.ff.stdin.write(data)
        self.frames += n

    def close(self):
        self.ff.stdin.close()
        self.ff.wait()


def title_card(r):
    img = Image.new("RGB", (W, H), HEADER_BG)
    d = ImageDraw.Draw(img)
    f1, f2, f3 = font(UNI, 62), font(UNI, 34), font(UNI, 28)
    lines = [("Demo: Hệ thống phát hiện xâm nhập phân tán thời gian thực", f1, H / 2 - 120, (255, 255, 255)),
             ("Kafka + PySpark trên cụm hai Jetson Orin Nano Super", f2, H / 2 - 20, (220, 228, 240)),
             (f"{AUTHOR}  ·  Luận văn Thạc sĩ", f2, H / 2 + 40, (220, 228, 240)),
             ("Toàn bộ lệnh và kết quả trong video được ghi từ một lần chạy thật", f3, H / 2 + 140, (180, 195, 215))]
    for text, f, y, col in lines:
        d.text(((W - d.textlength(text, font=f)) / 2, y), text, font=f, fill=col)
    r.emit(img, 6)


def terminal_scene(r, title, panes, layout, sched, end, size, hold=1.5):
    """Draw panes while scheduled events appear; panes keep their earlier lines."""
    idx = 0
    t = 0.0
    while t <= end + hold:
        while idx < len(sched) and sched[idx][0] <= t:
            ev = sched[idx][2]
            if ev["pane"] in panes:
                panes[ev["pane"]].add(ev)
            idx += 1
        typing = {}
        for show, start, ev in sched[idx:]:
            if ev["kind"] == "cmd" and start <= t < show and ev["pane"] in panes:
                n = int(len(ev["text"]) * (t - start) / max(show - start, 1e-6))
                typing[ev["pane"]] = (panes[ev["pane"]].prompt, ev["text"][:n])
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        header(d, title)
        for key, box in layout.items():
            draw_pane(img, d, box, panes[key], typing.get(key), size)
        r.emit(img)
        t += 1.0 / FPS


def grafana_scene(r, out_dir, t0, t_from, t_to):
    start = float(open(os.path.join(out_dir, "start.txt")).read())
    stamps, files, acc = [], [], start
    for line in open(os.path.join(out_dir, "frames.txt")):
        line = line.strip()
        if line.startswith("file "):
            files.append(line.split("'")[1])
        elif line.startswith("duration "):
            stamps.append(acc)
            acc += float(line.split()[1])
    files = files[:len(stamps)]
    cache = {}
    t = t_from
    while t <= t_to:
        epoch = t0 + t
        i = max(0, bisect.bisect_right(stamps, epoch) - 1)
        if i not in cache:
            cache.clear()
            # keep the dashboard's 16:9 aspect inside the 886 px between header and caption
            g = Image.open(os.path.join(out_dir, files[i])).convert("RGB").resize((1575, 886))
            cache[i] = g
        img = Image.new("RGB", (W, H), HEADER_BG)
        d = ImageDraw.Draw(img)
        header(d, TITLES["grafana"])
        img.paste(cache[i], ((W - 1575) // 2, 96))
        tw = d.textlength(GRAFANA_CAPTION, font=F_CAP)
        d.text(((W - tw) / 2, 1012), GRAFANA_CAPTION, font=F_CAP, fill=(255, 255, 255))
        r.emit(img)
        t += 1.0 / FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--live-seconds", type=float, default=45)
    args = ap.parse_args()
    tr = json.load(open(os.path.join(args.dir, "transcript.json")))
    ev = tr["events"]
    out = args.out or os.path.join(args.dir, "demo.mp4")

    def steps_between(a, b):
        ia = next(i for i, e in enumerate(ev) if e["kind"] == "step" and e["text"] == a)
        ib = next((i for i, e in enumerate(ev) if e["kind"] == "step" and e["text"] == b), len(ev))
        return [e for e in ev[ia + 1:ib] if e["kind"] in ("cmd", "out")]

    def until_ready(events):
        """Cut a start-up scene after the node's READY banner (the session kept
        waiting there, printing only periodic [MONITOR] lines)."""
        for i, e in enumerate(events):
            if "READY" in e["text"]:
                return events[:i + 2]
        return events

    def until_shutdown(events):
        """Stop each pane at 'Pipeline shutdown complete.'; PySpark's own SIGINT
        handler then prints a KeyboardInterrupt traceback that is not ours."""
        done, keep = set(), []
        for e in events:
            if e["pane"] in done:
                continue
            keep.append(e)
            if "Pipeline shutdown complete" in e["text"]:
                done.add(e["pane"])
        return keep

    mark = {e["text"]: e["t"] for e in ev if e["kind"] == "mark"}
    panes = {"host": Pane("host", "thainguyenvu@MacBook ~/Thesis_IDS/jetson % "),
             "j1": Pane("j1", "thainguyenvu@MacBook ~ % "),
             "j2": Pane("j2", "thainguyenvu@MacBook ~ % ")}
    full = {"host": (40, 110, W - 40, H - 30)}
    r = Renderer(out)
    title_card(r)
    for step in ("1", "2"):
        s, end = schedule(steps_between(step, str(int(step) + 1)))
        terminal_scene(r, TITLES[step], panes, full, s, end, 21, hold=2.5)
    s, end = schedule(until_ready(steps_between("3", "4")), cap=1.0)
    terminal_scene(r, TITLES["3"], panes, {"j1": (40, 110, W - 40, H - 30)}, s, end, 21, hold=2.5)
    s, end = schedule(until_ready(steps_between("4", "5")), cap=1.0)
    terminal_scene(r, TITLES["4"], panes, {"j2": (40, 110, W - 40, H - 30)}, s, end, 21, hold=2.5)

    three = {"host": (40, 110, W - 40, 430), "j1": (40, 450, W / 2 - 10, H - 30),
             "j2": (W / 2 + 10, 450, W - 40, H - 30)}
    live = [e for e in steps_between("5", "6") if e["t"] <= mark["send_start"] + args.live_seconds]
    s, end = schedule(live, realtime=True, lead=2.5)
    terminal_scene(r, TITLES["5"], panes, three, s, end, 17, hold=0.5)
    rest = [e for e in steps_between("5", "6") if e["t"] > mark["send_start"] + args.live_seconds]
    for e in rest:  # keep the panes current for the shutdown scene
        if e["kind"] in ("cmd", "out"):
            panes[e["pane"]].add(e)
    grafana_scene(r, args.dir, tr["t0"], mark["send_start"] + args.live_seconds, mark["record_end"])
    s, end = schedule(until_shutdown(steps_between("6", "__end__")), cap=0.8)
    terminal_scene(r, TITLES["6"], panes, three, s, end, 17, hold=5)
    r.close()
    print(f"[render] {out} ({r.frames / FPS:.0f} s)")


if __name__ == "__main__":
    main()
