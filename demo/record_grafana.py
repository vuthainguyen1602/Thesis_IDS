#!/usr/bin/env python
"""Record the live Grafana edge dashboard as timestamped frames with headless Chrome.

Logs in with Grafana's basic auth (the local Docker stack's admin account, see
jetson/docker-compose.yml), opens the dashboard in kiosk mode on a rolling
5-minute window refreshing every 5 s, and screenshots it about --fps times a
second for --seconds. Frames go to <out-dir>/frames/NNNNN.png with their epoch
in <out-dir>/frames.txt (ffconcat), so make_video.sh can turn them into a video
whose clock matches the wall clock, and captions can be timed against events.

    python demo/record_grafana.py --seconds 200 --out-dir demo/out
"""
import argparse
import base64
import os
import time

from playwright.sync_api import sync_playwright

DASH = ("http://localhost:3000/d/ids-edge-dashboard/ids-edge-jetson-dashboard"
        "?orgId=1&kiosk&theme=light&from=now-5m&to=now&refresh=5s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--user", default=os.environ.get("GRAFANA_USER", "admin"))
    ap.add_argument("--password", default=os.environ.get("GRAFANA_PASSWORD", "admin"))
    args = ap.parse_args()

    frames = os.path.join(args.out_dir, "frames")
    os.makedirs(frames, exist_ok=True)
    token = base64.b64encode(f"{args.user}:{args.password}".encode()).decode()
    stamps = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        ctx = browser.new_context(viewport={"width": 1920, "height": 1080},
                                  extra_http_headers={"Authorization": f"Basic {token}"})
        page = ctx.new_page()
        page.goto(DASH, wait_until="networkidle")
        start = time.time()
        print(f"[rec] recording from {start:.0f} for {args.seconds:.0f}s", flush=True)
        i = 0
        while time.time() - start < args.seconds:
            t = time.time()
            page.screenshot(path=os.path.join(frames, f"{i:05d}.png"))
            stamps.append(t)
            i += 1
            time.sleep(max(0.0, 1.0 / args.fps - (time.time() - t)))
        browser.close()
    # ffconcat with per-frame durations taken from the real capture times
    with open(os.path.join(args.out_dir, "frames.txt"), "w") as f:
        f.write("ffconcat version 1.0\n")
        for k, t in enumerate(stamps):
            dur = (stamps[k + 1] - t) if k + 1 < len(stamps) else 1.0 / args.fps
            f.write(f"file 'frames/{k:05d}.png'\nduration {dur:.3f}\n")
        f.write(f"file 'frames/{len(stamps) - 1:05d}.png'\n")
    with open(os.path.join(args.out_dir, "start.txt"), "w") as f:
        f.write(f"{stamps[0]:.3f}\n")
    print(f"[rec] {len(stamps)} frames in {args.out_dir}")


if __name__ == "__main__":
    main()
