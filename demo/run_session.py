#!/usr/bin/env python
"""Run the demo session for real and record a transcript of every command.

Each step runs an actual command, on the Mac or on a Jetson over SSH, and every
output line is stored with the time it arrived, so render_demo.py can replay
the session as terminal scenes without inventing any output. The Grafana
dashboard is recorded in parallel while flows are replayed.

Panes: "host" (the Mac that runs Kafka/PostgreSQL/InfluxDB/Grafana and the
sender), "j1" (Jetson #1, anomaly gate), "j2" (Jetson #2, PySpark classifier).

    python demo/run_session.py --out demo/out/<tag>
"""
import argparse
import json
import os
import re
import signal
import subprocess
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JETSON = os.path.join(ROOT, "jetson")
U = "bvdung"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
NOISE = re.compile(r"\[Stage \d+|^\s*$")


class Transcript:
    def __init__(self):
        self.t0 = time.time()
        self.events = []
        self.lock = threading.Lock()

    def add(self, pane, kind, text):
        with self.lock:
            self.events.append({"t": round(time.time() - self.t0, 3), "pane": pane,
                                "kind": kind, "text": text})

    def save(self, path, meta):
        with open(path, "w") as f:
            json.dump({"t0": self.t0, "meta": meta, "events": self.events}, f,
                      ensure_ascii=False, indent=1)


def pump(proc, tr, pane, ready=None, ready_event=None):
    for raw in proc.stdout:
        for line in raw.replace("\r", "\n").split("\n"):
            line = line.rstrip()
            if NOISE.search(line):
                continue
            tr.add(pane, "out", line)
            if ready and ready in line and ready_event:
                ready_event.set()


def run_host(tr, display, argv, cwd=JETSON, env=None):
    tr.add("host", "cmd", display)
    p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    pump(p, tr, "host")
    p.wait()


def start_node(tr, pane, host, exports, launch, ready):
    prompt_cmds = [f"ssh {U}@{host}",
                   "cd ~/Thesis_IDS/jetson && source venv/bin/activate",
                   f"export {exports}",
                   launch]
    for c in prompt_cmds:
        tr.add(pane, "cmd", c)
        time.sleep(0.4)
    remote = (f"cd ~/Thesis_IDS/jetson && source venv/bin/activate && export {exports} && "
              f"export PYTHONUNBUFFERED=1 && {launch} 2>&1")
    p = subprocess.Popen(SSH + [f"{U}@{host}", remote], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    ev = threading.Event()
    th = threading.Thread(target=pump, args=(p, tr, pane, ready, ev), daemon=True)
    th.start()
    return p, th, ev


def mark_time(tr, name):
    return next(e["t"] for e in tr.events if e["kind"] == "mark" and e["text"] == name)


def verify_step(tr, t_from, t_to):
    """Step 7: count the stored verdicts, then check them against the replay labels."""
    tr.add("host", "step", "7")
    sql = ("SELECT raw_features->>'route' AS route, prediction, count(*) FROM predictions "
           f"WHERE (raw_features->>'sent_ts')::float BETWEEN {t_from:.0f} AND {t_to:.0f} "
           "GROUP BY 1, 2 ORDER BY 1, 2;")
    run_host(tr, f'docker exec ids-postgres psql -U ids -d ids_edge -c "{sql}"',
             ["docker", "exec", "ids-postgres", "psql", "-U", "ids", "-d", "ids_edge", "-c", sql])
    run_host(tr, f"python ../demo/verify_run.py --from {t_from:.0f} --to {t_to:.0f}",
             ["python", os.path.join(ROOT, "demo", "verify_run.py"),
              "--from", f"{t_from:.0f}", "--to", f"{t_to:.0f}"])


def append_verify(session_dir):
    """Add step 7 to an existing session's transcript (run after the fact)."""
    path = os.path.join(session_dir, "transcript.json")
    data = json.load(open(path))
    tr = Transcript()
    tr.t0, tr.events = data["t0"], data["events"]
    tr.events = [e for e in tr.events if not (e["kind"] == "step" and e["text"] == "7")]
    last = tr.events[-1]["t"]
    t_from = tr.t0 + mark_time(tr, "send_start") - 1
    t_to = tr.t0 + mark_time(tr, "send_end") + 1
    start = len(tr.events)
    verify_step(tr, t_from, t_to)
    # the step ran later; keep its own pacing but place it right after the session
    shift = tr.events[start]["t"] - (last + 1.0)
    for e in tr.events[start:]:
        e["t"] = round(e["t"] - shift, 3)
    data["events"] = tr.events
    data["meta"]["verify_appended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(data, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"[session] step 7 appended to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--j1", default="192.168.1.81")
    ap.add_argument("--j2", default="192.168.1.39")
    ap.add_argument("--rate", type=int, default=15)
    ap.add_argument("--send-seconds", type=float, default=150)
    ap.add_argument("--tail-seconds", type=float, default=40)
    ap.add_argument("--append-verify", action="store_true",
                    help="only add step 7 (verification) to the session in --out")
    args = ap.parse_args()
    if args.append_verify:
        return append_verify(args.out)
    os.makedirs(args.out, exist_ok=True)

    mac_ip = subprocess.run(["ipconfig", "getifaddr", "en0"], capture_output=True,
                            text=True).stdout.strip()
    tag = time.strftime("%Y%m%d%H%M%S")
    for h in (args.j1, args.j2):  # stop leftovers; [k] keeps pkill off its own shell
        subprocess.run(SSH + [f"{U}@{h}", "pkill -f '[k]afka_consumer.py' || true"])
    tr = Transcript()

    # Step 1: infrastructure on the Mac
    tr.add("host", "step", "1")
    run_host(tr, 'docker compose ps --format "table {{.Name}}\\t{{.Image}}\\t{{.Status}}"',
             ["docker", "compose", "ps", "--format", "table {{.Name}}\t{{.Image}}\t{{.Status}}"])
    run_host(tr, "docker exec ids-kafka kafka-topics --bootstrap-server localhost:29092 --list",
             ["docker", "exec", "ids-kafka", "kafka-topics", "--bootstrap-server",
              "localhost:29092", "--list"])

    # Step 2: the two boards
    tr.add("host", "step", "2")
    for h, grep in ((args.j1, "anomaly"), (args.j2, "ids_pipeline_model")):
        remote = (f"hostname -I | cut -d' ' -f1; nvpmodel -q 2>/dev/null | head -1; "
                  f"ls ~/Thesis_IDS/jetson/model | grep {grep}")
        run_host(tr, f'ssh {U}@{h} "{remote}"', SSH + [f"{U}@{h}", remote])

    base = (f"KAFKA_BOOTSTRAP_SERVERS={mac_ip}:9092 POSTGRES_HOST={mac_ip} POSTGRES_PORT=5433 "
            f"INFLUXDB_URL=http://{mac_ip}:8086")
    # Step 3: anomaly gate on Jetson #1
    tr.add("j1", "step", "3")
    g = start_node(tr, "j1", args.j1, f"{base} KAFKA_GROUP_ID=ids-demo-{tag}",
                   "EDGE_NODE_ID=jetson-nano-1 EDGE_NODE_ROLE=anomaly_gate ALERT_ENABLED=0 "
                   "python edge/kafka_consumer.py", "ANOMALY GATE READY")
    g[2].wait(120)
    # Step 4: PySpark classifier on Jetson #2
    tr.add("j2", "step", "4")
    c = start_node(tr, "j2", args.j2, f"{base} KAFKA_CLASSIFIER_GROUP_ID=ids-demo-clf-{tag}",
                   "EDGE_NODE_ID=jetson-nano-2 EDGE_NODE_ROLE=classifier ALERT_ENABLED=1 "
                   "python edge/kafka_consumer.py", "CLASSIFIER READY")
    c[2].wait(240)
    time.sleep(3)

    # Step 5: replay CICIDS2017 from the Mac, dashboard recorded meanwhile
    rec = subprocess.Popen(["python", os.path.join(ROOT, "demo", "record_grafana.py"),
                            "--seconds", str(args.send_seconds + args.tail_seconds + 5),
                            "--out-dir", args.out], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    while not os.path.exists(os.path.join(args.out, "frames", "00000.png")):
        time.sleep(0.2)
    tr.add("host", "step", "5")
    tr.add("host", "cmd", f"python sender/data_sender.py --csv sender/replay_cicids2017.csv "
                          f"--rate {args.rate} --broker localhost:9092")
    tr.add("host", "mark", "send_start")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    s = subprocess.Popen(["python", "sender/data_sender.py", "--csv", "sender/replay_cicids2017.csv",
                          "--rate", str(args.rate), "--broker", "localhost:9092"],
                         cwd=JETSON, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    sth = threading.Thread(target=pump, args=(s, tr, "host"), daemon=True)
    sth.start()
    time.sleep(args.send_seconds)
    tr.add("host", "out", "^C")
    s.send_signal(signal.SIGINT)
    sth.join(20)
    tr.add("host", "mark", "send_end")
    rec.wait()
    tr.add("host", "mark", "record_end")

    # Step 6: stop both nodes; their shutdown prints the final statistics
    tr.add("j1", "step", "6")
    for pane, h, node in (("j1", args.j1, g), ("j2", args.j2, c)):
        tr.add(pane, "out", "^C")
        subprocess.run(SSH + [f"{U}@{h}", "pkill -INT -f '[k]afka_consumer.py' || true"])
    for node in (g, c):
        node[1].join(60)
        node[0].terminate()

    verify_step(tr, tr.t0 + mark_time(tr, "send_start") - 1, tr.t0 + mark_time(tr, "send_end") + 1)
    tr.save(os.path.join(args.out, "transcript.json"),
            {"mac_ip": mac_ip, "j1": args.j1, "j2": args.j2, "rate": args.rate,
             "send_seconds": args.send_seconds})
    print(f"[session] {len(tr.events)} events -> {args.out}/transcript.json")


if __name__ == "__main__":
    main()
