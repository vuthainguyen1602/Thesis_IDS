#!/usr/bin/env python
"""Phase-A demo session: distributed training on the Spark Standalone cluster.

Runs for real and records a transcript, as run_session.py does for the edge
demo: the Spark master on the Mac, a worker on each Jetson, a check that both
workers are alive (plus a recording of the Spark Master UI), then
demo/train_demo.py submitted through cluster/run_ml_remote.sh with Jetson #2
as the driver, while the job's Spark UI is recorded. Jetson #2 is the driver
because Jetson #1 has only ~2.6 GB free next to other jobs; executors use 2 GB.

    python demo/run_session_a.py --out demo/out/phaseA_<tag>
"""
import argparse
import os
import subprocess
import threading
import time
import urllib.request

from run_session import JETSON, ROOT, SSH, U, Transcript, pump, run_host

SPARK_HOME = os.environ.get("SPARK_HOME", os.path.expanduser("~/spark-3.5.3-bin-hadoop3"))


def wait_url(url, timeout):
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def start_worker(tr, pane, host):
    # Without sbin/start-worker.sh (pip pyspark) the worker runs in the
    # foreground, so it is started in the background and its log read back.
    cmds = [f"ssh {U}@{host}", "cd ~/Thesis_IDS",
            "nohup ./cluster/start_worker.sh > /tmp/start_worker.log 2>&1 &",
            "sleep 6; cat /tmp/start_worker.log"]
    for c in cmds[:3]:
        tr.add(pane, "cmd", c)
        time.sleep(0.4)
    remote = ("cd ~/Thesis_IDS && (nohup ./cluster/start_worker.sh > /tmp/start_worker.log 2>&1 < /dev/null &) "
              "&& sleep 6 && cat /tmp/start_worker.log")
    tr.add(pane, "cmd", cmds[3])
    p = subprocess.Popen(SSH + ["-n", f"{U}@{host}", remote], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    pump(p, tr, pane)
    p.wait()


def record(url, out_dir, seconds, until=None, reload_every=0):
    argv = ["python", os.path.join(ROOT, "demo", "record_grafana.py"), "--url", url,
            "--seconds", str(seconds), "--out-dir", out_dir, "--fps", "1"]
    if until:
        argv += ["--until-file", until]
    if reload_every:
        argv += ["--reload-every", str(reload_every)]
    return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--j1", default="192.168.1.81")
    ap.add_argument("--j2", default="192.168.1.39")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tr = Transcript()
    _add = tr.add

    def add_and_echo(pane, kind, text):  # live log of the session
        _add(pane, kind, text)
        print(f"[{pane}:{kind}] {text}", flush=True)
    tr.add = add_and_echo

    tr.add("host", "step", "1")
    run_host(tr, "./cluster/start_master_mac.sh", ["./cluster/start_master_mac.sh"], cwd=ROOT)
    wait_url("http://localhost:8080", 30)

    tr.add("j1", "step", "2")
    t1 = threading.Thread(target=start_worker, args=(tr, "j1", args.j1))
    t2 = threading.Thread(target=start_worker, args=(tr, "j2", args.j2))
    t1.start(); time.sleep(0.5); t2.start(); t1.join(); t2.join()
    time.sleep(10)  # let both workers register

    tr.add("host", "step", "3")
    run_host(tr, "python demo/cluster_status.py",
             ["python", os.path.join(ROOT, "demo", "cluster_status.py")], cwd=ROOT)
    tr.add("host", "mark", "master_ui_start")
    rec = record("http://localhost:8080", os.path.join(args.out, "ui_master"), 10)
    rec.wait()
    tr.add("host", "mark", "master_ui_end")

    tr.add("host", "step", "4")
    env = dict(os.environ, CLUSTER_DRIVER=f"{U}@{args.j2}", SPARK_DRIVER_HOST=args.j2,
               SPARK_DRIVER_MEMORY="2g", SPARK_EXECUTOR_MEMORY="2g", PYTHONUNBUFFERED="1")
    cmd = (f"CLUSTER_DRIVER={U}@{args.j2} SPARK_DRIVER_HOST={args.j2} SPARK_DRIVER_MEMORY=2g "
           f"SPARK_EXECUTOR_MEMORY=2g ./cluster/run_ml_remote.sh demo/train_demo.py")
    tr.add("host", "cmd", cmd)
    done = os.path.join(args.out, "train.done")
    p = subprocess.Popen(["./cluster/run_ml_remote.sh", "demo/train_demo.py"], cwd=ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    th = threading.Thread(target=pump, args=(p, tr, "host"), daemon=True)
    th.start()
    app_ui = f"http://{args.j2}:4040/jobs/"
    rec2 = None
    if wait_url(app_ui, 600):
        tr.add("host", "mark", "app_ui_start")
        rec2 = record(app_ui, os.path.join(args.out, "ui_app"), 3600, until=done, reload_every=3)
    p.wait()
    th.join(10)
    open(done, "w").close()
    tr.add("host", "mark", "train_end")
    if rec2:
        rec2.wait()

    # Clean-up (not part of the video): stop workers and master.
    for h in (args.j1, args.j2):
        subprocess.run(SSH + [f"{U}@{h}", "pkill -f '[o]rg.apache.spark.deploy.worker.Worker' || true"])
    subprocess.run([os.path.join(SPARK_HOME, "sbin", "stop-master.sh")], capture_output=True)
    tr.save(os.path.join(args.out, "transcript.json"), {"j1": args.j1, "j2": args.j2})
    print(f"[session-a] {len(tr.events)} events -> {args.out}/transcript.json")


if __name__ == "__main__":
    main()
