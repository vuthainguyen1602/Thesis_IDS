#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Clock offset and round-trip time of the edge boards against this host.

End-to-end latency is verdict time (board clock) minus send time (this
host's clock), so the offset between the two clocks goes straight into the
number. This records it: N requests go through one SSH channel per board,
the board answers each with its time.time(), and
offset = board - midpoint(send, receive) for the request with the smallest
round trip, whose half-RTT bounds the error. ``ping`` gives the network RTT
on its own. Run on the sender host:

    python scripts/clock_offset.py --out clock.json \
        bvdung@192.168.1.81:192.168.1.50 bvdung@192.168.1.39:192.168.1.204

Each target is user@host[:hostkey_alias]; the alias is the name the board's
key is filed under in known_hosts when its IP has changed.
"""

import argparse
import json
import re
import subprocess
import time

REMOTE = ("python3 -u -c 'import sys,time\n"
          "for _ in sys.stdin: print(repr(time.time()), flush=True)'")


def ssh_offset(target, alias, n):
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    if alias:
        cmd += ["-o", f"HostKeyAlias={alias}"]
    p = subprocess.Popen(cmd + [target, REMOTE], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True, bufsize=1)
    samples = []
    for _ in range(n):
        t0 = time.time()
        p.stdin.write("x\n")
        p.stdin.flush()
        remote = float(p.stdout.readline())
        t1 = time.time()
        samples.append((t1 - t0, remote - (t0 + t1) / 2))
        time.sleep(0.05)
    p.stdin.close()
    p.wait()
    rtt, offset = min(samples)
    rtts = sorted(s[0] for s in samples)
    return {
        "offset_s": round(offset, 6),
        "error_bound_s": round(rtt / 2, 6),
        "ssh_rtt_min_ms": round(rtt * 1000, 2),
        "ssh_rtt_median_ms": round(rtts[len(rtts) // 2] * 1000, 2),
        "samples": n,
    }


def ping_rtt(host, count):
    try:
        out = subprocess.run(["ping", "-c", str(count), "-i", "0.2", host],
                             capture_output=True, text=True, timeout=count * 0.5 + 15).stdout
    except (subprocess.TimeoutExpired, OSError):
        return {}
    m = re.search(r"= ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms", out)
    loss = re.search(r"([\d.]+)% packet loss", out)
    if not m:
        return {}
    return {
        "ping_rtt_min_ms": float(m.group(1)), "ping_rtt_avg_ms": float(m.group(2)),
        "ping_rtt_max_ms": float(m.group(3)), "ping_rtt_stddev_ms": float(m.group(4)),
        "ping_loss_pct": float(loss.group(1)) if loss else None, "ping_count": count,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("targets", nargs="+", help="user@host[:hostkey_alias]")
    ap.add_argument("-n", type=int, default=50, help="requests per board")
    ap.add_argument("--ping", type=int, default=20, help="ping packets per board (0 = skip)")
    ap.add_argument("--out", default=None, help="write the JSON here as well as to stdout")
    args = ap.parse_args()

    hosts = []
    for target in args.targets:
        alias = None
        if ":" in target:
            target, alias = target.split(":", 1)
        host = target.split("@", 1)[-1]
        entry = {"host": host, "target": target, "taken_at": time.time()}
        entry.update(ssh_offset(target, alias, args.n))
        if args.ping > 0:
            entry.update(ping_rtt(host, args.ping))
        hosts.append(entry)
    payload = {"taken_at": time.time(), "hosts": hosts}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
