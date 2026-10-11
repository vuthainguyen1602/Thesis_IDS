#!/usr/bin/env python
"""Print the Spark master's status and its alive workers (from the master's JSON API).

    python demo/cluster_status.py [http://localhost:8080]
"""
import json
import sys
import urllib.request

url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080").rstrip("/") + "/json/"
d = json.load(urllib.request.urlopen(url))
print(f"Master: {d['url']} | status: {d['status']} | alive workers: {d['aliveworkers']}")
print(f"Cluster: {d['cores']} cores, {d['memory'] // 1024} GiB memory")
for w in d["workers"]:
    if w["state"] == "ALIVE":
        print(f"  worker {w['host']:15} {w['state']:6} {w['cores']} cores  {w['memory'] // 1024} GiB")
