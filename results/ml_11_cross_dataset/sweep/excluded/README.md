# Runs kept out of the summary

`xd_sweep_summary.py` globs `*.csv` in the parent directory, so anything here is
recorded but not averaged. Each file is a run that completed yet did not meet the
condition the sweep is built to test: identical code, data *and* cluster
topology across repetitions.

## 20260926-1523_run3_seed23.csv

Jetson #2 (192.168.1.204) dropped off the network at 17:46, three minutes into
this run: ping, SSH and ARP all failed, the master marked the worker DEAD and the
application fell from 8 cores to 4. The board came back at 17:49 on its own and
the run finished on 8 cores again. It was a link outage, not a crash: uptime
stayed at 13 days and the worker JVMs were the same processes as before.

Spark recomputed the lost tasks, so the numbers are valid. They are excluded
because the partitioning and aggregation order, which is the source of the
variance this sweep measures, ran under two different topologies inside one run.
Seed 23 was repeated cleanly afterwards (label `rerun_run1_seed23`).
