# Run logs for the sweep

One file per run, with Spark's progress-bar lines stripped: 1.3 MB of `[Stage …]`
redraws down to 112 KB. What is left is what the CSVs do not record and what a
later reader would have to take on trust otherwise:

- how many workers were alive and which host drove the run
- Spark and JVM versions, and the executor hosts
- the common leak-free feature count (75) and which columns each dataset lacks
- the four metric rows as the run printed them, so a CSV can be checked against
  the run that produced it
- for `sweep_night2.log` and `sweep_confirm.log`, the sweep-level timeline:
  per-run start, the worker check before each run, and the outcome

`20260926-1523_run3_seed23.log` is the run that lost Jetson #2 for three minutes;
its CSV is in `../excluded/` and the event is visible here. `adapt_seed42.log` is
the run behind `cross_dataset_adaptation_high_regime.csv`.

The raw logs these were trimmed from are not in the repository and have since
been deleted from the Mac; a new sweep writes its own under `output/xd_sweep/`,
which the `*.log` rule in `.gitignore` keeps out. So these files are the only
surviving record of how the 18 runs executed.
