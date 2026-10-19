# Terminal figures for the two Jetson nodes (thesis Fig. "pipeline_terminal")

`j1.log` and `j2.log` are the console logs (`~/edge_jetson-nano-{1,2}.log`) of a short
pipeline-split run on 2026-10-20 (`papers/soict2026/run_dist_bench.sh split`, 20 flows/s,
90 s window). `render.py` draws `../pipeline_terminal_jetson{1,2}.png` from selected lines,
verbatim; Spark progress bars (carriage-return lines) are dropped and `[...]` marks skipped lines.

    python3 render.py j1.log ../pipeline_terminal_jetson1.png "bvdung@ubuntu: ~/Thesis_IDS/jetson — ssh 192.168.1.81 (jetson-nano-1)" "2-23,32-46,gap,84-88"
    python3 render.py j2.log ../pipeline_terminal_jetson2.png "bvdung@ubuntu: ~/Thesis_IDS/jetson — ssh 192.168.1.39 (jetson-nano-2)" "7-35,40-55,gap,81-86"
