#!/bin/bash
# A self-contained run on a synthetic unique-parent layered graph (n = 20,001, m = 1.7M) with all five methods
# of the paper: PACE (tree pinning), fixed step + tree pinning, fresh resampling, constraint generation, and
# certified Bellman-Ford. Takes about a minute on a laptop and needs about 2 GB of memory.
set -e
mkdir -p out
python edge_sssp.py '{"type":"layered","layers":40,"width":500,"back":1,"decoy":200,"seed":0,"s_mult":4,"eps":0.1,"R_max":400,
 "configs":[{"mode":"pin2","pin":true,"treepin":true,"tag":"pace"},
            {"mode":"fixed","pin":true,"treepin":true,"tag":"fixed_treepin"},
            {"mode":"fixed","pin":false,"tag":"fresh"},
            {"mode":"cg","pin":true,"tag":"cg"},
            {"mode":"bfcert","pin":true,"tag":"certified_bf"}]}' out/example.json
python summarize.py out/example.json
