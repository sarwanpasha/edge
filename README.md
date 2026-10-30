# PACE: a certified sample-and-solve loop for semi-streaming shortest paths

This repository contains the simulator, the experiment drivers, the summarized results, and the scripts that
produce the tables and figures of the paper *Adaptive Steps and Certificates for Semi-Streaming Shortest Paths*.

The algorithm, PACE (pinned, adaptive-step, certified), is a variant of the multiplicative-weights loop of
Assadi, Hoppenworth, and Sundaresan (SODA 2026) for (1+ε)-approximate single-source shortest paths in the
semi-streaming model. Each round samples a subgraph in proportion to edge weights, computes a shortest-path tree
of the spanner, the stored edges, and the sample, and then either stores the edges that violate the tree
distances (when they fit in half the sample budget) or raises their weights with an adaptive multiplier. The
loop keeps the round trees, which makes the stored distances monotone, and stops as soon as a window certificate
proves the current tree (1+ε)-approximate, or as soon as no edge violates, in which case the tree is exact.

## Requirements

Python 3.10 or newer with NumPy, SciPy, pandas, and Matplotlib:

```
pip install -r requirements.txt
```

The simulator keeps the whole graph in memory as edge arrays. A pass of a streaming algorithm is one vectorized
sweep over all edges; the stored structure (spanner, round trees, stored violators, distance vectors) is exactly
what a streaming implementation would keep, and its size is reported for every run.

## A working example

```
bash example.sh
```

builds a unique-parent layered graph with 20,001 vertices and 1.7 million edges, runs the five methods compared in
the paper (PACE, fixed step with tree pinning, fresh resampling, constraint generation, and certified
Bellman-Ford) with sample budget s = 4n and ε = 0.1, writes `out/example.json`, and prints

```
  method           rounds    stop true ratio  stored/m violators  Fmax   r0 passes
  pace                 11   exact     1.0000     0.137     15345     1    1     23
  fixed_treepin        12   exact     1.0000     0.126         0     2    -     25
  fresh                20   exact     1.0000     0.119         0     2    -     41
  cg                   11   exact     1.0000     0.471     12561     1    1     23
  certified_bf         34  window     1.0294     0.083         0     4    -     35
```

The columns are the number of rounds until the method stopped, which certificate stopped it (`exact`: no
violating edge; `window`: the window certificate reached ε; `cap`: the round cap), the true ratio
max_x d(x)/dist(x) of the output computed from exact distances, the stored structure at the end as a fraction of
m, the number of violators stored directly, the largest number of rounds in which any edge failed, the first
bulk round r0, and the number of passes (2R+1 for the sampling loops, R+1 for certified Bellman-Ford). The run
takes about a minute and 2 GB of memory. Results vary slightly with the NumPy version because of the random
sampling; the structure (PACE and constraint generation in about 11 rounds, fresh resampling in about 20,
certified Bellman-Ford in about 35 with a window stop) is stable.

## Running one configuration

`edge_sssp.py` takes a JSON specification and an output path:

```
python edge_sssp.py '<spec>' out/result.json
```

The specification names the graph and the methods. Graph types:

| `type` | fields | description |
|---|---|---|
| `er` | `n`, `d` | G(n,p) with expected degree `d` |
| `bottleneck` | `path`, `ncloud`, `d` | a path of `path` edges from the source into a G(n,p) cloud |
| `layered` | `layers`, `width`, `back`, `decoy` | unique-parent layers with `decoy` intra-layer edges per vertex |
| `thick` | `blocks`, `L`, `back` | a chain of `blocks` cliques of size `L` |
| `file` | `path` | an edge list (SNAP format, one `u v` pair per line) |

Common fields: `seed`, `source` (default 0), `s_mult` (sample budget as a multiple of n), `eps`, `R_max`
(round cap), `unit` and `wmax` (random integer weights in [1, wmax] when `unit` is false), `spanner` (`"none"`
to run without the spanner), and `configs`, a list of methods. Each method has a `mode`, a `pin` flag, an
optional `treepin` flag, and a `tag`:

| method in the paper | config |
|---|---|
| PACE | `{"mode":"pin2","pin":true,"treepin":true}` |
| fixed step + tree pinning | `{"mode":"fixed","pin":true,"treepin":true}` |
| fresh resampling (reference loop with the certificate) | `{"mode":"fixed","pin":false}` |
| constraint generation | `{"mode":"cg","pin":true}` |
| certified Bellman-Ford | `{"mode":"bfcert","pin":true}` |

With `pin` true and `treepin` false the loop keeps every sampled edge instead of the round trees (the variant
discussed in Section 3 of the paper as the alternative that costs more space). The output JSON holds, per
method, the number of rounds, the exactness and certificate rounds, the final true ratio, the final certificate
value and window length, and a per-round record (violating edges, violating weight fraction, multiplier, largest
failure count, true ratio, stored edges, stored violators, largest relative jump).

## Reproducing the experiments

The experiments ran on a Slurm cluster. `sweep.py` defines the instance lists of the sweeps `scale`, `budget`,
`eps`, `weighted`, `nospanner`, `real`, `dense`, and the extra lists `realsmall` and `small`; `rerun3.py`
submits PACE and fixed step with tree pinning for a sweep, `rerun2.py` submits constraint generation,
`bfjobs.py` submits certified Bellman-Ford, `hop.py` computes hop diameters, and `realsrc.py` runs the real
graphs from additional source vertices. Set the Slurm account and partition list in the environment before
generating jobs, and put the SNAP edge lists (`soc-pokec-relationships.txt`, `com-lj.ungraph.txt`,
`com-orkut.ungraph.txt`) in `data/`:

```
export EDGE_SLURM_ACCOUNT=your_account
export EDGE_SLURM_PARTITIONS=your_partition
python rerun3.py scale --submit      # PACE and fixed step with tree pinning, results/scale_t/
python rerun2.py scale --submit      # constraint generation, results/scale_g/
python bfjobs.py scale --submit      # certified Bellman-Ford, results/scale_bf/
python hop.py scale --submit         # hop diameters, results/hop/
```

Without `--submit` the scripts only write the job files to `jobs/`. Every job is a sequence of
`python edge_sssp.py '<spec>' <output>` lines, so the same specifications can be run on any machine.
`tsum.py` collects the per-run JSON files into `t.csv` and `trajt.json`.

## Tables and figures

`results_summary/` contains the collected results of all runs reported in the paper (`t.csv`: PACE, fixed step
with tree pinning, and certified Bellman-Ford; `g.csv`: constraint generation; `summary_m.csv` and
`realsrc.csv`: fresh resampling; `hop.csv`: hop diameters; `trajt.json` and `traj_m.json`: per-round
trajectories at n = 10^6). Running

```
python make_tables_figures.py
```

writes the LaTeX tables and the figures of the paper to `figures/` and prints the numbers quoted in the text.
