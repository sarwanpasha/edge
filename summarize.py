"""summarize.py -- print a compact table for one or more result files written by edge_sssp.py.

usage: python summarize.py out/example.json [more.json ...]

For every method (config tag) in each file it prints the number of rounds until the loop stopped, which certificate
stopped it (exact = no violating edge, window = window certificate at most eps), the true approximation ratio of the
output computed from exact distances, the stored structure at the end (spanner + round trees + stored violators) as a
fraction of m, the number of stored violators, the largest number of rounds in which any edge failed, and the first
bulk round r0 (the first round whose violators were stored directly).
"""
import json, sys

def summarize(path):
    D = json.load(open(path)); sp = D['spec']
    print(f"{path}: type={sp['type']} n={D['n']} m={D['m']} k={D['k']} eps={sp.get('eps', 0.1)} s={sp.get('s_mult', 8.0)}n |H|={D['H_size']}")
    print(f"  {'method':16s} {'rounds':>6s} {'stop':>7s} {'true ratio':>10s} {'stored/m':>9s} {'violators':>9s} {'Fmax':>5s} {'r0':>4s} {'passes':>6s}")
    for tag, x in D.items():
        if not isinstance(x, dict) or 'recs' not in x: continue
        rc = x['recs']; last = rc[-1]; R = x['R_used']
        stop = 'exact' if x['exact_round'] else ('window' if x['cert_round'] else 'cap')
        r0 = next((r['r'] for r in rc if r.get('dpin')), '-')
        passes = R + 1 if x['mode'] == 'bfcert' else 2 * R + 1
        print(f"  {tag:16s} {R:6d} {stop:>7s} {x['final_true_ratio']:10.4f} {last['stored'] / D['m']:9.3f} {last.get('vstored', 0):9d} {last['Fmax']:5d} {str(r0):>4s} {passes:6d}")

if __name__ == '__main__':
    for p in sys.argv[1:]:
        summarize(p)
