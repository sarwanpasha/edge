"""hop.py -- hop diameter h(G) from the source (min-hop count along shortest paths) and pass count of
synchronous streaming Bellman-Ford (= h(G) + 1) for every graph instance of a sweep.
usage: python hop.py <sweep> [--submit]   -> results/hop/<sweep>_<g>.json
"""
import json, os, sys, subprocess
import numpy as np
import sweep as S
from edge_sssp import make_graph, sssp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def hop_diameter(n, u, v, w, src, unit):
    d = sssp(n, u, v, w, src, unit=unit)
    # shortest-path DAG: edges tight in either direction
    t1 = np.isfinite(d[u]) & (np.abs(d[u] + w - d[v]) <= 1e-9 * np.maximum(1, d[v]))
    t2 = np.isfinite(d[v]) & (np.abs(d[v] + w - d[u]) <= 1e-9 * np.maximum(1, d[u]))
    a = np.concatenate([u[t1], v[t2]]); b = np.concatenate([v[t1], u[t2]])
    A = csr_matrix((np.ones(a.size), (a, b)), shape=(n, n))
    hops = shortest_path(A, directed=True, unweighted=True, indices=src)
    reach = np.isfinite(d)
    return int(np.max(hops[reach])), int(reach.sum()), float(np.mean(hops[reach])), int((reach).sum() - 1)

if __name__ == '__main__':
    sw = sys.argv[1]
    if '--run' in sys.argv:
        g = int(sys.argv[3]); specs = json.loads(open(f'jobs/hop_{sw}_{g}.json').read()); out = []
        for sp in specs:
            rng = np.random.default_rng(sp.get('seed', 0)); n, u, v, w = make_graph(sp, rng)
            src = sp.get('source', 0); unit = sp.get('unit', True)
            h, nreach, meanh, _ = hop_diameter(n, u, v, w, src, unit)
            out.append(dict(spec=sp, n=int(n), m=int(u.size), h=h, bf_passes=h + 1, reach=nreach, mean_hops=meanh))
            print(sp.get('type'), n, u.size, 'h=', h, flush=True)
            os.makedirs('../results/hop', exist_ok=True); json.dump(out, open(f'../results/hop/{sw}_{g}.json', 'w'))
        sys.exit(0)
    J = S.jobs_for(sw) if sw not in ('realsmall', 'small') else S.jobs_extra(sw)
    # unique graph instances (drop s_mult / eps / configs)
    seen = {}; U = []
    for sp in J:
        key = json.dumps({k: v for k, v in sp.items() if k not in ('s_mult', 'eps', 'configs', 'R_max', 'cert_every', 'spanner')}, sort_keys=True)
        if key not in seen: seen[key] = 1; U.append(json.loads(key))
    if sw == 'real':
        U = [dict(u_, source=srcv) for u_ in U for srcv in (0, 1, 2)]
    pk = 2 if sw == 'real' else 8
    groups = [U[i:i + pk] for i in range(0, len(U), pk)]
    os.makedirs('jobs', exist_ok=True)
    for g, grp in enumerate(groups):
        json.dump(grp, open(f'jobs/hop_{sw}_{g}.json', 'w'))
        big = any(x.get('n', 0) >= 500000 or x.get('ncloud', 0) >= 500000 or x.get('blocks', 0) * x.get('L', 0) >= 500000 or x.get('layers', 0) * x.get('width', 0) >= 500000 or x.get('type') == 'file' or x.get('d', 0) >= 1000 for x in grp)
        txt = f"""#!/bin/bash
#SBATCH --job-name=edge_hop{sw}_{g}
#SBATCH --account={S.ACCOUNT}
#SBATCH --partition={S.PARTITIONS}
#SBATCH --gpus=1
#SBATCH --cpus-per-task={16 if big else 8}
#SBATCH --mem={'180G' if big else '64G'}
#SBATCH --time={'03:00:00' if big else '01:00:00'}
#SBATCH --output=logs/hop{sw}_{g}_%j.out
#SBATCH --requeue
# activate your Python environment here if needed
cd {os.getcwd()}
python hop.py {sw} --run {g}
"""
        fn = f'jobs/hop_{sw}_{g}.sbatch'; open(fn, 'w').write(txt)
        if '--submit' in sys.argv:
            o = subprocess.run(['sbatch', fn], capture_output=True, text=True); print(fn, o.stdout.strip(), o.stderr.strip())
    print(f'{len(groups)} jobs ({len(U)} graphs) for {sw}')
