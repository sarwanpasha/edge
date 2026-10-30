"""rerun3.py -- tree-pinning configs {ours: pin2+treepin, fpt: fixed+treepin} for every sweep (and realsrc). usage: python rerun2.py <sweep> [--submit]"""
import json, os, sys, subprocess
import sweep as S
PACK = {'real': 2, 'realsmall': 2, 'realsrc': 2, 'scale': 6, 'budget': 6, 'dense': 3}
sw = sys.argv[1]; submit = '--submit' in sys.argv
if sw == 'realsrc':
    J = [dict(type='file', path=os.path.join('data', fn), seed=0, source=src, s_mult=sm, eps=0.1, R_max=400)
         for fn in ['soc-pokec-relationships.txt', 'com-lj.ungraph.txt', 'com-orkut.ungraph.txt'] for src in (1, 2) for sm in (0.0625, 0.25, 1.0)]
else:
    J = S.jobs_for(sw) if sw not in ('realsmall', 'small') else S.jobs_extra(sw)
cfg = [dict(mode='pin2', pin=True, treepin=True, tag='ours'), dict(mode='fixed', pin=True, treepin=True, tag='fpt')]
pk = PACK.get(sw, 8)
groups = [list(range(i, min(i + pk, len(J)))) for i in range(0, len(J), pk)]
HEAD = """#!/bin/bash
#SBATCH --job-name=edge_t{sw}_{g}
#SBATCH --account={acc}
#SBATCH --partition={parts}
#SBATCH --gpus=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={tm}
#SBATCH --output=logs/t{sw}_{g}_%j.out
#SBATCH --requeue
# activate your Python environment here if needed
export OMP_NUM_THREADS={cpus}
cd {cwd}
mkdir -p ../results/{sw}_t
"""
os.makedirs('jobs', exist_ok=True); os.makedirs('logs', exist_ok=True)
for g, idx in enumerate(groups):
    big = any(J[i].get('n', 0) >= 500000 or J[i].get('ncloud', 0) >= 500000 or J[i].get('blocks', 0) * J[i].get('L', 0) >= 500000
              or J[i].get('layers', 0) * J[i].get('width', 0) >= 500000 or J[i].get('type') == 'file' or J[i].get('d', 0) >= 1000 for i in idx)
    cpus = 16 if big else 8; mem = '180G' if big else '64G'; tm = '08:00:00' if big else '04:00:00'
    txt = HEAD.format(sw=sw, g=g, acc=S.ACCOUNT, parts=S.PARTITIONS, cpus=cpus, mem=mem, tm=tm, cwd=os.getcwd())
    for i in idx:
        spec = dict(J[i]); spec['configs'] = cfg
        txt += f"python edge_sssp.py '{json.dumps(spec)}' ../results/{sw}_t/{i}.json\n"
    fn = f'jobs/t{sw}_{g}.sbatch'; open(fn, 'w').write(txt)
    if submit:
        out = subprocess.run(['sbatch', fn], capture_output=True, text=True); print(fn, out.stdout.strip(), out.stderr.strip())
print(f'{len(groups)} jobs ({len(J)} specs) for sweep {sw}')
