"""realsrc.py -- real graphs with two additional source vertices (1 and 2), budgets n/16, n/4, n, all three methods.
usage: python realsrc.py [--submit]  -> results/realsrc/<i>.json
"""
import json, os, sys, subprocess
import sweep as S
FILES = ['soc-pokec-relationships.txt', 'com-lj.ungraph.txt', 'com-orkut.ungraph.txt']
J = []
for fn in FILES:
    for src in (1, 2):
        for sm in (0.0625, 0.25, 1.0):
            J.append(dict(type='file', path=os.path.join('data', fn), seed=0, source=src, s_mult=sm, eps=0.1, R_max=400,
                          configs=[dict(mode='floor', pin=True, tag='ours'), dict(mode='fixed', pin=True, tag='fixed_pin'), dict(mode='fixed', pin=False, tag='ahs')]))
os.makedirs('jobs', exist_ok=True); os.makedirs('logs', exist_ok=True)
for i, sp in enumerate(J):
    txt = f"""#!/bin/bash
#SBATCH --job-name=edge_realsrc_{i}
#SBATCH --account={S.ACCOUNT}
#SBATCH --partition={S.PARTITIONS}
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=logs/realsrc_{i}_%j.out
#SBATCH --requeue
# activate your Python environment here if needed
export OMP_NUM_THREADS=16
cd {os.getcwd()}
mkdir -p ../results/realsrc
python edge_sssp.py '{json.dumps(sp)}' ../results/realsrc/{i}.json
"""
    fn = f'jobs/realsrc_{i}.sbatch'; open(fn, 'w').write(txt)
    if '--submit' in sys.argv:
        o = subprocess.run(['sbatch', fn], capture_output=True, text=True); print(fn, o.stdout.strip(), o.stderr.strip())
print(f'{len(J)} jobs')
