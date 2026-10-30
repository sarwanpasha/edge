"""
sweep.py -- generate sbatch jobs for the experiments.
usage: python sweep.py <sweep_name> [--submit]
Each job = one graph instance (spec) with several algorithm configs; writes results/<name>/<jobid>.json
Jobs are named edge_<sweep>_<i> and submitted across a partition list so Slurm places them wherever they start first.
"""
import json, os, sys, itertools, subprocess

PARTITIONS = os.environ.get("EDGE_SLURM_PARTITIONS", "YOUR_PARTITION")
ACCOUNT = os.environ.get("EDGE_SLURM_ACCOUNT", "YOUR_ACCOUNT")

CONFIGS_MAIN = [
    dict(mode='harmonic', pin=True,  tag='ours'),
    dict(mode='fixed',    pin=True,  tag='fixed_pin'),
    dict(mode='fixed',    pin=False, tag='ahs'),
]
CONFIGS_ABL = CONFIGS_MAIN + [
    dict(mode='adaptive', pin=True,  tag='adaptive_cmin'),
    dict(mode='harmonic', pin=False, tag='harmonic_nopin'),
]

def jobs_for(sweep):
    J = []
    if sweep == 'scale':
        # scaling in n for 4 families, 3 seeds, s = 4n
        for seed in range(3):
            for n in [20000, 50000, 100000, 200000, 500000, 1000000]:
                d = 200
                J.append(dict(type='er', n=n, d=d, seed=seed, s_mult=4, eps=0.1, R_max=400, configs=CONFIGS_MAIN))
                J.append(dict(type='bottleneck', path=50, ncloud=n, d=d, seed=seed, s_mult=4, eps=0.1, R_max=400, configs=CONFIGS_MAIN))
                width = 500; layers = max(4, n // width)
                J.append(dict(type='layered', layers=layers, width=width, back=1, decoy=d, seed=seed, s_mult=4, eps=0.1, R_max=400, configs=CONFIGS_MAIN))
                L = 200; blocks = max(4, n // L)
                J.append(dict(type='thick', blocks=blocks, L=L, back=1, seed=seed, s_mult=4, eps=0.1, R_max=400, configs=CONFIGS_MAIN))
    elif sweep == 'budget':
        for seed in range(3):
            for s_mult in [1, 2, 4, 8, 16]:
                n = 100000; d = 200
                J.append(dict(type='er', n=n, d=d, seed=seed, s_mult=s_mult, eps=0.1, R_max=600, configs=CONFIGS_ABL))
                J.append(dict(type='bottleneck', path=50, ncloud=n, d=d, seed=seed, s_mult=s_mult, eps=0.1, R_max=600, configs=CONFIGS_ABL))
                J.append(dict(type='layered', layers=200, width=500, back=1, decoy=d, seed=seed, s_mult=s_mult, eps=0.1, R_max=600, configs=CONFIGS_ABL))
                J.append(dict(type='thick', blocks=500, L=200, back=1, seed=seed, s_mult=s_mult, eps=0.1, R_max=600, configs=CONFIGS_ABL))
    elif sweep == 'eps':
        for seed in range(2):
            for eps in [0.02, 0.05, 0.1, 0.2, 0.5]:
                n = 100000; d = 200
                J.append(dict(type='bottleneck', path=50, ncloud=n, d=d, seed=seed, s_mult=4, eps=eps, R_max=800, configs=CONFIGS_MAIN))
                J.append(dict(type='layered', layers=200, width=500, back=1, decoy=d, seed=seed, s_mult=4, eps=eps, R_max=800, configs=CONFIGS_MAIN))
    elif sweep == 'weighted':
        for seed in range(2):
            n = 100000; d = 200
            for wmax in [10, 100]:
                J.append(dict(type='er', n=n, d=d, seed=seed, s_mult=4, eps=0.1, R_max=600, unit=False, wmax=wmax, configs=CONFIGS_MAIN))
                J.append(dict(type='bottleneck', path=50, ncloud=n, d=d, seed=seed, s_mult=4, eps=0.1, R_max=600, unit=False, wmax=wmax, configs=CONFIGS_MAIN))
                J.append(dict(type='thick', blocks=500, L=200, back=1, seed=seed, s_mult=4, eps=0.1, R_max=600, unit=False, wmax=wmax, configs=CONFIGS_MAIN))
    elif sweep == 'nospanner':
        for seed in range(2):
            n = 100000; d = 200
            for fam in [dict(type='er', n=n, d=d), dict(type='bottleneck', path=50, ncloud=n, d=d),
                        dict(type='layered', layers=200, width=500, back=1, decoy=d), dict(type='thick', blocks=500, L=200, back=1)]:
                J.append(dict(fam, seed=seed, s_mult=4, eps=0.1, R_max=600, spanner='none', configs=CONFIGS_MAIN))
    elif sweep == 'real':
        for fn in ['com-orkut.ungraph.txt', 'com-lj.ungraph.txt', 'soc-pokec-relationships.txt', 'com-youtube.ungraph.txt', 'com-dblp.ungraph.txt']:
            for s_mult in [2, 4]:
                J.append(dict(type='file', path=os.path.join('data', fn), seed=0, s_mult=s_mult, eps=0.1, R_max=400, configs=CONFIGS_MAIN, cert_every=1))
    elif sweep == 'dense':
        # denser graphs: d = 1000 (m = 500n), n up to 200k
        for seed in range(2):
            for n in [50000, 100000, 200000]:
                d = 1000
                J.append(dict(type='er', n=n, d=d, seed=seed, s_mult=4, eps=0.1, R_max=400, configs=CONFIGS_MAIN))
                J.append(dict(type='bottleneck', path=50, ncloud=n, d=d, seed=seed, s_mult=4, eps=0.1, R_max=400, configs=CONFIGS_MAIN))
                J.append(dict(type='layered', layers=max(4, n // 500), width=500, back=1, decoy=d, seed=seed, s_mult=4, eps=0.1, R_max=400, configs=CONFIGS_MAIN))
    else:
        raise ValueError(sweep)
    return J

SBATCH = """#!/bin/bash
#SBATCH --job-name=edge_{sweep}_{i}
#SBATCH --account={account}
#SBATCH --partition={partitions}
#SBATCH --gpus=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output=logs/{sweep}_{i}_%j.out
#SBATCH --requeue
# activate your Python environment here if needed
export OMP_NUM_THREADS={cpus}
cd {cwd}
mkdir -p ../results/{sweep}
python edge_sssp.py '{spec}' ../results/{sweep}/{i}.json
"""

if __name__ == '__main__':
    sweep = sys.argv[1]; submit = '--submit' in sys.argv
    J = jobs_for(sweep)
    os.makedirs('jobs', exist_ok=True); os.makedirs('logs', exist_ok=True)
    for i, spec in enumerate(J):
        big = spec.get('n', 0) >= 500000 or spec.get('ncloud', 0) >= 500000 or spec.get('blocks', 0) * spec.get('L', 0) >= 500000 or spec.get('layers', 0) * spec.get('width', 0) >= 500000 or spec.get('type') == 'file' or spec.get('d', 0) >= 1000
        cpus = 16 if big else 8
        mem = '180G' if big else '64G'
        tm = '06:00:00' if big else '03:00:00'
        txt = SBATCH.format(sweep=sweep, i=i, account=ACCOUNT, partitions=PARTITIONS, cpus=cpus, mem=mem, time=tm, spec=json.dumps(spec), cwd=os.getcwd())
        fn = f'jobs/{sweep}_{i}.sbatch'
        open(fn, 'w').write(txt)
        if submit:
            out = subprocess.run(['sbatch', fn], capture_output=True, text=True)
            print(fn, out.stdout.strip(), out.stderr.strip())
    print(f'{len(J)} jobs for sweep {sweep}')
