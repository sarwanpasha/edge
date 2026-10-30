"""Tables and figures for the STOC version, version 4 (tree pinning). PACE = tag 'ours' in t.csv, fixed step + tree pinning = 'fpt' in t.csv,
certified Bellman-Ford = 'bfcert' in t.csv, constraint generation = 'cg' in g.csv, fresh resampling = 'ahs' in summary_m.csv / realsrc.csv,
hop diameters from hop.csv. Writes figures/tab_*.tex and figures/{scale,traj,budget}.pdf and prints the numbers used in the text."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = 'results_summary/'; OUT = 'figures/'
import os; os.makedirs(OUT, exist_ok=True)
pd.set_option('display.width', 250)

t = pd.read_csv(R + 't.csv'); t['fam'] = t.type.str.split('|').str[0]; t['file'] = t.type.str.split('|').str[4]
g = pd.read_csv(R + 'g.csv'); g['fam'] = g.type.str.split('|').str[0]; g['file'] = g.type.str.split('|').str[4]
o = pd.read_csv(R + 'summary_m.csv'); o['fam'] = o.type.astype(str).str.split('|').str[0]; o['file'] = o.type.astype(str).str.split('|').str[4]
H = pd.read_csv(R + 'hop.csv'); H['file'] = H.file.fillna(''); H['sweep'] = H.job.str.split('_').str[0]
RS = pd.read_csv(R + 'realsrc.csv')
key = ['sweep', 'job', 's_mult', 'seed']
oo = o[o.tag.isin(['fixed_pin', 'ahs'])].rename(columns={'R_used': 'R'})
BF = t[t.tag == 'bfcert'][['sweep', 'job', 'seed', 'source', 'file', 'R', 'exact', 'true']].rename(columns={'R': 'R_bf', 'exact': 'ex_bf', 'true': 'true_bf'})
G = t[t.tag == 'ours'].merge(t[t.tag == 'fpt'][key + ['R', 'pinned']].rename(columns={'R': 'R_fp', 'pinned': 'pin_fp'}), on=key, how='left') \
    .merge(oo[oo.tag == 'ahs'][key + ['R', 'cert_round']].rename(columns={'R': 'R_ahs', 'cert_round': 'cert_ahs'}), on=key, how='left') \
    .merge(g[g.tag == 'cg'][key + ['R', 'pinned', 'maxpinround', 'exact', 'true']].rename(columns={'R': 'R_cg', 'pinned': 'pin_cg', 'maxpinround': 'mpr_cg', 'exact': 'ex_cg', 'true': 'true_cg'}), on=key, how='left') \
    .merge(BF[BF.sweep.isin(['scale', 'dense'])][['sweep', 'job', 'seed', 'R_bf', 'ex_bf', 'true_bf']], on=['sweep', 'job', 'seed'], how='left')
G['s'] = G.s_mult * G.n; G['stored'] = (G.H + G.pinned) / G.m; G['stored_cg'] = (G.H + G.pin_cg) / G.m; G['stored_fp'] = (G.H + G.pin_fp) / G.m; G['peak_cg'] = G.mpr_cg / G.s
G['vn'] = G.vstored / G.n; G['pn'] = G.pinned / G.n
# bulk regime statistics: r0 = first bulk round; all rounds from r0 on bulk (a final round with no violators counts as bulk)
def bulk_all(row):
    b = str(row.bulk); r0 = row.r0
    if pd.isna(r0): return 0
    r0 = int(r0); tail = b[r0 - 1:]
    if pd.notna(row.exact): tail = tail[:-1]
    return int(all(ch == '1' for ch in tail))
G['ball'] = G.apply(bulk_all, axis=1)
G['J1'] = G.jumps.str.split('/').str[0].astype(float)
FAM = {'er': 'dense random', 'bottleneck': 'bottleneck', 'layered': 'unique-parent layers', 'thick': 'thick path'}
FAMS = ['er', 'bottleneck', 'layered', 'thick']
def Mfmt(m): return f'{m/1e6:.0f}M' if m >= 1e6 else f'{m/1e3:.0f}K'
def mm(x): return f'{int(x.min())}--{int(x.max())}' if x.min() != x.max() else f'{int(x.min())}'
def fresh(a, cap):
    capped = a.cert_ahs.isna()
    return (r'$>$%d' % cap if capped.all() else (f"{int(a.R_ahs[~capped].min())}--$>${cap}" if capped.any() else mm(a.R_ahs)))
def pk(x): return f'{x.max():.1f}'
def bfr(a):
    capped = a.ex_bf.isna() & (a.R_bf >= 400)
    return (r'$>$400' if capped.all() else (f"{int(a.R_bf[~capped].min())}--$>$400" if capped.any() else mm(a.R_bf)))

# ---- scaling table ----
s = G[G.sweep == 'scale']; hs = H[H.sweep == 'scale']
L = [r'{\setlength{\tabcolsep}{2.5pt}\footnotesize', r'\begin{tabular}{llrrrrrrrrrr}', r'\toprule',
     r'family & $n$ & $m$ & $h(G)$ & \PACE{} rounds & stored & $V/n$ & fixed+pin & CG & fresh & cBF & BF passes\\', r'\midrule']
for fam in FAMS:
    for n in sorted(s[s.fam == fam].n.unique()):
        gg = s[(s.fam == fam) & (s.n == n)]; hh = hs[(hs.type == fam) & (hs.n == n)].h
        L.append(f"{FAM[fam]} & {n:,} & {Mfmt(gg.m.mean())} & {mm(hh)} & {gg.R.mean():.1f} & {100*gg.stored.mean():.0f}\\% & {gg.vn.mean():.1f} & {gg.R_fp.mean():.1f} & {gg.R_cg.mean():.1f} & {fresh(gg, 400)} & {bfr(gg)} & {mm(hh+1)}\\\\")
L += [r'\bottomrule', r'\end{tabular}}']
open(OUT + 'tab_scale.tex', 'w').write('\n'.join(L) + '\n')

# ---- bulk regime table ----
def bulkrow(label, gg):
    jw = gg.jwin.max(); Lw = gg.certL
    return f"{label} & {FAM[gg.fam.iloc[0]]} & {len(gg)} & {int(gg.r0.max())} & {100*gg.ball.mean():.0f}\\% & {int(gg.J1.max())} & {int(jw)} & {int(Lw.min())}--{int(Lw.max())} & {mm(gg.R)}\\\\"
L = [r'{\setlength{\tabcolsep}{3pt}\footnotesize\begin{tabular}{llrrrrrrr}\toprule setting & family & runs & $\max r_0$ & bulk $\forall r\ge r_0$ & $\max J_1$ & $J^\star$ & $L$ & rounds\\\midrule']
for fam in FAMS: L.append(bulkrow(r'$s{=}4n$, $n\le10^6$', s[s.fam == fam]))
b = G[(G.sweep == 'budget')]
for fam in FAMS: L.append(bulkrow(r'$n{=}10^5$, $s\in[n,16n]$', b[b.fam == fam]))
sm = G[(G.sweep == 'small')]
for fam in FAMS: L.append(bulkrow(r'$n{=}10^5$, $s\in\{n/16,n/4\}$', sm[sm.fam == fam]))
dn = G[(G.sweep == 'dense')]
for fam in ['er', 'bottleneck', 'layered']: L.append(bulkrow(r'$d{=}1000$, $s{=}4n$', dn[dn.fam == fam]))
L.append(r'\bottomrule\end{tabular}}')
open(OUT + 'tab_bulk.tex', 'w').write('\n'.join(L) + '\n')

# ---- budget and small-budget tables ----
def budget_table(sub, cap, fn, label, caption):
    L = [r'\begin{table}[h]\centering{\setlength{\tabcolsep}{3pt}\footnotesize\begin{tabular}{lrrrrrrrr}\toprule family & $s/n$ & \PACE{} rounds & stored & exact & CG rounds & CG peak & fixed+pin & fresh\\\midrule']
    for fam in FAMS:
        for smu in sorted(sub[sub.fam == fam].s_mult.unique()):
            gg = sub[(sub.fam == fam) & (sub.s_mult == smu)]
            sm_s = f'1/{int(round(1/smu))}' if smu < 1 else f'{smu:g}'
            L.append(f"{FAM[fam]} & {sm_s} & {gg.R.mean():.1f} & {100*gg.stored.mean():.0f}\\% & {gg.exact.notna().mean():.2f} & {gg.R_cg.mean():.1f} & {pk(gg.peak_cg)} & {gg.R_fp.mean():.1f} & {fresh(gg, cap)}\\\\")
    L.append(r'\bottomrule\end{tabular}}\caption{' + caption + r'}\label{' + label + r'}\end{table}')
    open(OUT + fn, 'w').write('\n'.join(L) + '\n')
budget_table(b, 600, 'tab_budget.tex', 'tab:budget', r"""Budget sweep ($n=10^5$, $\eps=0.1$): mean rounds to certificate over 3 seeds. ``exact'' is the fraction of runs that stopped by the exactness certificate; ``CG peak'' is the largest number of edges constraint generation stored in one round, in units of $s$ (\PACE{} stores at most $1.5s$ per round by construction).""")
budget_table(sm, 600, 'tab_small.tex', 'tab:small', r'Small budgets ($n=10^5$, $\eps=0.1$, mean of 2 seeds). ``$>$600'': the fresh loop reached the round cap without certifying. Columns as in Table~\ref{tab:budget}.')

# ---- eps, weighted, dense, nospanner ----
def simple_table(sub, col, colname, fn, label, caption, cap=400, show_n=False):
    head = (r'family & $n$ & $m$ & ' if show_n else r'family & ' + colname + ' & ') + r'\PACE{} & stored & true ratio & CG & CG peak & fixed+pin & fresh' + (r' & cBF' if show_n else '') + r'\\\midrule'
    L = [r'\begin{table}[h]\centering{\setlength{\tabcolsep}{3pt}\footnotesize\begin{tabular}{' + ('llrrrrrrrrr' if show_n else 'llrrrrrrr') + r'}\toprule ' + head]
    for fam in FAMS:
        for v in sorted(sub[sub.fam == fam][col].unique()):
            gg = sub[(sub.fam == fam) & (sub[col] == v)]
            lead = f"{FAM[fam]} & {int(v):,} & {Mfmt(gg.m.mean())}" if show_n else f"{FAM[fam]} & {v:g}"
            L.append(f"{lead} & {gg.R.mean():.1f} & {100*gg.stored.mean():.0f}\\% & {gg.true.max():.3f} & {gg.R_cg.mean():.1f} & {pk(gg.peak_cg)} & {gg.R_fp.mean():.1f} & {fresh(gg, cap)}" + (f" & {bfr(gg)}" if show_n else "") + "\\\\")
    L.append(r'\bottomrule\end{tabular}}\caption{' + caption + r'}\label{' + label + r'}\end{table}')
    open(OUT + fn, 'w').write('\n'.join(L) + '\n')
e = G[G.sweep == 'eps'].copy(); e['eps'] = e.type.str.split('|').str[1].astype(float)
simple_table(e, 'eps', r'$\eps$', 'tab_eps.tex', 'tab:eps', r"""Rounds to certificate as a function of $\eps$ ($n=10^5$, $s=4n$, mean of 2 seeds). ``true ratio'' is the largest true ratio of a \PACE{} output over the seeds.""")
w = G[G.sweep == 'weighted'].copy(); w['wmax'] = w.type.str.split('|').str[2].astype(float)
simple_table(w, 'wmax', r'$w_{\max}$', 'tab_weighted.tex', 'tab:weighted', r'Weighted graphs ($n=10^5$, $s=4n$, $\eps=0.1$, mean of 2 seeds).')
simple_table(dn, 'n', '', 'tab_dense.tex', 'tab:dense', r'Denser graphs ($d=1000$, $s=4n$, $\eps=0.1$, mean of 2 seeds).', show_n=True)
ns = G[G.sweep == 'nospanner'].copy(); ns['v'] = 0
L = [r'\begin{table}[h]\centering{\setlength{\tabcolsep}{3pt}\footnotesize\begin{tabular}{lrrrrrrr}\toprule family & \PACE{} & stored & true ratio & CG & CG peak & fixed+pin & fresh\\\midrule']
for fam in FAMS:
    gg = ns[ns.fam == fam]
    L.append(f"{FAM[fam]} & {gg.R.mean():.1f} & {100*gg.stored.mean():.0f}\\% & {gg.true.max():.3f} & {gg.R_cg.mean():.1f} & {pk(gg.peak_cg)} & {gg.R_fp.mean():.1f} & {fresh(gg, 600)}\\\\")
L.append(r"""\bottomrule\end{tabular}}\caption{Spanner-free variant ($n=10^5$, $s=4n$, $\eps=0.1$, mean of 2 seeds); ``stored'' is $|P_r|/m$.}\label{tab:nosp}\end{table}""")
open(OUT + 'tab_nosp.tex', 'w').write('\n'.join(L) + '\n')

# ---- real graphs ----
NAME = {'soc-pokec-relationships.txt': 'Pokec', 'com-lj.ungraph.txt': 'LiveJournal', 'com-orkut.ungraph.txt': 'Orkut'}
rl = t[t.sweep.isin(['real', 'realsmall', 'realsrc']) & t.file.isin(NAME)].copy(); rl['s'] = rl.s_mult * rl.n
rc_ = g[g.sweep.isin(['real', 'realsmall', 'realsrc']) & g.file.isin(NAME)].copy(); rc_['s'] = rc_.s_mult * rc_.n; rc_['peak'] = rc_.maxpinround / rc_.s
old = o[o.sweep.isin(['real', 'realsmall']) & o.tag.isin(['ahs'])].copy(); old['source'] = 0
old = old[['file', 'source', 's_mult', 'tag', 'R_used', 'cert_round']].rename(columns={'R_used': 'R', 'cert_round': 'cert'})
old = pd.concat([old, RS[RS.tag == 'ahs'][['file', 'source', 's_mult', 'tag', 'R', 'cert']]], ignore_index=True)
L = [r'\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}\begin{tabular}{lrrrrrrrrrrrrr}\toprule graph & $n$ & $m$ & $|H|/m$ & $h(G)$ & $s/n$ & \PACE{} rounds & stored & $V/n$ & fixed+pin & CG & CG peak & fresh & cBF & BF\\\midrule'.replace('{lrrrrrrrrrrrrr}', '{lrrrrrrrrrrrrrr}')]
for fn in NAME:
    g0 = rl[rl.file == fn]; hh = H[H.file == fn].h; bfv = BF[BF.file == fn].R_bf
    for smu in sorted(g0.s_mult.unique()):
        gg = g0[g0.s_mult == smu]; oo_ = gg[gg.tag == 'ours']; ff = gg[gg.tag == 'fpt']; cc = rc_[(rc_.file == fn) & (rc_.s_mult == smu) & (rc_.tag == 'cg')]
        a = old[(old.file == fn) & (old.s_mult == smu)]
        sm_s = f'1/{int(round(1/smu))}' if smu < 1 else f'{smu:g}'
        capped = a.cert.isna()
        fr = (r'$>$400' if capped.all() else (f"{int(a.R[~capped].min())}--$>$400" if capped.any() else mm(a.R)))
        st = 100 * (oo_.H + oo_.pinned) / oo_.m
        def pr(p): return f"{p.min():.0f}\\%" + ('' if abs(p.min() - p.max()) < 0.5 else f"--{p.max():.0f}\\%")
        srcs = sorted(oo_.source.unique()); bfs = BF[(BF.file == fn) & BF.source.isin(srcs)].R_bf
        L.append(f"{NAME[fn]} & {int(oo_.n.iloc[0]):,} & {Mfmt(oo_.m.iloc[0])} & {100*oo_.H.iloc[0]/oo_.m.iloc[0]:.0f}\\% & {mm(hh)} & {sm_s} & {mm(oo_.R)} & {pr(st)} & {(oo_.vstored/oo_.n).max():.2f} & {mm(ff.R)} & {mm(cc.R)} & {cc.peak.max():.1f} & {fr} & {mm(bfs)} & {mm(hh+1)}\\\\")
L.append(r"\bottomrule\end{tabular}\caption{Real graphs ($\eps=0.1$). Budgets $s\le n$ use 3 source vertices (0, 1, and 2) and report the range over sources; budgets $s\ge2n$ use source 0. $h(G)$ is the hop diameter from the source, ``cBF'' is the round count of certified Bellman--Ford, and ``BF'' is the pass count $h(G)+1$ of plain streaming Bellman--Ford. ``stored'' is $(|H|+|P_R|)/m$ at the end and $V/n$ the stored violators per vertex; ``CG peak'' is the largest number of edges constraint generation stored in one round, in units of $s$. ``$>$400'': the fresh loop reached the round cap without certifying.}\label{tab:real}\end{table}")
open(OUT + 'tab_real.tex', 'w').write('\n'.join(L) + '\n')

# ---- figures ----
COL = {'ours': '#1f5fbf', 'fixed_pin': '#e08a1e', 'ahs': '#8c8c8c', 'cg': '#2a9d5c', 'bf': '#b03a7a'}
LAB = {'ours': 'PACE', 'fixed_pin': 'fixed step + pinning', 'ahs': 'fresh resampling', 'cg': 'constraint generation', 'bf': 'certified Bellman-Ford'}
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
                     'legend.fontsize': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
def kof(n): ln = np.log(n); return max(2, int(round(ln / np.log(ln))))
RC = {'ours': 'R', 'fixed_pin': 'R_fp', 'ahs': 'R_ahs', 'cg': 'R_cg', 'bf': 'R_bf'}
fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.0), sharey=True)
for ax, fam in zip(axes, FAMS):
    for tag in ['ahs', 'bf', 'fixed_pin', 'cg', 'ours']:
        gg = s[s.fam == fam].groupby('n')[RC[tag]].agg(['mean', 'min', 'max']).reset_index()
        ax.plot(gg.n, gg['mean'], 'o-', color=COL[tag], label=LAB[tag], ms=3.5, lw=1.4)
        ax.fill_between(gg.n, gg['min'], gg['max'], color=COL[tag], alpha=.18, lw=0)
    ns_ = np.array(sorted(s.n.unique())); ax.plot(ns_, [10 * kof(n) ** 2 / 0.1 for n in ns_], 'k--', lw=1.2, label=r'$10k^2/\varepsilon$')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('$n$'); ax.set_ylim(2, 5000)
    ax.text(0.04, 0.93, '(%s)' % 'abcd'[FAMS.index(fam)], transform=ax.transAxes, fontsize=11)
axes[0].set_ylabel('rounds')
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.12))
plt.tight_layout(rect=(0, 0.2, 1, 1)); plt.savefig(OUT + 'scale.pdf', bbox_inches='tight'); plt.close()

traj = json.load(open(R + 'traj_m.json')); trg = json.load(open(R + 'trajt.json'))
fig, axes = plt.subplots(2, 4, figsize=(8.6, 3.0))
keys = {'er': 'scale/20', 'bottleneck': 'scale/21', 'layered': 'scale/22', 'thick': 'scale/23'}
for j, fam in enumerate(FAMS):
    for tag in ['ahs', 'ours']:
        t = (trg if tag == 'ours' else traj)[f'{keys[fam]}/{tag}']; r = np.arange(1, len(t['nviol']) + 1)
        axes[0, j].plot(r, np.maximum(t['nviol'], 0.5), '-', color=COL[tag], label=LAB[tag], lw=1.4)
        axes[1, j].plot(r, t['ratio'], '-', color=COL[tag], lw=1.4)
    axes[0, j].set_yscale('log'); axes[0, j].text(0.80, 0.88, '(%s)' % 'abcd'[j], transform=axes[0, j].transAxes, fontsize=11)
    axes[1, j].set_xlabel('round $r$'); axes[1, j].axhline(1.1, color='k', ls=':', lw=1); axes[1, j].set_ylim(0.97, None)
    axes[0, j].tick_params(labelbottom=False)
axes[0, 0].set_ylabel('violating edges', fontsize=10); axes[1, 0].set_ylabel('true ratio', fontsize=10)
h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.savefig(OUT + 'traj.pdf'); plt.close()

bb = G[G.sweep.isin(['budget', 'small'])]
SC = {'ours': 'stored', 'fixed_pin': 'stored_fp', 'cg': 'stored_cg'}
fig, axes = plt.subplots(2, 4, figsize=(8.6, 2.9), sharey='row')
for j, fam in enumerate(FAMS):
    ax = axes[0, j]
    for tag in ['ahs', 'fixed_pin', 'cg', 'ours']:
        gg = bb[bb.fam == fam].groupby('s_mult')[RC[tag]].agg(['mean', 'min', 'max']).reset_index()
        ax.plot(gg.s_mult, gg['mean'], 'o-', color=COL[tag], label=LAB[tag], ms=3.5, lw=1.4)
        ax.fill_between(gg.s_mult, gg['min'], gg['max'], color=COL[tag], alpha=.18, lw=0)
    ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_ylim(2, 1000); ax.tick_params(labelbottom=False)
    ax.text(0.04, 0.90, '(%s)' % 'abcd'[j], transform=ax.transAxes, fontsize=11)
    ax = axes[1, j]
    for tag in ['fixed_pin', 'cg', 'ours']:
        gg = bb[bb.fam == fam].groupby('s_mult')[SC[tag]].agg(['mean', 'min', 'max']).reset_index()
        ax.plot(gg.s_mult, 100 * gg['mean'], 'o-', color=COL[tag], ms=3.5, lw=1.4)
        ax.fill_between(gg.s_mult, 100 * gg['min'], 100 * gg['max'], color=COL[tag], alpha=.18, lw=0)
    ax.set_xscale('log', base=2); ax.set_xlabel('$s/n$'); ax.set_ylim(0, 100)
    ax.text(0.04, 0.90, '(%s)' % 'efgh'[j], transform=ax.transAxes, fontsize=11)
axes[0, 0].set_ylabel('rounds'); axes[1, 0].set_ylabel('stored (% of $m$)')
h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.03))
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig(OUT + 'budget.pdf', bbox_inches='tight'); plt.close()

# ---- numbers for the text ----
print('ALL PACE runs', len(G), 'exact', G.exact.notna().sum(), 'true max', G.true.max(), 'true max eps=0.1', G[G.sweep != 'eps'].true.max())
print('ours<=fp', (G.R <= G.R_fp).mean(), 'ours<fp', (G.R < G.R_fp).mean(), 'ours<=cg', (G.R <= G.R_cg).mean(), 'ours<ahs', (G.R < G.R_ahs).mean())
print('scale: PACE', s.R.min(), s.R.max(), 'fat', s[s.fam.isin(['er','bottleneck'])].R.min(), s[s.fam.isin(['er','bottleneck'])].R.max(), 'thin', s[s.fam.isin(['layered','thick'])].R.min(), s[s.fam.isin(['layered','thick'])].R.max())
print('scale: stored', s.stored.min(), s.stored.max(), 'fp', s.R_fp.min(), s.R_fp.max(), 'ahs', s.R_ahs.min(), s.R_ahs.max(), 'cg', s.R_cg.min(), s.R_cg.max(), 'cg peak', s.peak_cg.min(), s.peak_cg.max())
for fam in FAMS:
    x = s[s.fam == fam]; print(fam, 'PACE', x.R.min(), x.R.max(), 'passes', (2*x.R+1).min(), (2*x.R+1).max(), 'cg', x.R_cg.min(), x.R_cg.max(), 'cgpeak', x.peak_cg.max(), 'fp', x.R_fp.min(), x.R_fp.max(), 'ahs', x.R_ahs.min(), x.R_ahs.max(), 'exact', x.exact.notna().mean(), 'true', x.true.max(), 'r0', x.r0.max(), 'J1', x.J1.max(), 'Fmax', x.Fmax.max(), 'viol1/n', (x.viol1/x.n).min(), (x.viol1/x.n).max())
    print('   mean rounds ratio fp/PACE', (x.R_fp.mean()/x.R.mean()), 'saved', 1 - x.R.mean()/x.R_fp.mean(), 'max saved', (1-x.R/x.R_fp).max(), 'stored', x.stored.min(), x.stored.max(), 'stored cg', x.stored_cg.min(), x.stored_cg.max())
print('budget: by s_mult')
for smu in sorted(b.s_mult.unique()):
    x = b[b.s_mult == smu]
    for fam in FAMS:
        y = x[x.fam == fam]; print(' s/n', smu, fam, 'PACE', y.R.min(), y.R.max(), 'stored', round(y.stored.min(),3), round(y.stored.max(),3), 'cg', y.R_cg.min(), y.R_cg.max(), 'cgpeak', round(y.peak_cg.max(),2), 'fp', y.R_fp.min(), y.R_fp.max(), 'ahs', y.R_ahs.min(), y.R_ahs.max(), 'r0', y.r0.max(), 'Fmax', y.Fmax.max(), 'exact', y.exact.notna().mean(), 'true', round(y.true.max(),4))
print('small:')
for smu in sorted(sm.s_mult.unique()):
    for fam in FAMS:
        y = sm[(sm.s_mult == smu) & (sm.fam == fam)]; print(' s/n', smu, fam, 'PACE', y.R.min(), y.R.max(), 'stored', round(y.stored.min(),3), round(y.stored.max(),3), 'cg', y.R_cg.min(), y.R_cg.max(), 'cgpeak', round(y.peak_cg.max(),1), 'cg stored', round(y.stored_cg.max(),3), 'fp', y.R_fp.min(), y.R_fp.max(), 'ahs', y.R_ahs.min(), y.R_ahs.max(), 'ahs capped', y.cert_ahs.isna().sum(), 'r0', y.r0.max(), 'Fmax', y.Fmax.max(), 'exact', y.exact.notna().mean(), 'true', round(y.true.max(),4), 'L', y.certL.min(), y.certL.max(), 'jwin', y.jwin.max())
print('eps:'); print(e.groupby(['fam','eps']).agg(R=('R','mean'), fp=('R_fp','mean'), cg=('R_cg','mean'), true=('true','max'), jwin=('jwin','max'), L=('certL','max'), exact=('exact', lambda x: x.notna().mean())).to_string())
print('dense:'); print(dn.groupby(['fam','n']).agg(R=('R','mean'), Rmin=('R','min'), Rmax=('R','max'), st=('stored','mean'), cg=('R_cg','mean'), cgpeak=('peak_cg','max'), fp=('R_fp','mean'), ahsmin=('R_ahs','min'), ahsmax=('R_ahs','max'), capped=('cert_ahs', lambda x: x.isna().sum())).to_string())
print('nosp:'); print(ns.groupby('fam').agg(R=('R','mean'), Rmin=('R','min'), Rmax=('R','max'), st=('stored','mean'), cg=('R_cg','mean'), cgpeak=('peak_cg','max'), fp=('R_fp','mean'), ahsmin=('R_ahs','min'), ahsmax=('R_ahs','max'), r0=('r0','max'), true=('true','max')).to_string())
print('weighted:'); print(w.groupby(['fam','wmax']).agg(R=('R','mean'), cg=('R_cg','mean'), fp=('R_fp','mean'), true=('true','max')).to_string())
# real numbers
ro = rl[rl.tag == 'ours']; rc = rc_[rc_.tag == 'cg']; rf = rl[rl.tag == 'fpt']
print('real ours: runs', len(ro), 'exact', ro.exact.notna().sum(), 'R', ro.R.min(), ro.R.max(), 'R at s<=n', ro[ro.s_mult<=1].R.min(), ro[ro.s_mult<=1].R.max(), 'stored% s<=n', (100*(ro.H+ro.pinned)/ro.m)[ro.s_mult<=1].min(), (100*(ro.H+ro.pinned)/ro.m)[ro.s_mult<=1].max(), 'V/n max', (ro.vstored/ro.n).max(), 'pinned/n max', (ro.pinned/ro.n).max(), 'r0', ro.r0.max(), 'Fmax', ro.Fmax.max())
print('real cg: R', rc.R.min(), rc.R.max(), 'peak', rc.peak.min(), rc.peak.max(), 'peak at n/16', rc[rc.s_mult==0.0625].peak.min(), rc[rc.s_mult==0.0625].peak.max(), 'at n/4', rc[rc.s_mult==0.25].peak.max())
mo = ro.merge(rf[['file','source','s_mult','R']].rename(columns={'R':'R_fp'}), on=['file','source','s_mult']).merge(old[old.tag=='ahs'][['file','source','s_mult','R']].rename(columns={'R':'R_ahs'}), on=['file','source','s_mult']).merge(rc[['file','source','s_mult','R']].rename(columns={'R':'R_cg'}), on=['file','source','s_mult']).merge(BF[['file','source','R_bf']], on=['file','source'])
print('real n', len(mo), 'ours<=fp', (mo.R<=mo.R_fp).sum(), 'ours<fp', (mo.R<mo.R_fp).sum(), 'ours<ahs', (mo.R<mo.R_ahs).sum(), 'ours<=cg', (mo.R<=mo.R_cg).sum(), 'ours>cg', (mo.R>mo.R_cg).sum(), 'max diff ours-cg', (mo.R-mo.R_cg).max(), 'ours<=bf', (mo.R<=mo.R_bf).sum(), 'ours<bf', (mo.R<mo.R_bf).sum(), 'bf range', mo.R_bf.min(), mo.R_bf.max())
x = mo[mo.s_mult == 0.0625]; print('n/16: ours', x.R.min(), x.R.max(), 'fp', x.R_fp.min(), x.R_fp.max(), 'ahs', x.R_ahs.min(), x.R_ahs.max(), 'cg', x.R_cg.min(), x.R_cg.max(), 'bf', x.R_bf.min(), x.R_bf.max())
x = mo[mo.s_mult <= 1]; print('s<=n: ours', x.R.min(), x.R.max(), 'fp', x.R_fp.min(), x.R_fp.max(), 'ahs', x.R_ahs.min(), x.R_ahs.max(), 'cg', x.R_cg.min(), x.R_cg.max())
print('source spread', ro[ro.s_mult<=1].groupby(['file','s_mult']).R.agg(lambda v: v.max()-v.min()).max())
print('H/m real', (ro.H/ro.m).groupby(ro.file).first().to_dict(), 'm/n', (ro.m/ro.n).groupby(ro.file).first().to_dict())
print('fixed mult range', (1+s.n**(1/s.k)).min(), (1+s.n**(1/s.k)).max())
# bf on scale
for fam in FAMS:
    x = s[s.fam == fam]; print('cBF', fam, 'R', x.R_bf.min(), x.R_bf.max(), 'exact', x.ex_bf.notna().mean(), 'true', x.true_bf.max(), 'PACE passes', (2*x.R+1).min(), (2*x.R+1).max(), 'cBF passes', (x.R_bf+1).min(), (x.R_bf+1).max())
x = s[s.n >= 1000000]; print('n=1e6 thin: PACE passes', (2*x[x.fam.isin(['layered','thick'])].R+1).min(), (2*x[x.fam.isin(['layered','thick'])].R+1).max(), 'fresh passes', (2*x[x.fam.isin(['layered','thick'])].R_ahs+1).min(), (2*x[x.fam.isin(['layered','thick'])].R_ahs+1).max())
print('pinned/n max all', G.pn.max(), 'V/n max all', G.vn.max(), 'stored max all synthetic', G[~G.sweep.str.startswith('real')].stored.max())
print('done')
