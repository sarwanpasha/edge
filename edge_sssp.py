"""
edge_sssp.py -- simulation of semi-streaming (1+eps)-SSSP MWU algorithms.

Implements
  * AHS baseline   : fixed multiplier mu = 1 + n^{1/k}, fresh sample each round, output = SPT(union of trees)
  * ADAPTIVE (ours): multiplier mu_r = 1 + c / rho_r  (rho_r = flagged-weight fraction), cumulative pinned
                     sample, exactness certificate, anytime suffix-window (1+eps) certificate.
All "passes" are simulated offline on in-memory edge arrays; the *stored* structure size
(|H| + |pinned| + trees) is what a streaming implementation would keep, and is reported.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, breadth_first_order
import time, json, sys, os

# ----------------------------------------------------------------------------- graphs

def _dedup(u, v, n):
    """remove self loops / duplicate undirected edges."""
    mask = u != v
    u, v = u[mask], v[mask]
    a = np.minimum(u, v); b = np.maximum(u, v)
    key = a.astype(np.int64) * n + b
    key = np.unique(key)
    return (key // n).astype(np.int64), (key % n).astype(np.int64)

def er_graph(n, d, rng, unit=True, wmax=1.0):
    """G(n, d/n): expected degree d. weights unit or uniform in [1, wmax]."""
    m_target = int(n * d / 2)
    u = rng.integers(0, n, size=int(m_target * 1.02) + 10)
    v = rng.integers(0, n, size=u.size)
    u, v = _dedup(u, v, n)
    if u.size > m_target:
        idx = rng.choice(u.size, size=m_target, replace=False)
        u, v = u[idx], v[idx]
    w = np.ones(u.size) if unit else rng.uniform(1.0, wmax, size=u.size)
    return n, u, v, w

def thick_path(n_blocks, L, back_deg, rng, unit=True, wmax=1.0, intra='clique'):
    """chain of n_blocks blocks of size L; each vertex in block j+1 has `back_deg` random
    edges to block j; intra-block edges: clique or ER with degree back_deg*4.
    source = vertex 0 (block 0 = {0})."""
    n = 1 + n_blocks * L
    us, vs = [], []
    # block index: block 0 = {0}; block j (>=1) = 1+(j-1)L ... jL
    def block(j):
        if j == 0: return np.array([0])
        return np.arange(1 + (j - 1) * L, 1 + j * L)
    for j in range(1, n_blocks + 1):
        cur = block(j); prev = block(j - 1)
        bd = min(back_deg, prev.size)
        for _ in range(bd):
            us.append(cur); vs.append(rng.choice(prev, size=cur.size, replace=True))
        if intra == 'clique':
            iu, iv = np.triu_indices(L, 1)
            us.append(cur[iu]); vs.append(cur[iv])
        else:
            deg = min(4 * back_deg, L - 1)
            cnt = L * deg // 2
            us.append(rng.choice(cur, size=cnt)); vs.append(rng.choice(cur, size=cnt))
    u = np.concatenate(us); v = np.concatenate(vs)
    u, v = _dedup(u, v, n)
    w = np.ones(u.size) if unit else rng.uniform(1.0, wmax, size=u.size)
    return n, u, v, w

def bottleneck_cloud(path_len, n_cloud, d, rng, n_attach=1, unit=True, wmax=1.0):
    """source -- path of length path_len -- attach vertex(es) -- ER cloud G(n_cloud, d/n_cloud).
    This is the 'semi-random family': a thin hidden part plus a dense random cloud."""
    n = path_len + 1 + n_cloud
    pu = np.arange(0, path_len); pv = np.arange(1, path_len + 1)
    cloud0 = path_len + 1
    _, cu, cv, _ = er_graph(n_cloud, d, rng)
    cu = cu + cloud0; cv = cv + cloud0
    att = rng.choice(n_cloud, size=n_attach, replace=False) + cloud0
    au = np.full(n_attach, path_len); av = att
    u = np.concatenate([pu, au, cu]); v = np.concatenate([pv, av, cv])
    u, v = _dedup(u, v, n)
    w = np.ones(u.size) if unit else rng.uniform(1.0, wmax, size=u.size)
    return n, u, v, w

def layered_planted(n_layers, width, back_deg_thin, decoy_deg, rng, unit=True):
    """layered graph: layers of `width` vertices; each vertex in layer j+1 has back_deg_thin
    edges to layer j (thin true-parent structure) plus decoy_deg intra-layer edges
    (dense, off shortest paths). source = 0 connected to all of layer 1."""
    n = 1 + n_layers * width
    us, vs = [], []
    def layer(j):
        if j == 0: return np.array([0])
        return np.arange(1 + (j - 1) * width, 1 + j * width)
    for j in range(1, n_layers + 1):
        cur = layer(j); prev = layer(j - 1)
        bd = 1 if j == 1 else back_deg_thin
        for _ in range(bd):
            us.append(cur); vs.append(rng.choice(prev, size=cur.size))
        cnt = width * decoy_deg // 2
        us.append(rng.choice(cur, size=cnt)); vs.append(rng.choice(cur, size=cnt))
    u = np.concatenate(us); v = np.concatenate(vs)
    u, v = _dedup(u, v, n)
    w = np.ones(u.size)
    return n, u, v, w

def load_edgelist(path, unit=True):
    """SNAP-style edge list (# comments). returns largest-component-agnostic graph."""
    arr = np.loadtxt(path, dtype=np.int64, comments='#')
    u, v = arr[:, 0], arr[:, 1]
    ids, inv = np.unique(np.concatenate([u, v]), return_inverse=True)
    n = ids.size
    u = inv[:u.size]; v = inv[u.size:]
    u, v = _dedup(u, v, n)
    w = np.ones(u.size) if unit else arr[:, 2].astype(float)
    return n, u, v, w

# ----------------------------------------------------------------------------- SSSP

def sssp(n, u, v, w, src, unit=False):
    if u.size == 0:
        d = np.full(n, np.inf); d[src] = 0.0; return d
    A = csr_matrix((np.concatenate([w, w]), (np.concatenate([u, v]), np.concatenate([v, u]))), shape=(n, n))
    if unit:
        # BFS distances via dijkstra with unit weights is fine and exact; use unweighted flag
        d = dijkstra(A, directed=False, indices=src, unweighted=True)
    else:
        d = dijkstra(A, directed=False, indices=src)
    return d

# ----------------------------------------------------------------------------- Baswana-Sen spanner

def baswana_sen_spanner(n, u, v, w, k, rng, p=None):
    """Offline Baswana-Sen (2k-1)-spanner (weighted). Returns boolean mask over edges.
    Phases 1..k-1: cluster sampling / joining; phase k: lightest edge to each adjacent cluster."""
    m = u.size
    keep = np.zeros(m, dtype=bool)
    E = np.arange(m)                 # remaining (unhandled) edges
    cl = np.arange(n)                # cluster id per vertex, -1 = removed
    p = n ** (-1.0 / k) if p is None else p
    for phase in range(1, k):
        if E.size == 0: break
        eu, ev, ew = u[E], v[E], w[E]
        act = (cl[eu] >= 0) & (cl[ev] >= 0) & (cl[eu] != cl[ev])
        E = E[act]; eu, ev, ew = eu[act], ev[act], ew[act]
        if E.size == 0: break
        oldcl = cl.copy()
        cids = np.unique(cl[cl >= 0])
        samp = np.zeros(n, dtype=bool)
        samp[cids[rng.random(cids.size) < p]] = True
        vsamp = (cl >= 0) & samp[cl.clip(0)]
        # directed view
        X = np.concatenate([eu, ev]); Y = np.concatenate([ev, eu]); W2 = np.concatenate([ew, ew]); EI = np.concatenate([E, E])
        act2 = ~vsamp[X]
        X, Y, W2, EI = X[act2], Y[act2], W2[act2], EI[act2]
        CY = oldcl[Y]
        order = np.lexsort((EI, W2, CY, X))
        X, W2, EI, CY = X[order], W2[order], EI[order], CY[order]
        first = np.ones(X.size, dtype=bool)
        first[1:] = (X[1:] != X[:-1]) | (CY[1:] != CY[:-1])
        Xl, Wl, EIl, CYl = X[first], W2[first], EI[first], CY[first]      # lightest edge per (vertex, adjacent cluster)
        sampC = samp[CYl]
        # per vertex: lightest edge into a sampled cluster
        order2 = np.lexsort((EIl, Wl, ~sampC, Xl))
        Xs, Ws, EIs, CYs, Ss = Xl[order2], Wl[order2], EIl[order2], CYl[order2], sampC[order2]
        firstv = np.ones(Xs.size, dtype=bool); firstv[1:] = Xs[1:] != Xs[:-1]
        vx = Xs[firstv]; join_ok = Ss[firstv]; join_w = Ws[firstv]; join_e = EIs[firstv]; join_c = CYs[firstv]
        has = np.zeros(n, dtype=bool); jw = np.full(n, np.inf); jc = np.full(n, -1)
        has[vx[join_ok]] = True; jw[vx[join_ok]] = join_w[join_ok]; jc[vx[join_ok]] = join_c[join_ok]
        keep[join_e[join_ok]] = True
        addmask = (~has[Xl]) | (Wl < jw[Xl])
        keep[EIl[addmask]] = True
        # covered (vertex, old-cluster) pairs
        cov = np.concatenate([Xl[addmask].astype(np.int64) * n + CYl[addmask],
                              vx[join_ok].astype(np.int64) * n + join_c[join_ok]])
        cov = np.unique(cov)
        # update clustering
        newcl = np.where(vsamp, cl, np.where(has, jc, -1))
        newcl[cl < 0] = -1
        cl = newcl
        # prune edge set
        eu, ev = u[E], v[E]
        k1 = eu.astype(np.int64) * n + oldcl[ev]
        k2 = ev.astype(np.int64) * n + oldcl[eu]
        covered = np.isin(k1, cov) | np.isin(k2, cov)
        alive = (cl[eu] >= 0) & (cl[ev] >= 0) & (cl[eu] != cl[ev]) & ~covered
        E = E[alive]
    if E.size > 0:
        eu, ev, ew = u[E], v[E], w[E]
        act = (cl[eu] >= 0) & (cl[ev] >= 0) & (cl[eu] != cl[ev])
        E = E[act]; eu, ev, ew = eu[act], ev[act], ew[act]
        if E.size > 0:
            X = np.concatenate([eu, ev]); Y = np.concatenate([ev, eu]); W2 = np.concatenate([ew, ew]); EI = np.concatenate([E, E])
            CY = cl[Y]
            order = np.lexsort((EI, W2, CY, X))
            X, CY, EI = X[order], CY[order], EI[order]
            first = np.ones(X.size, dtype=bool); first[1:] = (X[1:] != X[:-1]) | (CY[1:] != CY[:-1])
            keep[EI[first]] = True
    return keep

# ----------------------------------------------------------------------------- MWU driver

WINDOWS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]

class JumpHistory:
    """sparse per-round storage of positive relative violations; window certificates."""
    def __init__(self, m):
        self.m = m; self.idx = []; self.val = []
    def add(self, idx, val):
        self.idx.append(idx.astype(np.int64)); self.val.append(val.astype(np.float32))
    def window_cert(self, L):
        """max_e mean over last L rounds of jump_r(e)."""
        R = len(self.idx); L = min(L, R)
        if L == 1:
            v = self.val[R - 1]
            return float(v.max() / 1) if v.size else 0.0
        idx = np.concatenate(self.idx[R - L:R]); val = np.concatenate(self.val[R - L:R]).astype(np.float64)
        if idx.size == 0: return 0.0
        if np.isinf(val).any(): return float('inf')
        acc = np.bincount(idx, weights=val, minlength=self.m)
        return float(acc.max() / L)
    def best_cert(self, windows=WINDOWS):
        R = len(self.idx)
        best = (np.inf, 0)
        for L in windows:
            if L > R: break
            c = self.window_cert(L)
            if c < best[0]: best = (c, L)
        if R not in windows:
            c = self.window_cert(R)
            if c < best[0]: best = (c, R)
        return best

def run_mwu(n, u, v, w, src, k, eps, s_budget, mode='adaptive', pin=True, R_max=None,
            c=None, rng=None, spanner_mask=None, unit=True, stop_on_cert=True,
            verbose=False, tag='', d_true=None, cert_every=1, treepin=False):
    """
    mode: 'adaptive' -> mu_r = 1 + c/rho_r ; 'harmonic' -> mu_r = 1 + max(1/r, c)/rho_r ; 'floor' -> max(harmonic, 1+n^{1/k}) ; 'pin2' -> store violators directly when they fit in the round budget s, else floor rule ; 'cg' -> store all violators, uniform fresh samples ; 'fixed' -> mu = 1 + n^{1/k}
    pin : cumulative sampling (pinned)      ; False -> fresh sample each round (AHS)
    """
    rng = rng or np.random.default_rng(0)
    m = u.size
    if c is None: c = eps / (10.0 * k)
    if R_max is None: R_max = int(np.ceil(10 * k * k / eps))
    mu_fixed = 1.0 + n ** (1.0 / k)
    t0 = time.time()
    if d_true is None: d_true = sssp(n, u, v, w, src, unit=unit)
    reach = np.isfinite(d_true) & (np.arange(n) != src)
    H = spanner_mask if spanner_mask is not None else np.zeros(m, dtype=bool)
    P = np.zeros(m, dtype=bool)
    logq = np.zeros(m); logQ = np.log(m)
    fails = np.zeros(m, dtype=np.int32)
    union_tree = np.zeros(m, dtype=bool)
    log_s = np.log(s_budget); log_s_eff = log_s
    recs = []; JH = JumpHistory(m); nvstored = 0
    R_used = 0; exact_round = None; cert_round = None
    for r in range(1, R_max + 1):
        if mode == 'bfcert':
            # spanner-initialized streaming Bellman-Ford with the window certificate: one relaxation of all edges per round, nothing stored
            F = np.zeros(m, dtype=bool); stored = H
            if r == 1:
                idx = np.flatnonzero(H); d = sssp(n, u[idx], v[idx], w[idx], src, unit=unit)
            else:
                d = d_prev.copy(); cand = np.concatenate([d_prev[u] + w, d_prev[v] + w]); tgt = np.concatenate([v, u])
                np.minimum.at(d, tgt, cand); d[src] = 0.0
            d_prev = d
        else:
            p = np.exp(np.minimum(0.0, log_s_eff + logq - logQ))
            F = rng.random(m) < p
            if pin and not treepin:
                P |= F; stored = H | P
            elif pin:
                stored = H | P | F   # tree pinning: the fresh sample is used this round only
            else:
                stored = H | F
            idx = np.flatnonzero(stored)
            d = sssp(n, u[idx], v[idx], w[idx], src, unit=unit)
            if pin and treepin:
                # keep one tight parent edge per vertex (a shortest-path tree of the stored graph)
                du_, dv_ = d[u[idx]], d[v[idx]]
                tu = np.isfinite(du_) & (np.abs(du_ + w[idx] - dv_) <= 1e-9 * np.maximum(1.0, np.abs(dv_)))
                tv = np.isfinite(dv_) & (np.abs(dv_ + w[idx] - du_) <= 1e-9 * np.maximum(1.0, np.abs(du_)))
                child = np.concatenate([v[idx][tu], u[idx][tv]]); eid = np.concatenate([idx[tu], idx[tv]])
                _, first = np.unique(child, return_index=True)
                P[eid[first]] = True
        if not pin:
            du, dv = d[u[idx]], d[v[idx]]
            tight = np.isfinite(du) & np.isfinite(dv) & (np.abs(np.abs(du - dv) - w[idx]) < 1e-9)
            union_tree[idx[tight]] = True
        du, dv = d[u], d[v]
        fin_u, fin_v = np.isfinite(du), np.isfinite(dv)
        both = fin_u & fin_v
        diff = np.abs(np.where(both, du - dv, 0.0))
        viol = (both & (diff > w * (1 + 1e-9))) | (fin_u ^ fin_v)
        vidx = np.flatnonzero(viol)
        nviol = vidx.size
        if nviol == 0:
            W = 0.0; rho = 0.0
        else:
            lw = logq[vidx]; mx = lw.max()
            W = np.exp(mx) * np.exp(lw - mx).sum(); rho = W / np.exp(logQ)
        jv = np.where(both[vidx], np.maximum(0.0, diff[vidx] - w[vidx]) / w[vidx], np.inf)
        JH.add(vidx, jv)
        fails[vidx] += 1
        Fmax = int(fails.max())
        ok = reach & np.isfinite(d)
        unreached = int((reach & ~np.isfinite(d)).sum())
        ratio_true = float(np.max(d[ok] / d_true[ok])) if ok.any() else np.inf
        if unreached > 0: ratio_true = np.inf
        nwrong = int((ok & (d > d_true * (1 + 1e-9))).sum()) + unreached
        dpin = False; log_s_eff = log_s
        if nviol == 0:
            mu = 1.0; exact_round = exact_round or r
        elif mode == 'pin2' and nviol <= s_budget / 2:
            nvstored += nviol
            # bulk round: store the violators now, keep the weights, spend the rest of the budget on fresh samples
            P[vidx] = True; dpin = True; mu = 1.0; log_s_eff = np.log(max(1.0, s_budget - nviol))
        elif mode == 'pin2':
            mu = max(1.0 + max(1.0 / r, c) / rho, mu_fixed)
        elif mode == 'cg':
            P[vidx] = True; dpin = True; mu = 1.0; nvstored += nviol
        elif mode == 'bfcert':
            mu = 1.0
        elif mode == 'adaptive':
            mu = 1.0 + c / rho
        elif mode == 'harmonic':
            mu = 1.0 + max(1.0 / r, c) / rho
        elif mode == 'floor':
            mu = max(1.0 + max(1.0 / r, c) / rho, mu_fixed)
        elif mode == 'square':
            mu = 1.0 + max(1.0 / (r * r), c) / rho
        elif mode == 'const':
            mu = 1.0 + 1.0 / rho
        else:
            mu = mu_fixed
        if nviol > 0:
            logq[vidx] += np.log(mu)
            logQ = np.log(np.exp(logQ) + (mu - 1.0) * W)
        rec = dict(r=r, nviol=nviol, rho=rho, mu=mu, Fmax=Fmax, ratio_true=ratio_true, nwrong=nwrong, dpin=dpin,
                   stored=int(stored.sum()), pinned=int(P.sum()), sampled=int(F.sum()), vstored=int(nvstored),
                   wall=time.time() - t0, sumjump=float(jv[np.isfinite(jv)].sum()),
                   maxjump=float(jv[np.isfinite(jv)].max()) if np.isfinite(jv).any() and jv.size else 0.0)
        if r % cert_every == 0 or nviol == 0:
            cert, Lbest = JH.best_cert()
            rec['cert'] = cert; rec['cert_L'] = Lbest
            if cert <= eps and cert_round is None: cert_round = r
        recs.append(rec); R_used = r
        if verbose:
            print(f"[{tag}] r={r} viol={nviol} rho={rho:.3e} mu={mu:.3g} Fmax={Fmax} true={ratio_true:.4f} wrong={nwrong} stored={rec['stored']} cert={rec.get('cert', float('nan')):.4f} t={rec['wall']:.0f}s", flush=True)
        if stop_on_cert and (nviol == 0 or cert_round is not None):
            break
    cert, Lbest = JH.best_cert()
    out = dict(n=n, m=m, k=k, eps=eps, s=s_budget, mode=mode, pin=pin, c=c, R_used=R_used,
               exact_round=exact_round, cert_round=cert_round, H_size=int(H.sum()), recs=recs,
               final_true_ratio=recs[-1]['ratio_true'], final_cert=cert, final_cert_L=Lbest,
               wall=time.time() - t0, tag=tag)
    if not pin:
        idx = np.flatnonzero(union_tree | H)
        d = sssp(n, u[idx], v[idx], w[idx], src, unit=unit)
        ok = reach & np.isfinite(d)
        out['ahs_union_ratio'] = float(np.max(d[ok] / d_true[ok])) if ok.any() and not (reach & ~np.isfinite(d)).any() else np.inf
    return out

# ----------------------------------------------------------------------------- CLI

def make_graph(spec, rng):
    t = spec['type']
    if t == 'er':
        return er_graph(spec['n'], spec['d'], rng, unit=spec.get('unit', True), wmax=spec.get('wmax', 1.0))
    if t == 'thick':
        return thick_path(spec['blocks'], spec['L'], spec['back'], rng, unit=spec.get('unit', True), intra=spec.get('intra', 'clique'))
    if t == 'bottleneck':
        return bottleneck_cloud(spec['path'], spec['ncloud'], spec['d'], rng, n_attach=spec.get('attach', 1), unit=spec.get('unit', True))
    if t == 'layered':
        return layered_planted(spec['layers'], spec['width'], spec['back'], spec['decoy'], rng)
    if t == 'file':
        return load_edgelist(spec['path'], unit=spec.get('unit', True))
    raise ValueError(t)

def default_k(n):
    ln = np.log(n)
    return max(2, int(round(ln / np.log(ln))))

def build_spanner(spec, n, u, v, w, k, seed):
    sp = spec.get('spanner', 'bs')
    if sp == 'none':
        return np.zeros(u.size, dtype=bool)
    kH = spec.get('kH', k); pH = spec.get('pH')
    return baswana_sen_spanner(n, u, v, w, kH, np.random.default_rng(seed + 1), p=pH)

if __name__ == '__main__':
    spec = json.loads(sys.argv[1])
    out_path = sys.argv[2]
    seed = spec.get('seed', 0)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    n, u, v, w = make_graph(spec, rng)
    k = spec.get('k') or default_k(n)
    eps = spec.get('eps', 0.1)
    s_budget = spec.get('s_mult', 8.0) * n
    unit = spec.get('unit', True)
    print(f"graph n={n} m={u.size} k={k} eps={eps} s={s_budget:.0f} built in {time.time()-t0:.1f}s", flush=True)
    H = build_spanner(spec, n, u, v, w, k, seed)
    print(f"spanner |H|={int(H.sum())} ({H.sum()/n:.1f} n)", flush=True)
    src = spec.get('source', 0)
    d_true = sssp(n, u, v, w, src, unit=unit)
    results = {}
    for cfg in spec.get('configs', [dict(mode='adaptive', pin=True), dict(mode='fixed', pin=False)]):
        tag = cfg.get('tag') or f"{cfg['mode']}{'_pin' if cfg.get('pin', True) else ''}"
        res = run_mwu(n, u, v, w, src, k, eps, s_budget, mode=cfg['mode'], pin=cfg.get('pin', True),
                      R_max=cfg.get('R_max', spec.get('R_max')), c=cfg.get('c'), rng=np.random.default_rng(seed + 2),
                      spanner_mask=H, unit=unit, verbose=spec.get('verbose', False), tag=tag,
                      stop_on_cert=cfg.get('stop_on_cert', True), d_true=d_true, cert_every=spec.get('cert_every', 1), treepin=cfg.get('treepin', False))
        results[tag] = res
        print(f"== {tag}: R_used={res['R_used']} exact_round={res['exact_round']} cert_round={res['cert_round']} final_cert={res['final_cert']:.4f} (L={res['final_cert_L']}) true={res['final_true_ratio']:.4f} wall={res['wall']:.1f}s", flush=True)
        with open(out_path, 'w') as f:
            json.dump(dict(results, spec=spec, k=k, n=n, m=int(u.size), H_size=int(H.sum())), f)
