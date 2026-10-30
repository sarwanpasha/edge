import json, os, glob, sys
H = 'sweep,job,type,seed,source,s_mult,n,m,k,H,tag,R,exact,cert,true,certL,viol1,Fmax,pinned,vstored,maxstored,r0,jwin,jumps,bulk'
rows = []; traj = {}
for d in sorted(glob.glob('../results/*_t') + glob.glob('../results/*_bf')):
    sw = os.path.basename(d).rsplit('_', 1)[0]
    for fn in sorted(glob.glob(d + '/*.json'), key=lambda p: int(os.path.basename(p)[:-5])):
        j = int(os.path.basename(fn)[:-5]); D = json.load(open(fn)); sp = D['spec']
        p = sp.get('path'); p = os.path.basename(str(p)) if sp['type'] == 'file' else p
        ty = f"{sp['type']}|{sp.get('eps',0.1)}|{sp.get('wmax')}|{sp.get('d')}|{p if p is not None else ''}|{sp.get('spanner','')}"
        for tag in ('ours', 'fpt', 'bfcert'):
            if tag not in D: continue
            x = D[tag]; rc = x['recs']; R = x['R_used']; L = x['final_cert_L'] or 1
            mj = [r.get('maxjump', 0.0) for r in rc]; bk = ''.join('1' if r.get('dpin') else '0' for r in rc)
            r0 = bk.find('1') + 1 if '1' in bk else ''
            jwin = max(mj[max(0, R - L):R]) if R else 0
            rows.append([sw, j, ty, sp.get('seed', 0), sp.get('source', 0), sp.get('s_mult'), D['n'], D['m'], D['k'], D['H_size'], tag, R,
                         x['exact_round'] or '', x['cert_round'] or '', round(x['final_true_ratio'], 4), x['final_cert_L'], rc[0]['nviol'],
                         rc[-1]['Fmax'], rc[-1]['pinned'], rc[-1].get('vstored', ''), max(r['stored'] for r in rc), r0, jwin,
                         '/'.join(f'{v:g}' for v in mj), bk])
            if sw == 'scale' and j >= 20 and tag == 'ours':
                traj[f'scale/{j}/ours'] = dict(nviol=[r['nviol'] for r in rc], ratio=[round(r['ratio_true'], 4) for r in rc])
open('t.csv', 'w').write(H + '\n' + '\n'.join(','.join(str(v) for v in r) for r in rows) + '\n')
json.dump(traj, open('trajt.json', 'w'))
print(len(rows), 'rows', len(traj), 'traj')
