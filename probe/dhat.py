#!/usr/bin/env python3
"""Top DHAT allocation sites by bytes live at the global peak (t-gmax).
usage: dhat.py DHAT.json [N]"""
import json, sys, re
d = json.load(open(sys.argv[1])); n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
ft = d['ftbl']; pps = d['pps']
tot_g = sum(p.get('gb', 0) for p in pps); tot_b = sum(p['tb'] for p in pps)
print(f'  bytes at t-gmax {tot_g // 1024} KB, total allocated {tot_b // 1024} KB, sites {len(pps)}')
def frames(p):
    out = []
    for i in p['fs']:
        f = ft[i]
        m = re.search(r': (.*?)(?: \(|$)', f)
        name = m.group(1) if m else f
        if any(s in name for s in ('alloc::', 'core::ptr', '__rust', 'malloc', 'realloc', 'RawVec', 'hashbrown', 'smallvec', 'core::')):
            continue
        out.append(name[:90])
        if len(out) == 3: break
    return ' < '.join(out)
agg = {}
for p in pps:
    k = frames(p)
    a = agg.setdefault(k, [0, 0, 0])
    a[0] += p.get('gb', 0); a[1] += p.get('gbk', 0); a[2] += p['tb']
for k, (gb, gbk, tb) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:n]:
    print(f'  {gb // 1024:6d} KB {gbk:6d} blk (tot {tb // 1024:6d} KB)  {k}')
