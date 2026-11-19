"""DHAT JSON: allocation sites (first frame outside allocator/std plumbing,
plus its caller), blocks and bytes in total and live at exit, mean per
script above trivial; per-script totals."""
import collections
import glob
import json
import os
import sys

PLUMB = ('malloc', 'realloc', 'calloc', 'free', 'alloc::', 'core::', 'std::', 'hashbrown::',
         '__rust', '__rdl', '<alloc::', '<core::', '<std::', '<hashbrown::', 'smallvec::',
         '<smallvec::', 'rpds::', '<rpds::', 'archery::', '<archery::', '[root]', 'memalign',
         'posix_memalign', 'aligned_alloc', '<T as ', '<[T] as ', '<str as ')


def frame(s):
    # "0xADDR: name (file:line)" or "0xADDR: name (in /obj)"
    body = s.split(': ', 1)[1] if s.startswith('0x') else s
    i = body.rfind(' (')
    if i < 0:
        return body, ''
    return body[:i], body[i + 2:-1]


def sites(path):
    d = json.load(open(path))
    ftbl = d['ftbl']
    agg = collections.defaultdict(lambda: [0, 0, 0, 0])
    tot = [0, 0, 0, 0, 0]
    for pp in d['pps']:
        frames = [frame(ftbl[i]) for i in pp['fs']]
        keep = [f for f in frames if not f[0].startswith(PLUMB)]
        if keep:
            a = keep[0]
            b = keep[1] if len(keep) > 1 else ('', '')
            key = f'{a[0][:80]} ({a[1]}) <- {b[0][:60]} ({b[1]})'
        else:
            key = ' / '.join(f'{f[0][:40]} ({f[1]})' for f in frames[:4])
        r = agg[key]
        r[0] += pp['tbk']
        r[1] += pp['tb']
        r[2] += pp.get('ebk', 0)
        r[3] += pp.get('eb', 0)
        tot[0] += pp['tbk']
        tot[1] += pp['tb']
        tot[2] += pp.get('ebk', 0)
        tot[3] += pp.get('eb', 0)
        tot[4] += pp.get('gb', 0)
    return agg, tot


def main(d):
    tri, ttot = sites(os.path.join(d, 'trivial.json'))
    paths = sorted(p for p in glob.glob(os.path.join(d, '*.json')) if not p.endswith('trivial.json'))
    n = len(paths)
    ex = collections.defaultdict(lambda: [0, 0, 0, 0])
    print(f'trivial: blocks {ttot[0]} bytes {ttot[1]} live-at-exit blocks {ttot[2]} bytes {ttot[3]} t-gmax bytes {ttot[4]}')
    print('per script: blocks bytes exit-blocks exit-bytes gmax-bytes')
    for p in paths:
        agg, tot = sites(p)
        print(f'{tot[0]:7d} {tot[1]:9d} {tot[2]:7d} {tot[3]:9d} {tot[4]:9d} {os.path.basename(p)}')
        for k, r in agg.items():
            t = tri.get(k, [0, 0, 0, 0])
            e = ex[k]
            for i in range(4):
                e[i] += r[i] - t[i]
    print(f'== allocation sites by blocks, mean per script above trivial ({n} scripts): blocks bytes exit-blocks exit-bytes')
    for k, r in sorted(ex.items(), key=lambda kv: -kv[1][0])[:70]:
        print(f'{r[0] / n:7.1f} {r[1] / n:8.0f} {r[2] / n:7.1f} {r[3] / n:8.0f}  {k}')
    print('== allocation sites by bytes live at exit, mean per script above trivial')
    for k, r in sorted(ex.items(), key=lambda kv: -kv[1][3])[:40]:
        print(f'{r[0] / n:7.1f} {r[1] / n:8.0f} {r[2] / n:7.1f} {r[3] / n:8.0f}  {k}')
    print('== trivial program sites by blocks')
    for k, r in sorted(tri.items(), key=lambda kv: -kv[1][0])[:40]:
        print(f'{r[0]:7d} {r[1]:8d} {r[2]:7d} {r[3]:8d}  {k}')


main(sys.argv[1])
