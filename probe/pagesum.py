"""Summarize the pages probe: resident KB by mapping class with and
without fault-around, the text split hot/cold, and the executed runtime
functions' bytes and simulated 64 KiB windows."""
import collections
import glob
import json
import os
import re
import statistics
import sys

out = sys.argv[1]
sections = {}
for line in open(os.path.join(out, 'sections.txt')):
    m = re.search(r'\]\s+(\S+)\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)', line)
    if m:
        sections[m.group(1)] = (int(m.group(2), 16), int(m.group(4), 16))
hot = sections.get('.text.hot', (0, 0))
print('sections:', {k: (hex(a), s) for k, (a, s) in sections.items() if s > 4096})
syms = {}
for line in open(os.path.join(out, 'nm.txt')):
    p = line.split()
    if len(p) == 4 and p[2] in 'tTwW':
        syms[p[3]] = (int(p[0], 16), int(p[1], 16))


def classify(vmas):
    cls = {}
    binpath = None
    for v in vmas:
        if v['path'].startswith('/tmp/pages/bin/'):
            binpath = v['path']
    seen_r = 0
    prev_file = False
    for v in vmas:
        p = v['path']
        if p == binpath:
            if v['perm'] == 'r-xp':
                c = 'text'
            elif v['perm'].startswith('r--'):
                c = ['hdr', 'rodata', 'relro'][min(seen_r, 2)] if v['off'] else 'hdr'
                if v['off']:
                    seen_r += 1
                    c = ['rodata', 'relro'][min(seen_r - 1, 1)]
            else:
                c = 'data'
            prev_file = True
        elif p == '' and prev_file:
            c = 'bss'
            prev_file = False
        else:
            c = p.strip('[]') or 'anon'
            prev_file = False
        cls.setdefault(c, []).append(v)
    return cls


def kb(vs):
    return 4 * sum(len(v['present']) for v in vs)


def text_split(vs):
    h = c = 0
    for v in vs:
        for i in v['present']:
            a = v['lo'] + i * 4096
            if hot[0] <= a < hot[0] + hot[1]:
                h += 4
            else:
                c += 4
    return h, c


rows = collections.defaultdict(list)
per = {}
for f in sorted(glob.glob(os.path.join(out, 'pm', '*.fa.json'))):
    k = os.path.basename(f)[:-len('.fa.json')]
    g = os.path.join(out, 'pm', k + '.nofa.json')
    if not os.path.exists(g):
        continue
    a, b = json.load(open(f)), json.load(open(g))
    ca, cb = classify(a['vmas']), classify(b['vmas'])
    r = {'hwm_fa': a.get('VmHWM', 0), 'hwm_nofa': b.get('VmHWM', 0)}
    for c in set(ca) | set(cb):
        r[c + '_fa'] = kb(ca.get(c, []))
        r[c + '_nofa'] = kb(cb.get(c, []))
    r['hot_fa'], r['cold_fa'] = text_split(ca.get('text', []))
    r['hot_nofa'], r['cold_nofa'] = text_split(cb.get('text', []))
    fn = os.path.join(out, 'fns', k + '.txt')
    if os.path.exists(fn):
        names = [l.split()[0] for l in open(fn) if l.strip()]
        ran = [syms[n] for n in names if n in syms]
        r['fn_bytes'] = sum(s for _, s in ran) // 1024
        wins = set()
        pages = set()
        for a0, s in ran:
            for w in range(a0 >> 16, ((a0 + max(s, 1) - 1) >> 16) + 1):
                wins.add(w)
            for p in range(a0 >> 12, ((a0 + max(s, 1) - 1) >> 12) + 1):
                pages.add(p)
        r['sim_win_kb'] = 64 * len(wins)
        r['sim_page_kb'] = 4 * len(pages)
        r['fns'] = len(ran)
    per[k] = r
    rows[k.split('-', 1)[0]].append(r)

keys = ['hwm_fa', 'hwm_nofa', 'text_fa', 'text_nofa', 'hot_fa', 'hot_nofa', 'cold_fa', 'cold_nofa',
        'rodata_fa', 'rodata_nofa', 'relro_fa', 'relro_nofa', 'data_fa', 'data_nofa', 'bss_fa',
        'heap_fa', 'stack_fa', 'vdso_fa', 'hdr_fa', 'fn_bytes', 'sim_page_kb', 'sim_win_kb', 'fns']
for g, rs in sorted(rows.items()):
    print(f'== {g}: {len(rs)} programs, medians (KB)')
    for key in keys:
        vals = [r.get(key, 0) for r in rs]
        print(f'  {key:12s} median {statistics.median(vals):8.0f}  p10 {sorted(vals)[len(vals)//10]:7.0f}  p90 {sorted(vals)[9*len(vals)//10]:7.0f}')
print('== near-miss scripts (probe/scripts.txt)')
for s in open('probe/scripts.txt'):
    s = s.strip()
    if not s:
        continue
    k = 'snip-' + s.replace('/', '_')
    r = per.get(k)
    if r:
        print(f"  {r['hwm_fa']:6d} {r['hwm_nofa']:6d} text {r.get('text_fa',0):5d}/{r.get('text_nofa',0):5d} hot {r['hot_fa']:5d}/{r['hot_nofa']:5d} cold {r['cold_fa']:5d}/{r['cold_nofa']:5d} ro {r.get('rodata_fa',0):4d}/{r.get('rodata_nofa',0):4d} relro {r.get('relro_fa',0):4d}/{r.get('relro_nofa',0):4d} heap {r.get('heap_fa',0):4d} fnKB {r.get('fn_bytes',0):5d} simwin {r.get('sim_win_kb',0):5d}  {s}")
