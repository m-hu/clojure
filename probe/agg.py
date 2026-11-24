"""Aggregate callgrind profiles (throwaway probe, #271): per-profile totals
and the top inclusive functions, then self costs summed across profiles."""
import collections, glob, os, re, subprocess, sys

d = sys.argv[1]
line = re.compile(r'^\s*([\d,]+)\s+\(\s*[\d.]+%\)\s+(\S.*)$')

def annotate(path, inclusive):
    args = ['callgrind_annotate', '--threshold=99.9']
    if inclusive:
        args.append('--inclusive=yes')
    out = subprocess.run(args + [path], capture_output=True, text=True).stdout
    rows = []
    for l in out.splitlines():
        m = line.match(l)
        if m:
            fn = re.sub(r'^.*:', '', m.group(2)).split(' [')[0]
            rows.append((int(m.group(1).replace(',', '')), fn))
    return rows

def total(path):
    for l in open(path):
        if l.startswith(('summary:', 'totals:')):
            return int(l.split()[1])
    return 0

files = sorted(glob.glob(os.path.join(d, '*.cg')))
floor = total(os.path.join(d, 'trivial.cg'))
print(f'trivial floor {floor:,}')
agg = collections.Counter(); inc = collections.Counter(); triv_inc = dict((fn, n) for n, fn in annotate(os.path.join(d, 'trivial.cg'), True))
names = 0
for f in files:
    name = os.path.basename(f)[:-3]
    t = total(f)
    print(f'\n== {name} total {t:,} (+{t - floor:,} over the floor)')
    for n, fn in annotate(f, True)[:25]:
        print(f'  {n:>12,}  {fn[:110]}')
    if name != 'trivial':
        names += 1
        for n, fn in annotate(f, False):
            agg[fn] += n
        for n, fn in annotate(f, True):
            inc[fn] += n - triv_inc.get(fn, 0)
print('\n== self cost summed over the scripts (top 80)')
for fn, n in agg.most_common(80):
    print(f'  {n:>14,}  {fn[:120]}')
print(f'\n== inclusive cost above the trivial program, mean over {names} scripts (top 120)')
for fn, n in inc.most_common(120):
    print(f'  {n // max(names, 1):>12,}  {fn[:140]}')
