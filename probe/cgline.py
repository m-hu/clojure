"""Callgrind (--dump-line=yes --compress-pos=no): self cost by (fn, file,
line) and allocation calls by call site, mean per script above trivial."""
import collections
import glob
import os
import re
import sys

ALLOC = {'malloc', 'realloc', 'calloc', 'aligned_alloc', 'posix_memalign', 'memalign'}


def parse(path):
    names = {'fl': {}, 'fn': {}}
    fn = fi = cfn = None
    self_cost = collections.Counter()
    calls = collections.Counter()
    ncalls = collections.Counter()
    pending = None

    def res(kind, spec):
        m = re.match(r'\((\d+)\)\s*(.*)$', spec)
        if m:
            if m.group(2):
                names[kind][m.group(1)] = m.group(2)
            return names[kind].get(m.group(1), '?')
        return spec

    for line in open(path, errors='replace'):
        line = line.rstrip('\n')
        if not line:
            continue
        if line.startswith('fl='):
            fi = res('fl', line[3:])
        elif line.startswith(('fi=', 'fe=')):
            fi = res('fl', line[3:])
        elif line.startswith('fn='):
            fn = res('fn', line[3:])
        elif line.startswith(('cfl=', 'cfi=')):
            res('fl', line[4:])
        elif line.startswith('cfn='):
            cfn = res('fn', line[4:])
        elif line.startswith('calls='):
            pending = int(line[6:].split()[0])
        elif line[0].isdigit():
            parts = line.split()
            if len(parts) < 2:
                continue
            pos, cost = int(parts[0]), int(parts[1])
            if pending is not None:
                if cfn in ALLOC:
                    key = (fn, short(fi), pos, cfn)
                    calls[key] += cost
                    ncalls[key] += pending
                pending = None
                continue
            self_cost[(fn, short(fi), pos)] += cost
    return self_cost, calls, ncalls


def short(path):
    if path is None:
        return '?'
    for mark in ('/crates/', '/library/', '/registry/src/'):
        i = path.find(mark)
        if i >= 0:
            return path[i + len(mark):]
    return path.rsplit('/', 1)[-1]


def excess(files, tri, idx):
    tot = collections.Counter()
    for f in files:
        for k, v in f[idx].items():
            tot[k] += v - tri[idx].get(k, 0)
    return tot


def main(d):
    tri = parse(os.path.join(d, 'trivial.cg'))
    files = [parse(f) for f in sorted(glob.glob(os.path.join(d, '*.cg'))) if not f.endswith('trivial.cg')]
    n = len(files)
    selfx = excess(files, tri, 0)
    callx = excess(files, tri, 1)
    cntx = excess(files, tri, 2)
    print(f'== self cost by source line, mean per script above trivial ({n} scripts)')
    for (fn, fi, pos), v in selfx.most_common(90):
        print(f'{v / n:8.0f} {fi}:{pos}  [{fn[:70]}]')
    print('== rt_n_main_lite self cost by source line')
    main_lines = collections.Counter({k: v for k, v in selfx.items() if k[0] == 'rt_n_main_lite'})
    for (fn, fi, pos), v in main_lines.most_common(50):
        print(f'{v / n:8.0f} {fi}:{pos}')
    print('== allocation calls by call site (inclusive cost, calls), mean per script above trivial')
    for key, v in callx.most_common(70):
        fn, fi, pos, cfn = key
        print(f'{v / n:8.0f} {cntx[key] / n:7.1f} {cfn:7s} {fi}:{pos}  [{fn[:70]}]')
    tot_calls = sum(cntx.values()) / n
    print(f'total allocation calls above trivial per script: {tot_calls:.0f}; trivial: {sum(tri[2].values())}')


main(sys.argv[1])
