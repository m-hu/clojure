#!/usr/bin/env python3
"""Executed functions of BINARY from callgrind outputs, and a link order.

usage: cgfuncs.py BINARY OUT_PREFIX CG_FILE...

Writes OUT_PREFIX.funcs (count-of-runs Ir size name, hottest first),
OUT_PREFIX.order (exact names, for --symbol-ordering-file) and
OUT_PREFIX.ld (an INSERT BEFORE .text script with hash-free globs).
"""
import collections, re, subprocess, sys

binary, prefix, files = sys.argv[1], sys.argv[2], sys.argv[3:]
sizes = {}
for line in subprocess.run(['nm', '-S', '--defined-only', binary], capture_output=True, text=True, check=True).stdout.splitlines():
    p = line.split(None, 3)
    if len(p) == 4 and p[2] in 'tTwW':
        sizes[p[3]] = int(p[1], 16)

def resolve(table, spec):
    m = re.match(r'\((\d+)\)\s*(.*)$', spec)
    if not m:
        return spec
    if m.group(2):
        table[m.group(1)] = m.group(2)
    return table.get(m.group(1), '?')

runs = collections.Counter()
ir = collections.Counter()
first = {}
for path in files:
    obs, fns = {}, {}
    ob = fn = None
    skip = False
    seen = set()
    for line in open(path, errors='replace'):
        line = line.rstrip('\n')
        if line.startswith('ob='):
            ob = resolve(obs, line[3:])
        elif line.startswith('cob='):
            resolve(obs, line[4:])
        elif line.startswith('fn='):
            fn = resolve(fns, line[3:])
        elif line.startswith('cfn='):
            resolve(fns, line[4:])
        elif line.startswith('calls='):
            skip = True
        elif line[:1].isdigit() or line[:1] in '+-*':
            if skip:
                skip = False
                continue
            parts = line.split()
            if fn and ob and ob.endswith('clojure-rs') and len(parts) > 1:
                ir[fn] += int(parts[1])
                if fn not in seen:
                    seen.add(fn)
                    first.setdefault(fn, len(first))
    for f in seen:
        runs[f] += 1

order = sorted(runs, key=lambda f: (-runs[f], first[f]))
known = [f for f in order if f in sizes]
with open(prefix + '.funcs', 'w') as out:
    for f in order:
        out.write(f'{runs[f]} {ir[f]} {sizes.get(f, -1)} {f}\n')
with open(prefix + '.order', 'w') as out:
    out.write(''.join(f + '\n' for f in known))

def glob(name):
    m = re.match(r'^(.*17h)[0-9a-f]{16}E(\.llvm\.\d+)?$', name)
    if m:
        return m.group(1) + '*'
    return re.sub(r'\.llvm\.\d+$', '*', name)

pats, seen = [], set()
for f in known:
    g = glob(f)
    if g not in seen:
        seen.add(g)
        pats.append(g)
with open(prefix + '.ld', 'w') as out:
    out.write('SECTIONS\n{\n  .text.hot :\n  {\n')
    for g in pats:
        out.write(f'    *(.text.{g} .text.unlikely.{g})\n')
    out.write('  }\n}\nINSERT BEFORE .text;\n')
tot = sum(sizes[f] for f in known)
print(f'  executed functions: {len(order)} ({len(known)} sized), total size {tot // 1024} KB; {len(pats)} globs')
for n in sorted(set(runs.values()), reverse=True)[:6]:
    fs = [f for f in known if runs[f] == n]
    print(f'    in {n} runs: {len(fs)} functions, {sum(sizes[f] for f in fs) // 1024} KB')
