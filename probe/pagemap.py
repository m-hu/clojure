#!/usr/bin/env python3
"""Residency of a binary's pages at the end of a clojure-rs run.

usage: pagemap.py BINARY SCRIPT.clj SYMOUT

Runs `BINARY run SCRIPT` (the script must print @@READY@@ and then block
on (read-line)), reads /proc/PID/pagemap for every mapping of BINARY,
and prints resident KB per ELF section (file-backed vs anon). Writes the
symbols overlapping resident .text pages to SYMOUT.
"""
import os, re, struct, subprocess, sys, collections, bisect

binary, script, symout = sys.argv[1], sys.argv[2], sys.argv[3]
real = os.path.realpath(binary)
secs = []  # (file_off, size, name, vaddr)
for line in subprocess.run(['readelf', '-S', '-W', binary], capture_output=True, text=True, check=True).stdout.splitlines():
    m = re.match(r'\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)', line)
    if m and m.group(2) != 'NOBITS' and int(m.group(3), 16) and int(m.group(5), 16):
        secs.append((int(m.group(4), 16), int(m.group(5), 16), m.group(1), int(m.group(3), 16)))
secs.sort()
text = [s for s in secs if s[2].startswith('.text')]
syms = []  # (file_off, size, name)
for line in subprocess.run(['nm', '-S', '--defined-only', binary], capture_output=True, text=True, check=True).stdout.splitlines():
    p = line.split(None, 3)
    if len(p) == 4 and p[2] in 'tTwW':
        a, sz = int(p[0], 16), int(p[1], 16)
        for off, size, name, va in text:
            if va <= a < va + size:
                syms.append((off + a - va, sz, p[3]))
                break
syms.sort()
starts = [s[0] for s in syms]

proc = subprocess.Popen([binary, 'run', script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
for line in proc.stdout:
    if '@@READY@@' in line:
        break
pages = {}  # file offset -> is_file_page
anon_other = collections.Counter()
with open(f'/proc/{proc.pid}/maps') as fm, open(f'/proc/{proc.pid}/pagemap', 'rb') as pm:
    for line in fm:
        parts = line.split()
        start, end = (int(x, 16) for x in parts[0].split('-'))
        path = parts[5] if len(parts) > 5 else '[anon]'
        mine = path.startswith('/') and os.path.realpath(path) == real
        pm.seek(start // 4096 * 8)
        data = pm.read((end - start) // 4096 * 8)
        n = 0
        for i in range(0, len(data), 8):
            e = struct.unpack_from('<Q', data, i)[0]
            if e >> 63 & 1:
                n += 1
                if mine:
                    pages[int(parts[2], 16) + i // 8 * 4096] = (e >> 61) & 1
        if not mine:
            anon_other[path] += n * 4
proc.stdin.close()
proc.wait()

per = collections.Counter()
for off, filepage in pages.items():
    for s_off, size, name, va in secs:
        if s_off < off + 4096 and off < s_off + size:
            per[(name, 'file' if filepage else 'anon')] += 4
print(f'  binary resident: {len(pages) * 4} KB ({sum(1 for v in pages.values() if not v) * 4} KB anon)')
for s_off, size, name, va in secs:
    f, a = per[(name, 'file')], per[(name, 'anon')]
    if f or a or size > 65536:
        print(f'    {name:24s} size {size // 1024:6d} KB  resident file {f:6d} KB anon {a:5d} KB')
print('  other mappings (KB):', ', '.join(f'{k}={v}' for k, v in anon_other.most_common(8)))
touched = {}
for off in pages:
    if not any(o <= off < o + sz for o, sz, _, _ in text):
        continue
    i = max(0, bisect.bisect_right(starts, off) - 1)
    while i < len(syms) and syms[i][0] < off + 4096:
        o, sz, name = syms[i]
        if o + max(sz, 1) > off:
            touched[name] = sz
        i += 1
with open(symout, 'w') as f:
    for name, sz in touched.items():
        f.write(f'{sz} {name}\n')
print(f'  symbols on resident text pages: {len(touched)} totalling {sum(touched.values()) // 1024} KB')
