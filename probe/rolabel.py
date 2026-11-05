#!/usr/bin/env python3
"""Label resident .rodata pages: embedded .clj sources vs named symbols vs other.
usage: rolabel.py BINARY SCRIPT.clj SRCDIR"""
import os, re, struct, subprocess, sys, collections
binary, script, srcdir = sys.argv[1], sys.argv[2], sys.argv[3]
data = open(binary, 'rb').read()
ro = None
for line in subprocess.run(['readelf', '-S', '-W', binary], capture_output=True, text=True).stdout.splitlines():
    m = re.match(r'\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)', line)
    if m and m.group(1) == '.rodata':
        ro = (int(m.group(3), 16), int(m.group(4), 16), int(m.group(2), 16))
off0, size, va0 = ro
# locate embedded sources by content
labels = []
for root, _, files in os.walk(srcdir):
    for f in files:
        if f.endswith('.clj') or f.endswith('.edn'):
            txt = open(os.path.join(root, f), 'rb').read()
            if len(txt) < 2048: continue
            i = data.find(txt[:256], off0, off0 + size)
            if i >= 0:
                labels.append((i, i + len(txt), os.path.relpath(os.path.join(root, f), srcdir)))
syms = []
for line in subprocess.run(['nm', '-S', '--defined-only', '-C', binary], capture_output=True, text=True).stdout.splitlines():
    p = line.split(None, 3)
    if len(p) == 4 and p[2] in 'rR':
        a, sz = int(p[0], 16), int(p[1], 16)
        if va0 <= a < va0 + size and sz >= 512:
            syms.append((off0 + a - va0, off0 + a - va0 + sz, 'sym ' + p[3][:60]))
proc = subprocess.Popen([binary, 'run', script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
for line in proc.stdout:
    if '@@READY@@' in line: break
res = set()
real = os.path.realpath(binary)
with open(f'/proc/{proc.pid}/maps') as fm, open(f'/proc/{proc.pid}/pagemap', 'rb') as pm:
    for line in fm:
        parts = line.split()
        if len(parts) < 6 or not parts[5].startswith('/') or os.path.realpath(parts[5]) != real: continue
        start, end = (int(x, 16) for x in parts[0].split('-'))
        pm.seek(start // 4096 * 8); d = pm.read((end - start) // 4096 * 8)
        for i in range(0, len(d), 8):
            if struct.unpack_from('<Q', d, i)[0] >> 63 & 1:
                res.add(int(parts[2], 16) + i // 8 * 4096)
proc.stdin.close(); proc.wait()
agg = collections.Counter(); tot = collections.Counter()
for pg in range(off0 // 4096 * 4096, off0 + size, 4096):
    lab = 'other'
    for s, e, l in labels + syms:
        if s < pg + 4096 and pg < e:
            lab = l; break
    tot[lab] += 4
    if pg in res: agg[lab] += 4
print('  .rodata resident by content (KB resident / KB pages):')
for lab, kb in sorted(tot.items(), key=lambda kv: -agg[kv[0]]):
    if agg[lab] or kb >= 32:
        print(f'    {agg[lab]:6d} / {kb:6d}  {lab}')
