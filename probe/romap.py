#!/usr/bin/env python3
"""Map resident .rodata pages to the input sections of an lld link map.
usage: romap.py BINARY MAP SCRIPT.clj"""
import os, re, struct, subprocess, sys, collections
binary, mapf, script = sys.argv[1:4]
# lld -Map format: "     VMA      LMA     Size Align Out     In      Symbol"
secs = []  # (vma, size, name)
cur_out = None
for line in open(mapf, errors='replace'):
    m = re.match(r'\s*([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+(\d+)\s(\s*)(\S.*)$', line)
    if not m: continue
    vma, size, indent, rest = int(m.group(1), 16), int(m.group(3), 16), len(m.group(5)), m.group(6)
    if indent == 0:
        cur_out = rest.strip()
    elif cur_out and cur_out.startswith('.rodata') and indent <= 8 and ':(' in rest:
        name = rest.split(':(', 1)[1].rstrip(')')
        secs.append((vma, size, name))
print('rodata input sections', len(secs), 'total', sum(s for _, s, _ in secs) // 1024, 'KB')
kinds = collections.Counter(); ksz = collections.Counter()
def kind(n):
    n = re.sub(r'\.(Lanon|L__unnamed)\S*', '.<anon>', n)
    n = re.sub(r'_R\S+', '<sym>', n)
    n = re.sub(r'\.str1\.\d+', '.str1.N', n)
    n = re.sub(r'\.cst\d+', '.cstN', n)
    return n
for v, s, n in secs: kinds[kind(n)] += 1; ksz[kind(n)] += s
for k, c in ksz.most_common(12): print('  %6d KB %6d secs  %s' % (c // 1024, kinds[k], k))
proc = subprocess.Popen([binary, 'run', script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
for line in proc.stdout:
    if '@@READY@@' in line: break
res = set(); real = os.path.realpath(binary)
with open(f'/proc/{proc.pid}/maps') as fm, open(f'/proc/{proc.pid}/pagemap', 'rb') as pm:
    for line in fm:
        p = line.split()
        if len(p) < 6 or not p[5].startswith('/') or os.path.realpath(p[5]) != real: continue
        a, b = (int(x, 16) for x in p[0].split('-'))
        pm.seek(a // 4096 * 8); d = pm.read((b - a) // 4096 * 8)
        for i in range(0, len(d), 8):
            if struct.unpack_from('<Q', d, i)[0] >> 63 & 1: res.add(a + i // 8 * 4096)
proc.stdin.close(); proc.wait()
hit = collections.Counter(); hitn = collections.Counter()
for v, s, n in secs:
    pages = range(v // 4096 * 4096, v + max(s, 1), 4096)
    if any(pg in res for pg in pages):
        hit[kind(n)] += s; hitn[kind(n)] += 1
print('touched rodata input sections by kind (KB of section bytes on resident pages):')
for k, c in hit.most_common(12): print('  %6d KB %6d secs  %s' % (c // 1024, hitn[k], k))
