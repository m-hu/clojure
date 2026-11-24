"""Add the libc.a/libgcc.a members that profiled runs execute to an order
file's .text.hot section (glibc has no function sections, so whole
members move). usage: libc_members.py ORDER.ld OUT.ld FNS..."""
import collections, subprocess, sys
order, out, fns = sys.argv[1], sys.argv[2], sys.argv[3:]
owner = {}
for lib in ('libc.a', 'libgcc.a', 'libm.a'):
    path = subprocess.run(['cc', '-print-file-name=' + lib], capture_output=True, text=True).stdout.strip()
    if not path or path == lib:
        continue
    nm = subprocess.run(['nm', '-A', '--defined-only', path], capture_output=True, text=True, errors='replace').stdout
    for line in nm.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[-2].lower() not in ('t', 'i', 'w'):
            continue
        # GNU nm -A: "/path/libc.a:member.o:0000000000000000 T name"
        fields = parts[0].split(':')
        mem = [x for x in fields if x.endswith('.o') or x.endswith('.os')]
        if not mem:
            continue
        member = mem[-1]
        owner.setdefault(parts[-1], (lib, member))
count = collections.Counter(); first = {}
for f in fns:
    seen = set()
    for line in open(f):
        name = line.split()[0]
        m = owner.get(name)
        if m and m not in seen:
            seen.add(m); count[m] += 1; first.setdefault(m, len(first))
members = sorted(count, key=lambda m: (-count[m], first[m]))
print(f'{len(members)} archive members from {len(fns)} profiles: ' + ' '.join(m for _, m in members[:20]))
assert all(m.endswith(('.o', '.os')) for _, m in members), members[:5]
s = open(order).read()
i = s.index('.text.hot :'); j = s.index('\n', s.index('{', i)) + 1
open(out, 'w').write(s[:j] + ''.join(f'    *{lib}:{m}(.text .text.*)\n' for lib, m in members) + s[j:])
