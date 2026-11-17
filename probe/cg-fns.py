"""Executed functions of object BASE in a callgrind output, with self cost."""
import collections, os, re, sys
def resolve(table, spec):
    m = re.match(r'\((\d+)\)\s*(.*)$', spec)
    if not m:
        return spec
    if m.group(2):
        table[m.group(1)] = m.group(2)
    return table.get(m.group(1), '?')
obs, fns = {}, {}
ob = fn = None
skip = False
seen = collections.OrderedDict()
for line in open(sys.argv[1], errors='replace'):
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
        elif fn and ob and os.path.basename(ob) == sys.argv[2]:
            seen.setdefault(fn, 0)
for n in seen:
    print(n)
