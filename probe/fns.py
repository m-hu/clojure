"""Functions a callgrind profile ran (self cost > 0): `name cost` lines."""
import collections
import re
import sys

names = {}
cost = collections.Counter()
fn = None
skip = False
for line in open(sys.argv[1], errors='replace'):
    if line.startswith('fn='):
        m = re.match(r'fn=\((\d+)\)\s*(.*)$', line.rstrip('\n'))
        if m:
            if m.group(2):
                names[m.group(1)] = m.group(2)
            fn = names.get(m.group(1), '?')
        else:
            fn = line[3:].rstrip('\n')
    elif line.startswith('cfn='):
        m = re.match(r'cfn=\((\d+)\)\s*(.*)$', line.rstrip('\n'))
        if m and m.group(2):
            names[m.group(1)] = m.group(2)
    elif line.startswith('calls='):
        skip = True
    elif line[:1].isdigit() or line[:1] in '+-*':
        if skip:
            skip = False
            continue
        parts = line.split()
        if len(parts) >= 2:
            cost[fn] += int(parts[1])
for f, v in cost.items():
    if v > 0:
        print(re.sub(r"'\d+$", '', f), v)
