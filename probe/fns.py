"""Functions a callgrind profile ran (self cost > 0): `name cost` lines."""
import collections, re, sys
names = {}; cost = collections.Counter(); fn = None; skip = False
for line in open(sys.argv[1], errors='replace'):
    if line.startswith(('fn=', 'cfn=')):
        body = line.rstrip('\n').split('=', 1)[1]
        m = re.match(r'\((\d+)\)\s*(.*)$', body)
        if m:
            if m.group(2):
                names[m.group(1)] = m.group(2)
            name = names.get(m.group(1), '?')
        else:
            name = body
        if line.startswith('fn='):
            fn = name
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
