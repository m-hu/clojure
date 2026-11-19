"""Self cost per object and per function in a callgrind file (Ir)."""
import sys, re, collections
def parse(path):
    obs, fns = {}, {}
    ob = fn = None
    skip = False
    by_ob = collections.Counter(); by_fn = collections.Counter()
    def res(table, spec):
        m = re.match(r'\((\d+)\)\s*(.*)$', spec)
        if m:
            if m.group(2): table[m.group(1)] = m.group(2)
            return table.get(m.group(1), '?')
        return spec
    for line in open(path, errors='replace'):
        line = line.rstrip('\n')
        if line.startswith('ob='): ob = res(obs, line[3:])
        elif line.startswith('cob='): res(obs, line[4:])
        elif line.startswith('fn='): fn = res(fns, line[3:])
        elif line.startswith('cfn='): res(fns, line[4:])
        elif line.startswith('calls='): skip = True
        elif line[:1].isdigit() or line[:1] in '+-*':
            parts = line.split()
            if skip:
                skip = False; continue
            if len(parts) >= 2:
                c = int(parts[1]); by_ob[ob] += c; by_fn[(ob, fn)] += c
    return by_ob, by_fn
if __name__ == "__main__":
    a_ob, a_fn = parse(sys.argv[1])
      a_ob, a_fn = parse(sys.argv[1])
    print('total', sum(a_ob.values()))
    for o, c in a_ob.most_common(6): print(f'  {c:9d} {o}')
    if len(sys.argv) > 2:
        b_ob, b_fn = parse(sys.argv[2])
        print('base total', sum(b_ob.values()))
        diff = collections.Counter({k: a_fn[k] - b_fn.get(k, 0) for k in a_fn})
        for (o, f), c in diff.most_common(25):
            print(f'  {c:9d} {o.split("/")[-1][:14]:14s} {f[:100]}')
    
