"""Per-iteration self cost by function: (cg2 - cg1) / iters (throwaway, #271)."""
import collections, re, subprocess, sys
def selfcost(path):
    out = subprocess.run(['callgrind_annotate', '--inclusive=no', '--threshold=100', path],
                         capture_output=True, text=True).stdout
    cost = collections.Counter()
    for line in out.splitlines():
        m = re.match(r'\s*([\d,]+)\s+(?:\([^)]*\)\s+)?(.*)$', line)
        if m and ':' in m.group(2):
            fn = m.group(2).split(':', 1)[1].strip()
            cost[re.sub(r'\[.*\]$', '', fn).strip()] += int(m.group(1).replace(',', ''))
    return cost
a, b, iters, name = selfcost(sys.argv[1]), selfcost(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
d = {k: (b[k] - a.get(k, 0)) / iters for k in b}
tot = sum(d.values())
rows = sorted(d.items(), key=lambda kv: -kv[1])
print(f'== {name}: {tot:.0f} per iteration')
for k, v in rows[:14]:
    if v >= 1:
        print(f'   {v:7.1f}  {k[:150]}')
