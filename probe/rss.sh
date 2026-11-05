#!/usr/bin/env bash
# Linux RSS probe for #226: where a trivial clojure-rs run's peak RSS goes.
set -u
summ() { # smaps -> RSS per mapping (KB), largest first
  python3 - "$1" <<'PY'
import re, sys, collections
rss = collections.Counter(); anon = collections.Counter(); name = None
for line in open(sys.argv[1], errors='replace'):
    m = re.match(r'^[0-9a-f]+-[0-9a-f]+ \S+ \S+ \S+ \S+\s*(.*)$', line)
    if m:
        name = m.group(1).strip() or '[anon]'
        continue
    m = re.match(r'^(Rss|Anonymous):\s+(\d+) kB', line)
    if m and name:
        (rss if m.group(1) == 'Rss' else anon)[name] += int(m.group(2))
tot = sum(rss.values())
print(f'  total Rss {tot} KB')
for n, v in rss.most_common(14):
    print(f'  {v:7d} KB rss  {anon[n]:7d} KB anon  {n}')
PY
}
run3() { for i in 1 2 3; do /usr/bin/time -f "    maxrss_kb=%M minflt=%R" "$@" > /dev/null; done; }
echo "== kernel"; uname -r; cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null
printf '(println 1)\n' > /tmp/one.clj
printf '(print (slurp "/proc/self/smaps"))\n' > /tmp/smaps.clj
printf '(print (slurp "/proc/self/smaps_rollup"))\n' > /tmp/rollup.clj
for b in "$@"; do
  echo "== $b ($(stat -c %s "$b") bytes)"; ldd "$b" | sed 's/^/  /'
  echo "  trivial run:"; run3 "$b" run /tmp/one.clj
  "$b" run /tmp/rollup.clj | grep -E '^(Rss|Pss|Anonymous|AnonHugePages|Shared_Clean|Private_Clean|Private_Dirty)' | sed 's/^/  /'
  "$b" run /tmp/smaps.clj > /tmp/rs.smaps; summ /tmp/rs.smaps
done
b=$1
for env in "MIMALLOC_ARENA_EAGER_COMMIT=0" "MIMALLOC_EAGER_COMMIT=0" "MIMALLOC_ARENA_RESERVE=0" "MIMALLOC_ALLOW_LARGE_OS_PAGES=0"; do
  echo "== $b with $env"; run3 env "$env" "$b" run /tmp/one.clj
done
echo "== python3.11"; python3.11 --version
run3 python3.11 -c 'print(1)'
python3.11 -c "print(open('/proc/self/smaps_rollup').read())" | grep -E '^(Rss|Pss|Anonymous|AnonHugePages|Shared_Clean|Private_Clean|Private_Dirty)' | sed 's/^/  /'
python3.11 -c "print(open('/proc/self/smaps').read())" > /tmp/py.smaps; summ /tmp/py.smaps
