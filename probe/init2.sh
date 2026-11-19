#!/usr/bin/env bash
# Probe (throwaway, #271): callgrind on EMPTY stdin (no inputs) of the
# near-miss scripts at this head, plus a trivial program; per-function
# self cost summed over the scripts, minus the trivial run per script.
set -euo pipefail
root=$(pwd)
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q valgrind > /dev/null
(cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > /tmp/build.log 2>&1
c=$root/rust/target/release/clojure-rs
mkdir -p /tmp/prof /tmp/bin
printf '(println 1)\n' > /tmp/trivial.clj
"$c" build /tmp/trivial.clj -o /tmp/bin/prog > /dev/null 2>&1
valgrind --tool=callgrind --dump-instr=no --callgrind-out-file=/tmp/prof/trivial.cg /tmp/bin/prog < /dev/null > /dev/null 2>&1 || true
while read -r s; do
  [ -z "$s" ] && continue
  n=$(echo "$s" | tr '/' '_')
  "$c" build "$root/snip/src/$s.clj" -o /tmp/bin/prog > /dev/null 2>&1 || { echo "$s build-fail"; continue; }
  valgrind --tool=callgrind --dump-instr=no --callgrind-out-file="/tmp/prof/$n.cg" /tmp/bin/prog < /dev/null > /dev/null 2>&1 || true
done < probe/scripts.txt
python3 - <<'PY'
import collections, glob, os, sys
sys.path.insert(0, 'probe')
from cgsum import parse
tri_ob, tri_fn = parse('/tmp/prof/trivial.cg')
print('trivial total', sum(tri_ob.values()))
tot = collections.Counter(); totals = []
files = sorted(f for f in glob.glob('/tmp/prof/*.cg') if not f.endswith('trivial.cg'))
for f in files:
    ob, fn = parse(f)
    totals.append((sum(ob.values()), os.path.basename(f)))
    for k, v in fn.items():
        tot[k] += v - tri_fn.get(k, 0)
for t, n in sorted(totals):
    print(f'{t:9d} {n}')
n = len(files)
print(f'per-script mean above trivial by function ({n} scripts):')
for (o, f), v in tot.most_common(60):
    print(f'{v / n:9.0f} {o.split("/")[-1][:10]:10s} {f[:110]}')
PY
