#!/usr/bin/env bash
# Probe (throwaway, #271): Linux callgrind of empty-stdin startup for
# public near-line programs, built by this tree's compiler with a fresh
# AOT hot-text order (as CI does), then the hottest functions overall.
set -euo pipefail
out=/tmp/st; mkdir -p $out
bash scripts/bench/hot-text/regen-aot.sh rust > $out/regen.log 2>&1 || { tail -20 $out/regen.log; exit 1; }
(cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > $out/build.log 2>&1 || { tail -30 $out/build.log; exit 1; }
c=$(pwd)/rust/target/release/clojure-rs
for f in probe/sprog/*.clj; do
  k=$(basename $f .clj)
  $c build $f -o $out/$k --keep-symbols > /dev/null 2>&1 || { echo "$k build-fail"; continue; }
  valgrind --tool=callgrind --callgrind-out-file=$out/$k.cg $out/$k < /dev/null > /dev/null 2> $out/$k.vg || true
  echo "$k $(grep -m1 '^summary:' $out/$k.cg | awk '{print $2}')"
done | tee $out/totals.txt
for f in $out/*.cg; do callgrind_annotate --inclusive=no --threshold=100 $f 2>/dev/null | sed -n '/Ir/,$p' | head -60 > ${f%.cg}.top; done
python3 - <<'PY'
import glob, re, collections
tot = collections.Counter(); n = 0
for f in glob.glob('/tmp/st/*.top'):
    if 'trivial' in f: continue
    n += 1
    for line in open(f):
        m = re.match(r'\s*([\d,]+)\s+\([^)]*\)\s+(?:\S+:)?(.+?)\s*(?:\[.*\])?$', line)
        if m:
            tot[m.group(2)[:120]] += int(m.group(1).replace(',', ''))
print(f"programs {n}; per-program mean self Ir by function:")
for k, v in tot.most_common(50): print(f"{v//n:9d} {k}")
PY
echo "== trivial"; head -30 $out/trivial.top
