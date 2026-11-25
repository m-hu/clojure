#!/usr/bin/env bash
# Probe (throwaway, #271): AOT optimization settings (O3 pipeline, x86-64-v3).
# Per variant: regenerate the AOT hot-text order (as CI does), build,
# then per program (public snippet sources, synthetic inputs written for
# this probe) the callgrind instruction total of one run and the median
# peak RSS of 7 runs, on its input and on empty stdin.
set -euo pipefail
root=$(pwd)
out=/tmp/opt; mkdir -p $out
variant() {
  git checkout -q -- rust
  unset RUSTFLAGS
  case $1 in
    base) ;;
    o3|o3v3)
      sed -i 's/optimize_module(&module, &machine, "default<O2>")/optimize_module(\&module, \&machine, "default<O3>")/' rust/crates/clj-aot/src/lib.rs
      sed -i 's/            OptimizationLevel::Default,/            OptimizationLevel::Aggressive,/' rust/crates/clj-aot/src/target.rs ;;
  esac
  case $1 in
    v3|o3v3)
      sed -i 's/            "generic",/            "x86-64-v3",/' rust/crates/clj-aot/src/target.rs
      export RUSTFLAGS="-C target-cpu=x86-64-v3" ;;
  esac
  git diff --stat rust
}
for v in base o3 v3 o3v3; do
  echo "######## $v"
  variant $v
  bash scripts/bench/hot-text/regen-aot.sh rust > $out/$v-regen.log 2>&1 || { tail -20 $out/$v-regen.log; continue; }
  (cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > $out/$v-build.log 2>&1 || { tail -30 $out/$v-build.log; continue; }
  c=$root/rust/target/release/clojure-rs
  for f in probe/prog/*.clj; do
    k=$(basename $f .clj)
    $c build $f -o $out/$v-$k > /dev/null 2>&1 || { echo "$v $k build-fail"; continue; }
    for inp in probe/prog/$k.in /dev/null; do
      tag=$([ $inp = /dev/null ] && echo empty || echo input)
      valgrind --tool=callgrind --callgrind-out-file=$out/$v-$k-$tag.cg $out/$v-$k < $inp > /dev/null 2>$out/$v-$k-$tag.vg || true
      ins=$(grep -m1 '^summary:' $out/$v-$k-$tag.cg | awk '{print $2}')
      rss=$(for i in 1 2 3 4 5 6 7; do /usr/bin/time -f %M $out/$v-$k < $inp 2>&1 >/dev/null | tail -1; done | sort -n | sed -n 4p)
      echo "$v $k $tag ins=$ins rss=$rss"
    done
  done | tee $out/$v.txt
  ls -l $out/$v-sudoku | awk '{print "size", $5}'
done
python3 - <<'PY'
import collections, re
rows = collections.defaultdict(dict)
for v in ('base', 'o3', 'v3', 'o3v3'):
    try: lines = open(f'/tmp/opt/{v}.txt').read().splitlines()
    except FileNotFoundError: continue
    for l in lines:
        m = re.match(r'(\S+) (\S+) (\S+) ins=(\d+) rss=(\d+)', l)
        if m: rows[(m.group(2), m.group(3))][v] = (int(m.group(4)), int(m.group(5)))
print(f"{'program':16} {'tag':6} {'base ins':>12} {'o3/b':>7} {'v3/b':>7} {'o3v3/b':>7} | {'base KB':>8} {'o3-b':>6} {'v3-b':>6} {'o3v3-b':>6}")
for (k, t), d in sorted(rows.items()):
    g = d.get('base')
    if not g: continue
    f = lambda v, i: (f"{d[v][0]/g[0]:.3f}" if i == 0 else f"{d[v][1]-g[1]:+d}") if v in d else '-'
    print(f"{k:16} {t:6} {g[0]:12d} {f('o3',0):>7} {f('v3',0):>7} {f('o3v3',0):>7} | {g[1]:8d} {f('o3',1):>6} {f('v3',1):>6} {f('o3v3',1):>6}")
PY
