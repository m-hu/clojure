#!/usr/bin/env bash
# Probe (throwaway, #271): Linux A/B of two revisions (A = HEAD~2's rust/,
# B = HEAD~1's rust/; HEAD adds only this probe). Per side: regenerate the
# AOT hot-text order (as CI does), build, then per program (public snippet
# sources, synthetic inputs written for this probe) the callgrind
# instruction total of one run and the median peak RSS of 7 runs.
set -euo pipefail
root=$(pwd); out=/tmp/ab; mkdir -p $out
for v in A B; do
  ref=HEAD~2; [ $v = B ] && ref=HEAD~1
  git checkout -q $ref -- rust
  echo "######## $v $(git rev-parse --short $ref)"
  bash scripts/bench/hot-text/regen-aot.sh rust > $out/$v-regen.log 2>&1 || { tail -20 $out/$v-regen.log; exit 1; }
  (cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > $out/$v-build.log 2>&1 || { tail -30 $out/$v-build.log; exit 1; }
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
done
git checkout -q HEAD -- rust
python3 - <<'PY'
import collections, re
rows = collections.defaultdict(dict)
for v in ('A', 'B'):
    for l in open(f'/tmp/ab/{v}.txt').read().splitlines():
        m = re.match(r'(\S+) (\S+) (\S+) ins=(\d+) rss=(\d+)', l)
        if m: rows[(m.group(2), m.group(3))][v] = (int(m.group(4)), int(m.group(5)))
print(f"{'program':16} {'tag':6} {'A ins':>12} {'B/A':>7} | {'A KB':>6} {'B-A':>6}")
for (k, t), d in sorted(rows.items()):
    if 'A' in d and 'B' in d:
        a, b = d['A'], d['B']
        print(f"{k:16} {t:6} {a[0]:12d} {b[0]/a[0]:7.3f} | {a[1]:6d} {b[1]-a[1]:+6d}")
PY
