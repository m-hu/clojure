#!/usr/bin/env bash
# Probe (throwaway, #271): Linux peak RSS of generic programs (the i259
# corpus, no stdin) for two revisions x three AOT orders (none, the
# committed one, one regenerated now by regen-aot.sh): separates code
# effects from layout effects on MEM.
set -euo pipefail
root=$(pwd)
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q valgrind time > /dev/null
out=/tmp/rss; mkdir -p $out
blank='SECTIONS { .text.hot : { } } INSERT BEFORE .text;'
for rev in 90509c29 dd228091; do
  w=/tmp/w-$rev
  git worktree add -f "$w" "$rev" > /dev/null 2>&1
  cp "$w/rust/hot-text-aot.ld" "$out/committed-$rev.ld"
  bash "$root/scripts/bench/hot-text/regen-aot.sh" "$w/rust" "$out/fresh-$rev.ld" > "$out/regen-$rev.log" 2>&1 || { tail -20 "$out/regen-$rev.log"; exit 1; }
  echo "$blank" > "$out/none-$rev.ld"
  for order in none committed fresh; do
    cp "$out/$order-$rev.ld" "$w/rust/hot-text-aot.ld"
    (cd "$w/rust" && cargo build --release --locked -p clojure-rs > /dev/null 2>&1 && cargo rustc --release --locked -p clj-rt --crate-type staticlib > /dev/null 2>&1)
    b=$out/bin-$rev-$order; mkdir -p "$b"; cp "$w/rust/target/release/clojure-rs" "$w/rust/target/release/libcljrt.a" "$b/"
    for f in "$root"/docs/spec/i259-native-compiler/corpus/*.clj "$root"/scripts/bench/hot-text/workloads/*.clj; do
      k=$(basename "$f" .clj)
      "$b/clojure-rs" build "$f" -o "$b/$k" > /dev/null 2>&1 || continue
      best=""
      for r in 1 2 3; do
        m=$(/usr/bin/time -f '%M' -o "$b/$k.t" "$b/$k" < /dev/null > /dev/null 2>&1; tail -1 "$b/$k.t")
        if [ -z "$best" ] || [ "$m" -lt "$best" ]; then best=$m; fi
      done
      echo "$rev $order $(basename $(dirname $f)) $k $best" >> $out/rss.txt
      rm -f "$b/$k"
    done
    echo "== $rev $order done: $(grep -c "^$rev $order " $out/rss.txt) programs"
  done
done
python3 - <<'PY'
import collections, statistics
rows = [l.split() for l in open('/tmp/rss/rss.txt')]
by = collections.defaultdict(dict)
for rev, order, group, k, m in rows:
    by[(rev, order, group)][k] = int(m)
for (rev, order, group), d in sorted(by.items()):
    print(f'{rev} {order:9s} {group:9s} n={len(d):3d} median {statistics.median(d.values()):7.0f} KB')
base = by[('dd228091', 'committed', 'corpus')]
for key in sorted(by):
    if key[2] != 'corpus': continue
    common = [k for k in by[key] if k in base]
    if common:
        delta = sorted(by[key][k] - base[k] for k in common)
        print(f'vs dd228091/committed corpus: {key[0]} {key[1]:9s} median delta {statistics.median(delta):+.0f} p10 {delta[len(delta)//10]:+d} p90 {delta[9*len(delta)//10]:+d}')
PY
