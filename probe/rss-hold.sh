#!/usr/bin/env bash
# Probe (throwaway, #271): does training the AOT hot-text order on more
# generic programs cut peak RSS on programs it never saw? Profiles the
# workloads and the i259 corpus once (blank order), then builds orders
# from W = workloads only and WC = workloads + odd-numbered corpus files,
# and measures Linux peak RSS on the even-numbered corpus files (held
# out of both) and on the workloads.
set -euo pipefail
root=$(pwd)
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q valgrind time > /dev/null
here=$root/scripts/bench/hot-text
out=/tmp/hold; mkdir -p $out/cg-w $out/cg-c
ws=$root/rust
cp $ws/hot-text-aot.ld $out/committed.ld
printf '%s\n' 'SECTIONS' '{' '  .text.hot :' '  {' '  }' '}' 'INSERT BEFORE .text;' > $ws/hot-text-aot.ld
(cd $ws && cargo build --release --locked -p clojure-rs > /dev/null 2>&1 && cargo rustc --release --locked -p clj-rt --crate-type staticlib > /dev/null 2>&1)
c=$ws/target/release/clojure-rs
ls docs/spec/i259-native-compiler/corpus/*.clj | sort > $out/corpus.txt
awk 'NR % 2 == 1' $out/corpus.txt > $out/train.txt
awk 'NR % 2 == 0' $out/corpus.txt > $out/test.txt
profile() {  # FILE INPUT DIR
  local name; name=$(basename "$1" .clj)
  "$c" build "$1" -o "$out/prog" > /dev/null 2>&1 || return 0
  valgrind --tool=callgrind --demangle=no --dump-instr=no --callgrind-out-file="$3/$name.cg" "$out/prog" < "$2" > /dev/null 2>&1 || true
}
for w in $here/workloads/*.clj $here/aot-workloads/*.clj; do
  input=${w%.clj}.in; [ -f "$input" ] || input=/dev/null
  profile "$w" "$input" $out/cg-w
done
while read -r f; do profile "$f" /dev/null $out/cg-c; done < $out/train.txt
echo "profiles: workloads $(ls $out/cg-w | wc -l), corpus-train $(ls $out/cg-c | wc -l)"
python3 $here/hot_text.py "$out/prog" $out/W.ld $out/cg-w/*.cg
python3 $here/hot_text.py "$out/prog" $out/WC.ld $out/cg-w/*.cg $out/cg-c/*.cg
for order in W WC committed; do
  cp $out/$order.ld $ws/hot-text-aot.ld
  (cd $ws && cargo build --release --locked -p clojure-rs > /dev/null 2>&1 && cargo rustc --release --locked -p clj-rt --crate-type staticlib > /dev/null 2>&1)
  b=$out/bin-$order; mkdir -p $b; cp $ws/target/release/clojure-rs $ws/target/release/libcljrt.a $b/
  { cat $out/test.txt | sed 's/^/test /'; ls $here/workloads/*.clj | sed 's/^/work /'; } | while read -r group f; do
    k=$(basename "$f" .clj)
    "$b/clojure-rs" build "$f" -o "$b/$k" > /dev/null 2>&1 || continue
    best=""
    for r in 1 2 3; do
      /usr/bin/time -f '%M' -o "$b/$k.t" "$b/$k" < /dev/null > /dev/null 2>&1 || true
      m=$(tail -1 "$b/$k.t")
      if [ -z "$best" ] || [ "$m" -lt "$best" ]; then best=$m; fi
    done
    echo "$order $group $k $best" >> $out/rss.txt
    rm -f "$b/$k"
  done
  echo "== $order: $(grep -c "^$order " $out/rss.txt) programs"
done
python3 - <<'PY'
import collections, statistics
rows = [l.split() for l in open('/tmp/hold/rss.txt')]
by = collections.defaultdict(dict)
for order, group, k, m in rows:
    by[(order, group)][k] = int(m)
for key, d in sorted(by.items()):
    print(f'{key[0]:9s} {key[1]:5s} n={len(d):3d} median {statistics.median(d.values()):7.0f} KB')
for group in ('test', 'work'):
    base = by[('W', group)]
    for order in ('WC', 'committed'):
        d = by.get((order, group), {})
        common = [k for k in d if k in base]
        if common:
            delta = sorted(d[k] - base[k] for k in common)
            print(f'{order} vs W on {group}: median {statistics.median(delta):+.0f} p10 {delta[len(delta)//10]:+d} p90 {delta[9*len(delta)//10]:+d} (n={len(common)})')
PY
