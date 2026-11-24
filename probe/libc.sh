#!/usr/bin/env bash
# Probe (throwaway, #271): does packing the glibc members the workloads
# run into .text.hot cut peak RSS? A = CI order (regen-aot.sh), B = A plus
# libc/libgcc members. Measured on snippet programs, empty stdin (no
# inputs); trained only on scripts/bench/hot-text workloads + corpus.
set -euo pipefail
root=$(pwd)
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q valgrind time > /dev/null
ws=$root/rust; here=$root/scripts/bench/hot-text; out=/tmp/libc; mkdir -p $out/fns
bash $here/regen-aot.sh "$ws" $out/A.ld > /tmp/regen.log 2>&1 || { tail -30 /tmp/regen.log; exit 1; }
cp $out/A.ld $ws/hot-text-aot.ld
build() { (cd $ws && cargo build --release --locked -p clojure-rs > /dev/null 2>&1 && cargo rustc --release --locked -p clj-rt --crate-type staticlib > /dev/null 2>&1); }
build; c=$ws/target/release/clojure-rs
for w in $here/workloads/*.clj $here/aot-workloads/*.clj docs/spec/i259-native-compiler/corpus/*.clj; do
  input=${w%.clj}.in; [ -f "$input" ] || input=/dev/null
  k=$(basename $w .clj)
  "$c" build "$w" -o $out/prog > /dev/null 2>&1 || continue
  valgrind --tool=callgrind --dump-instr=no --demangle=no --callgrind-out-file=/tmp/cg.out $out/prog < "$input" > /dev/null 2>&1 || true
  python3 probe/fns.py /tmp/cg.out > $out/fns/$k.txt
done
python3 probe/libc_members.py $out/A.ld $out/B.ld $out/fns/*.txt
sed -n '1,12p' $out/B.ld | cut -c1-100
find snip/src -name '*.clj' | sort | awk 'NR % 4 == 0' > $out/progs.txt
for order in A B; do
  cp $out/$order.ld $ws/hot-text-aot.ld; build
  b=$out/bin-$order; mkdir -p $b; cp $ws/target/release/clojure-rs $ws/target/release/libcljrt.a $b/
  while read -r f; do
    k=$(echo "${f%.clj}" | sed 's|^snip/src/||; s|/|_|g')
    "$b/clojure-rs" build "$f" -o "$b/$k" > /dev/null 2>&1 || continue
    best=""
    for r in 1 2 3; do
      /usr/bin/time -f '%M' -o "$b/$k.t" "$b/$k" < /dev/null > /dev/null 2>&1 || true
      m=$(tail -1 "$b/$k.t"); if [ -z "$best" ] || [ "$m" -lt "$best" ]; then best=$m; fi
    done
    echo "$order $k $best" >> $out/rss.txt; rm -f "$b/$k"
  done < $out/progs.txt
done
python3 - <<'PY'
import collections, statistics
by = collections.defaultdict(dict)
for l in open('/tmp/libc/rss.txt'):
    o, k, m = l.split(); by[o][k] = int(m)
for o, d in sorted(by.items()):
    print(o, 'n', len(d), 'median', statistics.median(d.values()))
common = [k for k in by['A'] if k in by['B']]
delta = sorted(by['B'][k] - by['A'][k] for k in common)
print(f'B - A: median {statistics.median(delta):+.0f} p10 {delta[len(delta)//10]:+d} p90 {delta[9*len(delta)//10]:+d} n={len(common)}')
PY
