#!/usr/bin/env bash
# Probe (throwaway, #271): resident memory at exit (gdb at exit_group,
# /proc/PID/smaps) on empty stdin, with the committed hot-text order (A)
# and with three alternative orders, over the same programs.
set -euo pipefail
root=$(pwd)
mkdir -p /tmp/rss /tmp/bin
printf '(println 1)\n' > /tmp/trivial.clj
run_one() {  # tag name binary
  gdb -q -batch -ex 'catch syscall exit_group' -ex 'run < /dev/null > /dev/null' \
    -ex 'python print(open("/proc/%d/smaps" % gdb.selected_inferior().pid).read())' \
    "$3" > "/tmp/rss/$2.$1.smaps" 2> /dev/null || true
  python3 probe/smaps.py "/tmp/rss/$2.$1.smaps" "$1 $2" "$3"
}
phase() {  # tag
  (cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > "/tmp/build-$1.log" 2>&1
  c=$root/rust/target/release/clojure-rs
  "$c" build /tmp/trivial.clj -o "/tmp/bin/trivial.$1" > /dev/null 2>&1
  run_one "$1" trivial "/tmp/bin/trivial.$1"
  while read -r s; do
    [ -z "$s" ] && continue
    n=$(echo "$s" | tr '/' '_')
    "$c" build "snip/src/$s.clj" -o "/tmp/bin/$n.$1" > /dev/null 2>&1 || { echo "$1 $n build-fail"; continue; }
    run_one "$1" "$n" "/tmp/bin/$n.$1"
  done < probe/rss-ab-scripts.txt
}
# A: the committed order; B: no order; C: an order trained only on the
# independent workloads; D: the order trained on the snippets.
cp rust/hot-text-aot.ld /tmp/order-A.ld
for tag in A B C D; do
  [ "$tag" = A ] || cp "probe/order-$tag.ld" rust/hot-text-aot.ld
  phase "$tag"
done
