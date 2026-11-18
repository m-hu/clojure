#!/usr/bin/env bash
# Probe (throwaway, #271): resident memory at exit (gdb at exit_group,
# /proc/PID/smaps) on empty stdin for the same programs and runtime under
# four AOT hot-text orders: A committed, R the CI regeneration for
# f9a16ff5, R2 regen-aot.sh rerun here, C the earlier workloads-only probe.
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
cp rust/hot-text-aot.ld probe/order-A.ld
echo "== regen-aot.sh at this revision"
bash scripts/bench/hot-text/regen-aot.sh rust probe/order-R2.ld
(cd rust && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > /dev/null 2>&1
echo "nm sizes rows: $(nm -S --size-sort -t d rust/target/release/libcljrt.a 2>/dev/null | wc -l)"
echo "nm on archive (first lines):"; nm -S --size-sort -t d rust/target/release/libcljrt.a 2>&1 | tail -3
for tag in A R R2 C; do
  echo "order $tag: $(grep -c '^    \*(' probe/order-$tag.ld) globs"
  cp "probe/order-$tag.ld" rust/hot-text-aot.ld
  phase "$tag"
done
cp probe/order-A.ld rust/hot-text-aot.ld
