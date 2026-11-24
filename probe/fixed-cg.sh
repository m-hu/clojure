#!/usr/bin/env bash
# Probe (throwaway, #271): callgrind on empty stdin for the scripts whose
# aot-ocaml INS losses are all within 300K of the margin (per-program
# fixed costs), plus the `(println 1)` floor; costs aggregated by function.
set -euo pipefail
root=$(pwd)
(cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > /tmp/build.log 2>&1
c=$root/rust/target/release/clojure-rs
mkdir -p /tmp/prof /tmp/bin
printf '(println 1)\n' > /tmp/trivial.clj
"$c" build /tmp/trivial.clj -o /tmp/bin/trivial > /dev/null 2>&1
valgrind --tool=callgrind --callgrind-out-file=/tmp/prof/trivial.cg /tmp/bin/trivial < /dev/null > /dev/null 2>&1 || true
while read -r s; do
  [ -z "$s" ] && continue
  f="$root/snip/src/$s.clj"
  n=$(echo "$s" | tr '/' '_')
  "$c" build "$f" -o "/tmp/bin/$n" > /dev/null 2>&1 || { echo "$s build-fail"; continue; }
  valgrind --tool=callgrind --callgrind-out-file="/tmp/prof/$n.cg" "/tmp/bin/$n" < /dev/null > /dev/null 2>&1 || true
done < probe/scripts.txt
python3 probe/agg.py /tmp/prof | tee /tmp/prof/agg.txt
