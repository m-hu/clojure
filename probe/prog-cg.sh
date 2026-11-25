#!/usr/bin/env bash
# Probe (throwaway, #271): per-run Linux instruction cost BY FUNCTION of
# public snippet programs on synthetic inputs (probe/prog/*.in, written
# for this probe): callgrind self cost of the program looped 6x minus
# looped 1x, divided by 5 (startup cancels out).
set -euo pipefail
root=$(pwd)
(cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > /tmp/build.log 2>&1 || { tail -50 /tmp/build.log; exit 1; }
c=$root/rust/target/release/clojure-rs
mkdir -p /tmp/pg
for f in probe/prog/*.clj; do
  k=$(basename "$f" .clj)
  for n in 1 6; do
    python3 probe/loopify.py "$f" "/tmp/pg/$k.$n.clj" $n
    "$c" build "/tmp/pg/$k.$n.clj" -o "/tmp/pg/$k.$n" > /dev/null 2>&1 || { echo "$k build-fail"; continue 2; }
    valgrind --tool=callgrind --dump-instr=no --callgrind-out-file="/tmp/pg/$k.$n.cg" "/tmp/pg/$k.$n" < "probe/prog/$k.in" > /dev/null 2>&1 || true
  done
  python3 probe/prog-delta.py "/tmp/pg/$k.1.cg" "/tmp/pg/$k.6.cg" 5 "$k"
done | tee /tmp/pg/summary.txt
python3 probe/prog-agg.py /tmp/pg/summary.txt | tee /tmp/pg/agg.txt
