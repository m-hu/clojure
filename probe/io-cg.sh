#!/usr/bin/env bash
# Probe (throwaway, #271): Linux callgrind of the snippet I/O idiom steps
# (probe/io/p0..p5, tiny synthetic input), self cost by function, each
# step against the previous one.
set -euo pipefail
root=$(pwd)
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q valgrind > /dev/null
(cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > /tmp/build.log 2>&1
c=$root/rust/target/release/clojure-rs
mkdir -p /tmp/io
for p in p0 p1 p2 p3 p4 p5; do
  "$c" build "probe/io/$p.clj" -o "/tmp/io/$p" > /dev/null 2>&1 || { echo "$p build-fail"; continue; }
  valgrind --tool=callgrind --dump-instr=no --callgrind-out-file="/tmp/io/$p.cg" "/tmp/io/$p" < probe/io/in.txt > /dev/null 2>&1 || true
  echo "$p total $(grep -E '^(summary|totals):' /tmp/io/$p.cg | head -1)"
done
prev=p0
for p in p1 p2 p3 p4 p5; do
  python3 probe/mb-delta.py "/tmp/io/$prev.cg" "/tmp/io/$p.cg" 1 "$p-vs-$prev"
  prev=$p
done | tee /tmp/io/summary.txt
