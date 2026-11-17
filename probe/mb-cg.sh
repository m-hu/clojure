#!/usr/bin/env bash
# Probe (throwaway, #271): exact per-iteration instruction costs of loop
# idioms (probe/mb/*.clj, n from stdin) under callgrind, by function:
# self cost at n2 minus n1, divided by n2 - n1.
set -euo pipefail
root=$(pwd)
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q valgrind > /dev/null
(cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > /tmp/build.log 2>&1
c=$root/rust/target/release/clojure-rs
mkdir -p /tmp/mb
echo 1000 > /tmp/n1; echo 5000 > /tmp/n2
for f in probe/mb/*.clj; do
  k=$(basename "$f" .clj)
  "$c" build "$f" -o "/tmp/mb/$k" > /dev/null 2>&1 || { echo "$k build-fail"; continue; }
  for n in 1 2; do
    valgrind --tool=callgrind --dump-instr=no --callgrind-out-file="/tmp/mb/$k.$n.cg" "/tmp/mb/$k" < /tmp/n$n > /dev/null 2>&1 || true
  done
  python3 probe/mb-delta.py "/tmp/mb/$k.1.cg" "/tmp/mb/$k.2.cg" 4000 "$k"
done | tee /tmp/mb/summary.txt
