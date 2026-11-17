#!/usr/bin/env bash
# Probe (throwaway, #271): callgrind on empty stdin vs a small synthetic
# input for snippets whose cheapest case carries a large fixed cost.
set -euo pipefail
root=$(pwd)
(cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > /tmp/build.log 2>&1
c=$root/rust/target/release/clojure-rs
mkdir -p /tmp/prof /tmp/bin /tmp/in
printf '3\n0 0\n2 0\n1 1\n' > /tmp/in/computational_geometry_triangulation
printf '1\n5 3 0 0 7 0 0 0 0\n6 0 0 1 9 5 0 0 0\n0 9 8 0 0 0 0 6 0\n8 0 0 0 6 0 0 0 3\n4 0 0 8 0 3 0 0 1\n7 0 0 0 2 0 0 0 6\n0 6 0 0 0 0 2 8 0\n0 0 0 4 1 9 0 0 5\n0 0 0 0 8 0 0 7 9\n' > /tmp/in/search_backtracking
printf '2\n(a|b)* 5\nab* 3\n' > /tmp/in/string_related_automata
for s in computational_geometry/triangulation search/backtracking string_related/automata; do
  f=$(ls "$root"/snip/src/$s/*.clj | head -1); n=$(echo "$s" | tr '/' '_')
  "$c" build "$f" -o /tmp/bin/$n > /dev/null 2>&1 || { echo "$s build-fail"; continue; }
  valgrind --tool=callgrind --callgrind-out-file="/tmp/prof/$n.empty.cg" /tmp/bin/$n < /dev/null > /dev/null 2>&1 || true
  valgrind --tool=callgrind --callgrind-out-file="/tmp/prof/$n.input.cg" /tmp/bin/$n < /tmp/in/$n > /tmp/prof/$n.out 2>/dev/null || true
  e=$(grep -E '^(summary|totals):' /tmp/prof/$n.empty.cg | head -1 | awk '{print $2}')
  i=$(grep -E '^(summary|totals):' /tmp/prof/$n.input.cg | head -1 | awk '{print $2}')
  echo "$s empty=$e input=$i delta=$((i-e)) out=$(head -c 60 /tmp/prof/$n.out | tr '\n' '|')"
done
