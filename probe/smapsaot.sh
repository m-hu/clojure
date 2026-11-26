#!/usr/bin/env bash
# Probe (throwaway, #271): where an AOT binary's peak RSS goes (smaps by
# mapping, at the end of a run held open by a busy wait), built as CI
# builds it (fresh AOT order from this tree's training set).
set -euo pipefail
out=/tmp/sm; mkdir -p $out
ref=$(sed -n 2p probe/rss-refs)
git checkout -q $ref -- rust scripts
bash scripts/bench/hot-text/regen-aot.sh rust > $out/regen.log 2>&1 || { tail -20 $out/regen.log; exit 1; }
(cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > $out/build.log 2>&1 || { tail -30 $out/build.log; exit 1; }
c=$(pwd)/rust/target/release/clojure-rs
for k in bentley delaunay voronoi beam sudoku; do
  { cat probe/prog/$k.clj; printf '\n%s\n' '(let [t (System/currentTimeMillis)] (loop [] (when (< (- (System/currentTimeMillis) t) 1500) (recur))))'; } > $out/$k-hold.clj
  $c build $out/$k-hold.clj -o $out/$k > /dev/null 2>&1 || { echo "$k build-fail"; continue; }
  size -A $out/$k | awk '$1 ~ /^\.(text|text\.hot|rodata|data|bss|data\.rel\.ro|eh_frame|gcc_except_table)$/ {printf "  %s=%d", $1, $2}'; echo
  $out/$k < probe/prog/$k.in > /dev/null & pid=$!; sleep 1.2
  echo "== $k (pid $pid) RSS by mapping, KB"
  awk '/^[0-9a-f]+-[0-9a-f]+ / {name=$6; if (name=="") name="[anon]"; perm=$2} /^Rss:/ {k=name" "perm; s[k]+=$2} END {for (k in s) if (s[k] > 16) printf "  %7d %s\n", s[k], k}' /proc/$pid/smaps | sort -rn | head -12
  grep -E '^(Rss|Anonymous):' /proc/$pid/smaps_rollup | tr -s ' ' | sed 's/^/  /'
  wait $pid || true
done
