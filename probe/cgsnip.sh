#!/usr/bin/env bash
# Probe (throwaway, #271): where the AOT spends Linux instructions on
# losing snippet programs (public code, synthetic inputs). Callgrind of
# each program built by the PR head's compiler with symbols kept.
set -eu
out=/tmp/cg; mkdir -p $out
ref=$(sed -n 2p probe/rss-refs)
git checkout -q $ref -- rust scripts
(cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > $out/build.log 2>&1 || { tail -30 $out/build.log; exit 1; }
cc=$(pwd)/rust/target/release/clojure-rs
git checkout -q HEAD -- rust scripts
cd probe/snip
for f in *.clj; do
  s=${f%.clj}
  $cc build --keep-symbols $f -o $out/$s > $out/$s.blog 2>&1 || { echo "$s: build failed"; tail -3 $out/$s.blog; continue; }
  valgrind --tool=callgrind --callgrind-out-file=$out/$s.cg $out/$s < $s.in > $out/$s.out 2> $out/$s.err || true
done
cd $out
for f in *.cg; do
  s=${f%.cg}
  tot=$(callgrind_annotate --threshold=100 $f 2>/dev/null | awk '/PROGRAM TOTALS/ {gsub(",", "", $1); print $1}')
  echo "=== $s total $tot  output: $(head -c 60 $s.out | tr '\n' '|')"
  echo "  -- self"
  callgrind_annotate --inclusive=no --threshold=100 $f > $s.self 2>/dev/null || true
  grep -E '^ *[0-9,]+ \(' $s.self | sed -n 2,16p | sed -E 's/ \[[^]]*\]$//' | cut -c1-170 || true
  echo "  -- inclusive"
  callgrind_annotate --inclusive=yes --threshold=100 $f > $s.incl 2>/dev/null || true
  grep -E '^ *[0-9,]+ \(' $s.incl | sed -n 2,30p | sed -E 's/ \[[^]]*\]$//' | cut -c1-170 || true
done
