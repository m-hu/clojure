#!/usr/bin/env bash
# Probe (throwaway, #271): regen-aot.sh with --archive members links and runs.
set -euo pipefail
root=$(pwd); ws=$root/rust; out=/tmp/libc; mkdir -p $out
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q valgrind > /dev/null
bash scripts/bench/hot-text/regen-aot.sh "$ws" $out/A.ld 2>&1 | tail -5
grep -c 'libc.a:' $out/A.ld; sed -n 1,10p $out/A.ld
cp $out/A.ld $ws/hot-text-aot.ld
(cd $ws && cargo build --release --locked -p clojure-rs > /dev/null 2>&1 && cargo rustc --release --locked -p clj-rt --crate-type staticlib > /dev/null 2>&1)
c=$ws/target/release/clojure-rs; ok=0; bad=0
for f in docs/spec/i259-native-compiler/corpus/*.clj; do
  "$c" build "$f" -o $out/p > /dev/null 2>&1 || continue
  if "$c" run "$f" < /dev/null > $out/r.txt 2>&1; [ "$($out/p < /dev/null 2>&1)" = "$(cat $out/r.txt)" ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "DIFF $f"; fi
done
echo "corpus parity ok=$ok bad=$bad"
nm $out/p | grep -c . ; readelf -S $out/p | grep -A1 text.hot
