#!/usr/bin/env bash
# Probe (throwaway): rebuild hot-text.ld for the #254 tree, and compare
# peak RSS of the training workloads linked with the old and new script.
set -euo pipefail
root=$(pwd)
mkdir -p /tmp/out
w=$root/scripts/bench/hot-text/workloads
(cd rust && cargo build --release --locked --no-default-features)
cp rust/target/release/clojure-rs /tmp/out/old-bin
scripts/bench/hot-text/run.sh /tmp/out/hot-text.ld
cp /tmp/out/hot-text.ld rust/hot-text.ld
(cd rust && touch build.rs && cargo build --release --locked --no-default-features)
cp rust/target/release/clojure-rs /tmp/out/new-bin
echo "workload old-rss-kb new-rss-kb"
for f in "$w"/*.clj; do
  o=$( { /usr/bin/time -f '%M' /tmp/out/old-bin run "$f" > /dev/null; } 2>&1 | tail -1 )
  n=$( { /usr/bin/time -f '%M' /tmp/out/new-bin run "$f" > /dev/null; } 2>&1 | tail -1 )
  echo "$(basename "$f") $o $n"
done
rm -f /tmp/out/old-bin /tmp/out/new-bin
diff <(git show HEAD:rust/hot-text.ld) /tmp/out/hot-text.ld | head -40 || true
