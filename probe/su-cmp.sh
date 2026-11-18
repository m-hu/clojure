#!/usr/bin/env bash
# Probe (throwaway, #271): exact Linux startup (callgrind, empty stdin)
# of the baseline f469938a vs this tree, per script.
set -euo pipefail
root=$(pwd)
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q valgrind > /dev/null
build() { (cd "$1/rust" && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > /tmp/build-$2.log 2>&1; }
git fetch -q --depth 50 origin f469938a 2>/dev/null || git fetch -q origin
git worktree add -q /tmp/base f469938a
build /tmp/base base
build "$root" new
mkdir -p /tmp/su
printf '(println 1)\n' > /tmp/trivial.clj
while read -r s; do
  [ -z "$s" ] && continue
  n=$(echo "$s" | tr '/' '_')
  line="$s"
  for v in base new; do
    c=$root/rust/target/release/clojure-rs; [ $v = base ] && c=/tmp/base/rust/target/release/clojure-rs
    "$c" build "snip/src/$s.clj" -o /tmp/su/p > /dev/null 2>&1 || { line="$line $v:fail"; continue; }
    valgrind --tool=callgrind --dump-instr=no --callgrind-out-file=/tmp/su/$n.$v.cg /tmp/su/p < /dev/null > /dev/null 2>&1 || true
    t=$(grep -E '^(summary|totals):' /tmp/su/$n.$v.cg | head -1 | awk '{print $2}')
    line="$line $v:$t"
  done
  echo "$line"
done < probe/scripts.txt | tee /tmp/su/totals.txt
