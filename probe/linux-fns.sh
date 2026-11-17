#!/usr/bin/env bash
# Probe (throwaway, #271): each snippet's executed functions on Linux
# (callgrind, empty stdin, blank hot-text order) plus every function
# symbol any snippet binary links, for the hot-text order generator.
set -euo pipefail
root=$(pwd)
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q valgrind > /dev/null
printf '%s\n' '/* placeholder */' 'SECTIONS' '{' '  .text.hot :' '  {' '  }' '}' 'INSERT BEFORE .text;' > rust/hot-text-aot.ld
(cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > /tmp/build.log 2>&1
c=$root/rust/target/release/clojure-rs
mkdir -p /tmp/fns /tmp/b
: > /tmp/allsyms.txt
run_one() {  # name source input
  "$c" build "$2" -o /tmp/b/prog > /dev/null 2>&1 || { echo "$1 build-fail"; return; }
  valgrind --tool=callgrind --demangle=no --dump-instr=no --callgrind-out-file=/tmp/b/cg /tmp/b/prog < "$3" > /dev/null 2>&1 || true
  python3 probe/cg-fns.py /tmp/b/cg prog > "/tmp/fns/$1.txt"
  nm --defined-only /tmp/b/prog | awk '$2 ~ /^[tTwW]$/ {print $3}' >> /tmp/allsyms.txt
  echo "$1 $(wc -l < /tmp/fns/$1.txt)"
}
for w in scripts/bench/hot-text/workloads/*.clj scripts/bench/hot-text/aot-workloads/*.clj; do
  n=$(basename "$w" .clj); in=${w%.clj}.in; [ -f "$in" ] || in=/dev/null
  run_one "wl_$n" "$w" "$in"
done
find snip/src -name '*.clj' | sort | while read -r f; do
  rel=${f#snip/src/}; n=$(echo "${rel%.clj}" | sed 's#/#__#g')
  run_one "$n" "$f" /dev/null
done
sort -u /tmp/allsyms.txt -o /tmp/allsyms.txt
tar czf /tmp/linux-fns.tgz -C /tmp fns allsyms.txt
