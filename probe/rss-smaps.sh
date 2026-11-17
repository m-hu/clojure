#!/usr/bin/env bash
# Probe (throwaway, #271): resident memory by mapping at exit (gdb stops
# the process at exit_group and reads /proc/PID/smaps) for `(println 1)`
# and the scripts in probe/rss-scripts.txt, on empty stdin.
set -euo pipefail
root=$(pwd)
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q gdb > /dev/null
(cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > /tmp/build.log 2>&1
c=$root/rust/target/release/clojure-rs
mkdir -p /tmp/rss /tmp/bin
run_one() {
  gdb -q -batch -ex 'catch syscall exit_group' -ex 'run < /dev/null > /dev/null' \
    -ex 'python print(open("/proc/%d/smaps" % gdb.selected_inferior().pid).read())' \
    "$2" > "/tmp/rss/$1.smaps" 2> /dev/null || true
  python3 probe/smaps.py "/tmp/rss/$1.smaps" "$1" "$2"
}
printf '(println 1)\n' > /tmp/trivial.clj
"$c" build /tmp/trivial.clj -o /tmp/bin/trivial > /dev/null 2>&1
run_one trivial /tmp/bin/trivial
while read -r s; do
  [ -z "$s" ] && continue
  n=$(echo "$s" | tr '/' '_')
  "$c" build "$root/snip/src/$s.clj" -o "/tmp/bin/$n" > /dev/null 2>&1 || { echo "$s build-fail"; continue; }
  run_one "$n" "/tmp/bin/$n"
done < probe/rss-scripts.txt
