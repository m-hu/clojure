#!/usr/bin/env bash
# Probe (throwaway, #271): where the near-miss scripts' startup goes on
# EMPTY stdin (no inputs), minus a trivial program: callgrind self cost by
# source line (line tables) and allocation call sites, and DHAT allocation
# sites with their blocks and bytes (total and live at exit).
set -euo pipefail
root=$(pwd)
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q valgrind > /dev/null
export CARGO_PROFILE_RELEASE_DEBUG=line-tables-only
(cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > /tmp/build.log 2>&1 || { tail -40 /tmp/build.log; exit 1; }
c=$root/rust/target/release/clojure-rs
mkdir -p /tmp/prof /tmp/dhat /tmp/bin
printf '(println 1)\n' > /tmp/trivial.clj
run() {
  "$c" build "$2" -o "/tmp/bin/$1" > /dev/null 2>&1 || { echo "$1 build-fail"; return 0; }
  valgrind --tool=callgrind --dump-instr=no --dump-line=yes --compress-pos=no \
    --callgrind-out-file="/tmp/prof/$1.cg" "/tmp/bin/$1" < /dev/null > /dev/null 2>&1 || true
  valgrind --tool=dhat --soname-synonyms=somalloc=NONE --fullpath-after=crates/ \
    --dhat-out-file="/tmp/dhat/$1.json" "/tmp/bin/$1" < /dev/null > /dev/null 2> "/tmp/dhat/$1.err" || tail -3 "/tmp/dhat/$1.err"
}
run trivial /tmp/trivial.clj
while read -r s; do
  [ -z "$s" ] && continue
  run "$(echo "$s" | tr '/' '_')" "$root/snip/src/$s.clj"
done < probe/scripts.txt
ls /tmp/prof | wc -l
ls /tmp/dhat/*.json 2> /dev/null | wc -l
python3 probe/cgline.py /tmp/prof || echo "cgline failed"
python3 probe/dhatsum.py /tmp/dhat || echo "dhatsum failed"
tar czf /tmp/raw.tgz -C /tmp prof dhat
