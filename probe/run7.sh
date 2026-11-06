#!/usr/bin/env bash
# Probe v7 (#226): final branch code — regenerate hot-text.ld, build
# with it exactly as the gate does, measure RSS + Pin vs CPython.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
R=$ROOT/rust; PR=$ROOT/probe; W=$ROOT/scripts/bench/hot-text/workloads; V=/tmp/v
mkdir -p $V /tmp/out
cp $R/build.rs.real $R/build.rs
rm -f $R/hot-text.ld
echo "== generate hot-text.ld"
bash $ROOT/scripts/bench/hot-text/run.sh /tmp/out/hot-text.ld 2>&1 | grep -vE '^\s+Compiling' | tail -2
cp $R/target/release/clojure-rs $V/plain
cp /tmp/out/hot-text.ld $R/hot-text.ld
echo "== gate build (cargo build --release --locked --no-default-features)"
(cd $R && cargo build --release --locked --no-default-features 2>&1 | grep -E '^error|^warning|Finished' | tail -3)
cp $R/target/release/clojure-rs $V/hot
readelf -S -W $V/hot | grep -E ' \.text' | awk '{print "  "$2, "0x"$6}'
med() { sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
rss() { local f=/tmp/rss.out; : > $f; for k in 1 2 3 4 5; do /usr/bin/time -a -o $f -f %M "$@" > /dev/null 2>&1; done; med < $f; }
echo "== RSS (median of 5, KB)"
for w in $(cd $W && ls *.clj | sed 's/\.clj$//'); do printf '%-14s %8s %8s\n' $w $(rss $V/plain run $W/$w.clj) $(rss $V/hot run $W/$w.clj); done
PW=$PR/wl
printf 'python3.11:'; for i in 0 1 2 3 4 5 6 7 8 9 10 11; do printf ' s%s=%s' $i $(rss python3.11 $PW/s$i.py); done; echo
echo "== Pin instruction counts (hot vs python3.11)"
bash $PR/pin/setup-instrumentation.sh 2>/dev/null | grep '^INSTRUMENTATION' > /tmp/pin.env; . /tmp/pin.env
pin_count() { local o; o=$(mktemp); "$INSTRUMENTATION_BIN" -t "$INSTRUMENTATION_TOOL" -o $o -- "$@" > /dev/null 2>&1; awk '/^Count/{print $2}' $o; rm -f $o; }
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
  printf 's%-3s rs=%-12s py=%s\n' $i $(pin_count $V/hot run $PW/s$i.clj) $(pin_count setarch $(uname -m) -R python3.11 $PW/s$i.py)
done
