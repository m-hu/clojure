#!/usr/bin/env bash
# Probe v4 (#226): executed-function layout + heap sites + rodata labels.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
R=$ROOT/rust; PR=$ROOT/probe; W=$PR/wl; V=/tmp/v
mkdir -p $V /tmp/cg /tmp/pm /tmp/out
cp $PR/build.rs $R/build.rs
MAIN=$R/src/bin/clojure-rs/main.rs
sed -i '/^\/\/\/ mimalloc (#226)/,/^static GLOBAL/d' $MAIN
WL="0 1 2 3 4 5 6 7 8 9 10 11"
build() { local n=$1; shift; echo "== build $n: $*"; local t0=$SECONDS
  ( cd $R && PROBE_LINK_ARGS="$*" cargo build --release --locked --no-default-features 2>&1 | grep -E '^error|^warning: |Finished|symbol' | tail -6 )
  cp $R/target/release/clojure-rs $V/$n 2>/dev/null && echo "  size $(stat -c %s $V/$n) built in $((SECONDS - t0))s" || echo "  BUILD FAILED"; }
build sys-nopie -no-pie
echo "== callgrind (sys-nopie)"
for i in $WL; do
  valgrind --tool=callgrind --demangle=no --dump-instr=no --callgrind-out-file=/tmp/cg/s$i.cg $V/sys-nopie run $W/s$i.clj > /dev/null 2> /tmp/cg/s$i.err || tail -3 /tmp/cg/s$i.err
done
grep -m3 '^ob=' /tmp/cg/s0.cg
echo "-- all workloads"; python3 $PR/cgfuncs.py $V/sys-nopie /tmp/out/order /tmp/cg/*.cg
echo "-- s0 only"; python3 $PR/cgfuncs.py $V/sys-nopie /tmp/out/order0 /tmp/cg/s0.cg | head -3
wc -l /tmp/out/order.order /tmp/out/order.ld
echo "== DHAT s0 / s10"
for i in 0 10; do
  valgrind --tool=dhat --dhat-out-file=/tmp/out/dhat-s$i.json $V/sys-nopie run $W/s$i.clj > /dev/null 2> /tmp/out/dhat-s$i.err
  echo "-- s$i"; python3 $PR/dhat.py /tmp/out/dhat-s$i.json 40
done
for i in $WL; do { cat $W/s$i.clj; echo '(println "@@READY@@") (flush) (read-line)'; } > /tmp/pm/s$i.clj; done
echo "== rodata labels (sys-nopie s0)"
python3 $PR/rolabel.py $V/sys-nopie /tmp/pm/s0.clj $R/crates
build sys-nopie-sym -no-pie -Wl,--symbol-ordering-file=/tmp/out/order.order -Wl,--no-warn-symbol-ordering
build sys-nopie-ld -no-pie -Wl,-T,/tmp/out/order.ld
build sys-nopie-sym0 -no-pie -Wl,--symbol-ordering-file=/tmp/out/order0.order -Wl,--no-warn-symbol-ordering
for b in sys-nopie sys-nopie-sym sys-nopie-ld; do
  [ -x $V/$b ] && echo "  $b: $(readelf -S -W $V/$b | grep -E ' \.text' | awk '{print $2, "size=0x"$6}' | tr '\n' ' ')"
done
med() { sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
rss() { local f=/tmp/rss.out; : > $f; for k in 1 2 3 4 5; do /usr/bin/time -a -o $f -f %M "$@" > /dev/null 2>&1; done; med < $f; }
echo "== RSS (median of 5, KB)"
printf '%-16s' "maxrss_kb"; for i in $WL; do printf '%7s' s$i; done; echo
for b in sys-nopie sys-nopie-sym sys-nopie-ld sys-nopie-sym0; do
  [ -x $V/$b ] || continue
  printf '%-16s' $b; for i in $WL; do printf '%7s' $(rss $V/$b run $W/s$i.clj); done; echo
done
printf '%-16s' python3.11; for i in $WL; do printf '%7s' $(rss python3.11 $W/s$i.py); done; echo
echo "== residency (default fault-around)"
for b in sys-nopie sys-nopie-sym sys-nopie-ld; do
  [ -x $V/$b ] || continue
  for i in 0 10; do echo "-- $b s$i"; python3 $PR/pagemap.py $V/$b /tmp/pm/s$i.clj /tmp/out/$b-s$i.syms | grep -E 'binary resident|\.text |\.rodata |symbols on|other mappings'; done
done
