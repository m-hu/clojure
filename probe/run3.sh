#!/usr/bin/env bash
# Linux RSS / instruction probe v3 for #226 (throwaway branch).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
R=$ROOT/rust; PR=$ROOT/probe; W=$PR/wl; V=/tmp/v
mkdir -p $V /tmp/cg /tmp/pm /tmp/out
cp $PR/build.rs $R/build.rs
MAIN=$R/src/bin/clojure-rs/main.rs
cp $MAIN /tmp/main.rs.orig; cp $R/Cargo.toml /tmp/Cargo.toml.orig
WL="0 1 2 3 4 5 6 7 8 9 10 11"

build() { # name linkargs...
  local n=$1; shift
  echo "== build $n: $*"
  local t0=$SECONDS
  ( cd $R && PROBE_LINK_ARGS="$*" cargo build --release --locked --no-default-features 2>&1 | grep -E '^error|^warning: |Finished' | tail -4 )
  cp $R/target/release/clojure-rs $V/$n 2>/dev/null && echo "  size $(stat -c %s $V/$n) built in $((SECONDS - t0))s" || echo "  BUILD FAILED"
}
sysalloc() { cp /tmp/main.rs.orig $MAIN; sed -i '/^\/\/\/ mimalloc (#226)/,/^static GLOBAL/d' $MAIN; }
mimalloc() { cp /tmp/main.rs.orig $MAIN; }

sysalloc
build sys-pie
build sys-nopie -no-pie
build sys-relr -Wl,-z,pack-relative-relocs
build sys-nopie-icf -no-pie -Wl,--icf=all
mimalloc
sed -i 's/^mimalloc = { version = "0.1.52", default-features = false }/mimalloc = { version = "0.1.52", default-features = false, features = ["no_thp"] }/' $R/Cargo.toml
grep -n '^mimalloc' $R/Cargo.toml
build mi-nothp-nopie -no-pie
cp /tmp/Cargo.toml.orig $R/Cargo.toml
build mi-default
sysalloc

echo "== callgrind (sys-nopie)"
for i in $WL; do
  valgrind --tool=callgrind --demangle=no --dump-instr=no --callgrind-out-file=/tmp/cg/s$i.cg $V/sys-nopie run $W/s$i.clj > /dev/null 2> /tmp/cg/s$i.err || tail -3 /tmp/cg/s$i.err
done
python3 $PR/cgfuncs.py $V/sys-nopie /tmp/out/order /tmp/cg/*.cg
python3 $PR/cgfuncs.py $V/sys-nopie /tmp/out/order0 /tmp/cg/s0.cg
head -c 600 /tmp/out/order.ld; echo
build sys-nopie-ld -no-pie -Wl,-T,/tmp/out/order.ld
build sys-nopie-sym -no-pie -Wl,--symbol-ordering-file=/tmp/out/order.order -Wl,--no-warn-symbol-ordering
build sys-pie-ld -Wl,-T,/tmp/out/order.ld
for b in sys-nopie sys-nopie-ld sys-nopie-sym; do
  [ -x $V/$b ] && echo "  $b: $(readelf -S -W $V/$b | grep -E ' \.text' | awk '{print $2, "size=0x"$6}' | tr '\n' ' ')"
done

med() { sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
rss() { local f=/tmp/rss.out; : > $f; for k in 1 2 3 4 5; do /usr/bin/time -a -o $f -f %M "$@" > /dev/null 2>&1; done; med < $f; }
rss_table() {
  printf '%-16s' "maxrss_kb"; for i in $WL; do printf '%7s' s$i; done; echo
  for b in "$@"; do
    [ -x $V/$b ] || continue
    printf '%-16s' $b; for i in $WL; do printf '%7s' $(rss $V/$b run $W/s$i.clj); done; echo
  done
  printf '%-16s' python3.11; for i in $WL; do printf '%7s' $(rss python3.11 $W/s$i.py); done; echo
}
ALL="mi-default mi-nothp-nopie sys-pie sys-relr sys-nopie sys-nopie-icf sys-nopie-ld sys-nopie-sym sys-pie-ld"
echo "== RSS (median of 5, KB)"
rss_table $ALL

for i in $WL; do { cat $W/s$i.clj; echo '(println "@@READY@@") (flush) (read-line)'; } > /tmp/pm/s$i.clj; done
echo "== residency at exit (default fault-around)"
for b in sys-pie sys-nopie sys-nopie-ld sys-nopie-sym; do
  [ -x $V/$b ] || continue
  for i in 0 10; do echo "-- $b s$i"; python3 $PR/pagemap.py $V/$b /tmp/pm/s$i.clj /tmp/out/$b-s$i.syms; done
done

FA=/sys/kernel/debug/fault_around_bytes
sudo mount -t debugfs none /sys/kernel/debug 2> /dev/null
if sudo test -e $FA; then
  echo "== fault_around_bytes was $(sudo cat $FA); now 4096"
  sudo sh -c "echo 4096 > $FA"
  rss_table sys-nopie sys-nopie-ld sys-nopie-sym
  for b in sys-nopie sys-nopie-ld; do echo "-- $b s0 (4K)"; python3 $PR/pagemap.py $V/$b /tmp/pm/s0.clj /tmp/out/$b-s0-4k.syms; done
  sudo sh -c "echo 65536 > $FA"
else
  echo "== no $FA"
fi

echo "== Pin instruction counts"
bash $PR/pin/setup-instrumentation.sh > /tmp/pin.env && . /tmp/pin.env
pin_count() { local o; o=$(mktemp); "$INSTRUMENTATION_BIN" -t "$INSTRUMENTATION_TOOL" -o $o -- "$@" > /dev/null 2>&1; awk '/^Count/{print $2}' $o; rm -f $o; }
printf '%-16s' "pin_ins"; for i in $WL; do printf '%13s' s$i; done; echo
for b in mi-default mi-nothp-nopie sys-nopie sys-pie; do
  [ -x $V/$b ] || continue
  printf '%-16s' $b; for i in $WL; do printf '%13s' $(pin_count $V/$b run $W/s$i.clj); done; echo
done
printf '%-16s' python3.11; for i in $WL; do printf '%13s' $(setarch -R true 2>/dev/null && pin_count setarch -R python3.11 $W/s$i.py || pin_count python3.11 $W/s$i.py); done; echo
