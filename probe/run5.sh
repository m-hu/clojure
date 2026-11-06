#!/usr/bin/env bash
# Probe v5 (#226): the branch's real build.rs + allocator + hot-text generator.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
R=$ROOT/rust; PR=$ROOT/probe; W=$ROOT/scripts/bench/hot-text/workloads; V=/tmp/v
mkdir -p $V /tmp/out /tmp/pm
cp $R/build.rs.real $R/build.rs
rm -f $R/hot-text.ld
echo "== generate hot-text.ld (build without it, callgrind the workloads)"
t0=$SECONDS
bash $ROOT/scripts/bench/hot-text/run.sh /tmp/out/hot-text.ld 2>&1 | grep -vE '^\s+Compiling' | tail -5
echo "  generated in $((SECONDS - t0))s"; cp $R/target/release/clojure-rs $V/plain
wc -l /tmp/out/hot-text.ld; head -12 /tmp/out/hot-text.ld | cut -c1-160
cp /tmp/out/hot-text.ld $R/hot-text.ld
echo "== build with hot-text.ld (lld)"
t0=$SECONDS; (cd $R && cargo build --release --locked --no-default-features 2>&1 | grep -E '^error|^warning|Finished' | tail -3); echo "  built in $((SECONDS - t0))s"
cp $R/target/release/clojure-rs $V/hot
echo "== build with hot-text.ld (GNU ld)"
t0=$SECONDS; (cd $R && RUSTFLAGS="-C link-arg=-fuse-ld=bfd" cargo build --release --locked --no-default-features --target-dir /tmp/t-bfd 2>&1 | grep -E '^error|^warning|Finished|undefined|ld:' | tail -5); echo "  built in $((SECONDS - t0))s"
cp /tmp/t-bfd/release/clojure-rs $V/hot-bfd 2>/dev/null || echo "  bfd build missing"
for b in plain hot hot-bfd; do [ -x $V/$b ] && echo "  $b: $(stat -c %s $V/$b) bytes; $(file -b $V/$b | cut -c1-60); $(readelf -S -W $V/$b | grep -E ' \.text' | awk '{print $2, "0x"$6}' | tr '\n' ' ')"; done
med() { sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
rss() { local f=/tmp/rss.out; : > $f; for k in 1 2 3 4 5; do /usr/bin/time -a -o $f -f %M "$@" > /dev/null 2>&1; done; med < $f; }
WL=$(cd $W && ls *.clj | sed 's/\.clj$//')
echo "== RSS (median of 5, KB)"
printf '%-14s' run; for b in plain hot hot-bfd; do printf '%9s' $b; done; echo
for w in $WL; do
  printf '%-14s' $w; for b in plain hot hot-bfd; do [ -x $V/$b ] && printf '%9s' $(rss $V/$b run $W/$w.clj) || printf '%9s' -; done; echo
done
PW=$PR/wl
printf '%-14s' "python3.11:"; for i in 0 1 2 3 4 5 6 7 8 9 10 11; do printf ' s%s=%s' $i $(rss python3.11 $PW/s$i.py); done; echo
echo "== residency (hot, trivial)"
{ cat $W/trivial.clj; echo '(println "@@READY@@") (flush) (read-line)'; } > /tmp/pm/t.clj
python3 $PR/pagemap.py $V/hot /tmp/pm/t.clj /tmp/out/hot-trivial.syms | grep -E 'binary resident|\.text|\.rodata |symbols on|other mappings'
echo "== Pin instruction counts"
bash $PR/pin/setup-instrumentation.sh 2>/dev/null | grep '^INSTRUMENTATION' > /tmp/pin.env; . /tmp/pin.env
pin_count() { local o; o=$(mktemp); "$INSTRUMENTATION_BIN" -t "$INSTRUMENTATION_TOOL" -o $o -- "$@" > /dev/null 2>&1; awk '/^Count/{print $2}' $o; rm -f $o; }
for w in trivial map-assoc str-concat lazy-pipeline split-parse frequencies strings seqs sets-maps; do
  printf '%-14s plain=%s hot=%s\n' $w $(pin_count $V/plain run $W/$w.clj) $(pin_count $V/hot run $W/$w.clj)
done
