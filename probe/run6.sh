#!/usr/bin/env bash
# Probe v6 (#226): structural globs, rodata ordering variant, link map.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
R=$ROOT/rust; PR=$ROOT/probe; W=$ROOT/scripts/bench/hot-text/workloads; V=/tmp/v
mkdir -p $V /tmp/out /tmp/pm
cp $R/build.rs.real $R/build.rs
# probe hook: extra link args for the bin
cat >> $R/build.rs <<'RS'
#[allow(dead_code)]
fn probe_extra() {}
RS
sed -i 's|^fn main() {|fn main() {\n    if let Ok(a) = std::env::var("PROBE_LINK_ARGS") { for x in a.split_whitespace() { println!("cargo:rustc-link-arg-bins={x}"); } }\n    println!("cargo:rerun-if-env-changed=PROBE_LINK_ARGS");|' $R/build.rs
rm -f $R/hot-text.ld
echo "== generate hot-text.ld (structural globs)"
bash $ROOT/scripts/bench/hot-text/run.sh /tmp/out/hot-text.ld 2>&1 | grep -vE '^\s+Compiling' | tail -2
cp $R/target/release/clojure-rs $V/plain
head -8 /tmp/out/hot-text.ld | cut -c1-200
# rodata variant: same globs for .rodata.<fn> input sections
python3 - <<'PY'
src = open('/tmp/out/hot-text.ld').read()
pats = [l.strip()[len('*(.text.'):].split(' ')[0] for l in src.splitlines() if l.strip().startswith('*(.text.')]
body = '\n'.join('    *(.rodata.%s)' % p for p in pats)
ro = src.replace('INSERT BEFORE .text;', 'INSERT BEFORE .text;\nSECTIONS\n{\n  .rodata.hot :\n  {\n' + body + '\n  }\n}\nINSERT BEFORE .rodata;')
open('/tmp/out/hot-text-ro.ld', 'w').write(ro)
PY
build() { local n=$1; shift; local t0=$SECONDS
  (cd $R && PROBE_LINK_ARGS="$*" cargo build --release --locked --no-default-features 2>&1 | grep -E '^error|^warning|Finished' | tail -3)
  cp $R/target/release/clojure-rs $V/$n && echo "  $n built in $((SECONDS - t0))s: $(readelf -S -W $V/$n | grep -E ' \.(text|rodata)' | awk '{print $2, "0x"$6}' | tr '\n' ' ')"; }
cp /tmp/out/hot-text.ld $R/hot-text.ld
build hot -Wl,-Map=/tmp/out/hot.map
cp /tmp/out/hot-text-ro.ld $R/hot-text.ld
build hot-ro -Wl,-Map=/tmp/out/hot-ro.map
cp /tmp/out/hot-text.ld $R/hot-text.ld
med() { sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
rss() { local f=/tmp/rss.out; : > $f; for k in 1 2 3 4 5; do /usr/bin/time -a -o $f -f %M "$@" > /dev/null 2>&1; done; med < $f; }
WL=$(cd $W && ls *.clj | sed 's/\.clj$//')
echo "== RSS (median of 5, KB)"
printf '%-14s' run; for b in plain hot hot-ro; do printf '%9s' $b; done; echo
for w in $WL; do printf '%-14s' $w; for b in plain hot hot-ro; do printf '%9s' $(rss $V/$b run $W/$w.clj); done; echo; done
echo "== rodata map (hot, trivial)"
{ cat $W/trivial.clj; echo '(println "@@READY@@") (flush) (read-line)'; } > /tmp/pm/t.clj
python3 $PR/romap.py $V/hot /tmp/out/hot.map /tmp/pm/t.clj
echo "== rodata map (hot-ro, trivial)"
python3 $PR/romap.py $V/hot-ro /tmp/out/hot-ro.map /tmp/pm/t.clj | tail -8
for b in hot hot-ro; do echo "-- $b residency"; python3 $PR/pagemap.py $V/$b /tmp/pm/t.clj /tmp/out/$b.syms | grep -E 'binary resident|\.text|\.rodata|other mappings'; done
