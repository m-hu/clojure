#!/usr/bin/env bash
# Probe (throwaway, #271): does stripping the interpreter change its peak
# RSS? One tree (the PR head), one hot-text order, three binaries: linked
# unstripped, linked stripped (cargo strip = true), and the unstripped
# one stripped after the fact (objcopy). Master (unstripped, as it ships)
# for reference. Interleaved runs, median of 9.
set -euo pipefail
out=/tmp/strip; mkdir -p $out
ref=$(sed -n 1p probe/rss-refs)
git checkout -q $ref -- rust scripts
bash scripts/bench/hot-text/regen-run.sh rust > $out/A-regen.log 2>&1 || { tail -20 $out/A-regen.log; exit 1; }
(cd rust && cargo build --release --locked --no-default-features -p clojure-rs) > $out/A-build.log 2>&1 || { tail -30 $out/A-build.log; exit 1; }
cp rust/target/release/clojure-rs $out/master
ref=$(sed -n 2p probe/rss-refs)
git checkout -q $ref -- rust scripts
bash scripts/bench/hot-text/regen-run.sh rust > $out/B-regen.log 2>&1 || { tail -20 $out/B-regen.log; exit 1; }
(cd rust && CARGO_PROFILE_RELEASE_STRIP=false cargo build --release --locked --no-default-features -p clojure-rs) > $out/B1.log 2>&1 || { tail -30 $out/B1.log; exit 1; }
cp rust/target/release/clojure-rs $out/unstripped
objcopy --strip-all $out/unstripped $out/objcopy
(cd rust && cargo build --release --locked --no-default-features -p clojure-rs) > $out/B2.log 2>&1 || { tail -30 $out/B2.log; exit 1; }
cp rust/target/release/clojure-rs $out/stripped
git checkout -q HEAD -- rust scripts
ls -l $out/master $out/unstripped $out/objcopy $out/stripped | awk '{print $5, $9}'
for b in master unstripped objcopy stripped; do readelf -lW $out/$b | awk '/LOAD/ {print "  '$b'", $2, $3, $6, $7}'; done
cmp <(objcopy -O binary --only-section=.text $out/unstripped /dev/stdout | md5sum) <(objcopy -O binary --only-section=.text $out/stripped /dev/stdout | md5sum) && echo ".text identical (unstripped vs stripped)" || echo ".text DIFFERS (unstripped vs stripped)"
for p in trivial hill bentley; do
  declare -A r=()
  for i in 1 2 3 4 5 6 7 8 9; do
    for b in master unstripped objcopy stripped; do
      r[$b]+="$(/usr/bin/time -f %M $out/$b run probe/rssprog/$p.clj < probe/rssprog/$p.in 2>&1 >/dev/null | tail -1) "
    done
  done
  line="$p"
  for b in master unstripped objcopy stripped; do
    m=$(echo ${r[$b]} | tr ' ' '\n' | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
    line+=" $b=$m"
  done
  echo "$line"
  unset r
done
# The same binaries freshly copied (page cache from a plain read+write).
for b in master unstripped stripped; do cat $out/$b > $out/$b.fresh; chmod +x $out/$b.fresh; done
for p in trivial hill; do
  line="$p (fresh copies)"
  for b in master unstripped stripped; do
    m=$(for i in 1 2 3 4 5 6 7; do /usr/bin/time -f %M $out/$b.fresh run probe/rssprog/$p.clj < probe/rssprog/$p.in 2>&1 >/dev/null | tail -1; done | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
    line+=" $b=$m"
  done
  echo "$line"
done
