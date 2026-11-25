#!/usr/bin/env bash
# Probe (throwaway, #271): PGO for the AOT runtime archive. Training set =
# the hot-text training programs (scripts/bench/hot-text workloads +
# aot-workloads, the i259 corpus): generic programs, never the snippets.
# Measured: public snippet programs on synthetic inputs (probe/prog), the
# callgrind instruction total of one run and the median peak RSS of 7.
set -euo pipefail
root=$(pwd); out=/tmp/pgo; mkdir -p $out/raw
rustup component add llvm-tools-preview > /dev/null
profdata=$(find "$(rustc --print sysroot)" -name llvm-profdata | head -1)
echo "profdata: $profdata"
bash scripts/bench/hot-text/regen-aot.sh rust > $out/regen.log 2>&1 || { tail -20 $out/regen.log; exit 1; }
(cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > $out/build.log 2>&1 || { tail -30 $out/build.log; exit 1; }
c=$root/rust/target/release/clojure-rs
base=$root/rust/target/release/libcljrt.a
(cd rust && CARGO_TARGET_DIR=$root/rust/target-gen RUSTFLAGS="-Cprofile-generate=$out/raw" cargo rustc --release --locked -p clj-rt --crate-type staticlib) > $out/gen.log 2>&1 || { tail -30 $out/gen.log; exit 1; }
gen=$root/rust/target-gen/release/libcljrt.a
# The instrumented archive references the LLVM profiler runtime, which
# rustc links into executables itself: add its objects for `cc` links.
pb=$(find "$(rustc --print sysroot)" -name 'libprofiler_builtins-*.rlib' | head -1)
echo "profiler builtins: $pb"
mkdir -p $out/pb && (cd $out/pb && ar x "$pb") && ar r "$gen" $out/pb/*.o
n=0
train() { # src input
  # Link by hand: the profile writer registers at startup only when the
  # binary references `__llvm_profile_runtime` (rustc's own links do).
  $c build "$1" -o $out/train --emit-obj $out/train.o > $out/train.err 2>&1 \
    && cc $out/train.o $gen -o $out/train -lm -Wl,-u,__llvm_profile_runtime >> $out/train.err 2>&1 \
    || { echo "train build fail $1: $(tail -1 $out/train.err | cut -c1-200)"; return 0; }
  LLVM_PROFILE_FILE="$out/raw/%m-%p.profraw" $out/train < "$2" > /dev/null 2>&1 || true
  n=$((n+1))
}
h=scripts/bench/hot-text
for w in $h/workloads/*.clj $h/aot-workloads/*.clj; do i=${w%.clj}.in; [ -f "$i" ] || i=/dev/null; train $w $i; done
for w in docs/spec/i259-native-compiler/corpus/*.clj; do train $w /dev/null; done
echo "trained $n programs; $(ls $out/raw | wc -l) raw profiles"
[ "$(ls $out/raw | wc -l)" -gt 20 ] || { echo "too few profiles"; exit 1; }
$profdata merge -o $out/merged.profdata $out/raw/*.profraw
ls -l $out/merged.profdata
(cd rust && CARGO_TARGET_DIR=$root/rust/target-use RUSTFLAGS="-Cprofile-use=$out/merged.profdata -Cllvm-args=-pgo-warn-missing-function=false" cargo rustc --release --locked -p clj-rt --crate-type staticlib) > $out/use.log 2>&1 || { tail -30 $out/use.log; exit 1; }
use=$root/rust/target-use/release/libcljrt.a
ls -l $base $use
for v in base pgo; do
  a=$base; [ $v = pgo ] && a=$use
  for f in probe/prog/*.clj; do
    k=$(basename $f .clj)
    $c build $f -o $out/$v-$k --runtime-archive $a > /dev/null 2>&1 || { echo "$v $k build-fail"; continue; }
    for inp in probe/prog/$k.in /dev/null; do
      tag=$([ $inp = /dev/null ] && echo empty || echo input)
      valgrind --tool=callgrind --callgrind-out-file=$out/$v-$k-$tag.cg $out/$v-$k < $inp > /dev/null 2>$out/$v-$k-$tag.vg || true
      ins=$(grep -m1 '^summary:' $out/$v-$k-$tag.cg | awk '{print $2}')
      rss=$(for i in 1 2 3 4 5 6 7; do /usr/bin/time -f %M $out/$v-$k < $inp 2>&1 >/dev/null | tail -1; done | sort -n | sed -n 4p)
      echo "$v $k $tag ins=$ins rss=$rss"
    done
  done | tee $out/$v.txt
  ls -l $out/$v-sudoku | awk '{print "size", $5}'
done
python3 - <<'PY'
import collections, re
rows = collections.defaultdict(dict)
for v in ('base', 'pgo'):
    for l in open(f'/tmp/pgo/{v}.txt').read().splitlines():
        m = re.match(r'(\S+) (\S+) (\S+) ins=(\d+) rss=(\d+)', l)
        if m: rows[(m.group(2), m.group(3))][v] = (int(m.group(4)), int(m.group(5)))
print(f"{'program':16} {'tag':6} {'base ins':>12} {'pgo/b':>7} | {'base KB':>8} {'pgo-b':>6}")
for (k, t), d in sorted(rows.items()):
    if 'base' in d and 'pgo' in d:
        b, p = d['base'], d['pgo']
        print(f"{k:16} {t:6} {b[0]:12d} {p[0]/b[0]:7.3f} | {b[1]:8d} {p[1]-b[1]:+6d}")
PY
