#!/usr/bin/env bash
# Regenerate the AOT hot-text order (rust/hot-text-aot.ld) for one
# revision (#271). Linux AOT links pack the functions the order lists
# into one `.text.hot` section, so a run faults in a few 64 KiB windows
# of the runtime's text instead of most of it. The order names runtime
# functions, so it goes stale whenever runtime code moves (renamed or
# new functions fall back to the cold `.text`): CI regenerates it for
# every AOT build it ships or measures (rust-release.yml, the `hot-text`
# job of h2h-cpu-mem.yml).
#
# Training set: this directory's workloads/*.clj (empty stdin) and
# aot-workloads/*.clj (stdin from NAME.in), plus the native compiler's
# test corpus, docs/spec/i259-native-compiler/corpus/*.clj (empty stdin;
# programs that do not build or that fail are skipped). These are generic
# programs, never the benchmark snippets or their inputs, so the order
# follows the runtime's own hot paths, whatever program links it. The
# corpus widens what the order covers: held out of training, half of it
# measured 508 KB lower median peak RSS with an order trained on the
# workloads and the other half than with the workloads alone (#271).
#
# Profiles: callgrind of each workload compiled against the revision's
# runtime, linked with an empty order (callgrind attributes the
# `.text.hot` output section to an unnamed object, so an order in place
# would hide its own functions). hot_text.py lists the functions they
# run, the ones most workloads run first, the same layout as the
# interpreter's rust/hot-text.ld. (A layout packing 62 KiB windows of
# co-used functions measured better on empty stdin but raised median
# peak RSS by 116 KB on the h2h inputs, #271.)
#
# Needs Linux, valgrind, python3, and LLVM for `clojure-rs build`.
#
# usage: regen-aot.sh RUST_WORKSPACE [OUT.ld]
#   OUT.ld defaults to RUST_WORKSPACE/hot-text-aot.ld (in place).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ws=$(cd "$1" && pwd)
out=${2:-$ws/hot-text-aot.ld}
order=$ws/hot-text-aot.ld
work=$(mktemp -d)
cp "$order" "$work/previous.ld"
# Restore the revision's order on any exit (the build below links with
# a blank one).
trap 'cp "$work/previous.ld" "$order"; rm -rf "$work"' EXIT
printf '%s\n' '/* Placeholder while scripts/bench/hot-text/regen-aot.sh profiles. */' \
  'SECTIONS' '{' '  .text.hot :' '  {' '  }' '}' 'INSERT BEFORE .text;' > "$order"

echo "== build $ws (blank order)"
(cd "$ws" && cargo build --release --locked -p clojure-rs \
  && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > "$work/build.log" 2>&1 \
  || { tail -30 "$work/build.log" >&2; exit 1; }
compiler=$ws/target/release/clojure-rs
# Profiles name functions, so the workloads link with their symbols
# (release builds strip them, #271); revisions before that flag keep
# symbols anyway.
keep=()
"$compiler" build --help 2>/dev/null | grep -q -- --keep-symbols && keep=(--keep-symbols)

echo "== profile the workloads"
# Every workload links as `prog`, the object hot_text.py reads.
mkdir -p "$work/cg"
# Built from a path relative to the working directory, as programs are
# built (#271): the binary then absolutizes its relative `*file*` at
# startup (`current_dir`, `getcwd`, `Path::join`), code every real run
# executes; built from absolute paths, it stayed in cold `.text`.
for w in "$here"/workloads/*.clj "$here"/aot-workloads/*.clj; do
  name=$(basename "$w" .clj)
  input=${w%.clj}.in
  [ -f "$input" ] || input=/dev/null
  (cd "$(dirname "$w")" && "$compiler" build "${keep[@]}" "$name.clj" -o "$work/prog") > /dev/null 2> "$work/$name.err" || {
    tail -n 5 "$work/$name.err" >&2
    echo "regen-aot: workload $name failed to compile" >&2
    exit 1
  }
  valgrind --tool=callgrind --demangle=no --dump-instr=no \
    --callgrind-out-file="$work/cg/$name.cg" "$work/prog" < "$input" > /dev/null 2>> "$work/$name.err" || {
    tail -n 5 "$work/$name.err" >&2
    echo "regen-aot: workload $name failed" >&2
    exit 1
  }
  echo "   $name"
done

echo "== profile the compiler corpus"
corpus=$(cd "$here/../../../docs/spec/i259-native-compiler/corpus" && pwd)
for c in "$corpus"/*.clj; do
  name=corpus-$(basename "$c" .clj)
  (cd "$corpus" && "$compiler" build "${keep[@]}" "$(basename "$c")" -o "$work/prog") > /dev/null 2>&1 || continue
  valgrind --tool=callgrind --demangle=no --dump-instr=no \
    --callgrind-out-file="$work/cg/$name.cg" "$work/prog" < /dev/null > /dev/null 2>&1 || true
done
echo "   $(ls "$work"/cg/corpus-*.cg 2> /dev/null | wc -l) corpus profiles"

echo "== order"
# The C runtime links statically without function sections: list the
# archive members the profiles ran (hot_text.py --archive).
archives=()
for lib in libc.a libm.a libgcc.a; do
  path=$(cc -print-file-name="$lib")
  [ -f "$path" ] && archives+=(--archive "$path")
done
python3 "$here/hot_text.py" "${archives[@]}" "$work/prog" "$work/new.ld" "$work"/cg/*.cg
# Restore the revision's order, then write OUT (which may be that
# order itself), and drop the restoring trap.
cp "$work/previous.ld" "$order"
cp "$work/new.ld" "$out"
trap 'rm -rf "$work"' EXIT
