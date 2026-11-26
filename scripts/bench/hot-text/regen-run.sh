#!/usr/bin/env bash
# Regenerate the interpreter hot-text order (rust/hot-text.ld) for one
# revision (#226, #271). Linux release links of `clojure-rs` pack the
# functions the order lists into one `.text.hot` section, so a run
# faults in a few 64 KiB windows of `.text` instead of most of it. The
# order names functions, so it goes stale whenever evaluator/value code
# moves (renamed or new functions fall back to the cold `.text`): CI
# regenerates it for every build it ships or measures (rust-release.yml,
# the `hot-text` job of h2h-cpu-mem.yml).
#
# Training set: this directory's workloads/*.clj (27 generic programs,
# empty stdin) run under `clojure-rs run` with callgrind, never the
# benchmark snippets or their inputs.
#
# Needs Linux, valgrind and python3 (no LLVM: builds
# `--no-default-features`).
#
# usage: regen-run.sh RUST_WORKSPACE [OUT.ld]
#   OUT.ld defaults to RUST_WORKSPACE/hot-text.ld (in place).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ws=$(cd "$1" && pwd)
out=${2:-$ws/hot-text.ld}
order=$ws/hot-text.ld
work=$(mktemp -d)
cp "$order" "$work/previous.ld"
# Restore the revision's order on any exit (the build below links with
# a blank one: callgrind attributes `.text.hot` to an unnamed object, so
# an order in place would hide its own hot functions).
trap 'cp "$work/previous.ld" "$order"; rm -rf "$work"' EXIT
printf '%s\n' '/* Placeholder while scripts/bench/hot-text/regen-run.sh profiles. */' \
  'SECTIONS' '{' '  .text.hot :' '  {' '  }' '}' 'INSERT BEFORE .text;' > "$order"

echo "== build $ws --no-default-features (blank order)"
# Profiles name functions: keep the symbols release builds strip (#271).
(cd "$ws" && CARGO_PROFILE_RELEASE_STRIP=false \
  cargo build --release --locked --no-default-features -p clojure-rs) > "$work/build.log" 2>&1 \
  || { tail -30 "$work/build.log" >&2; exit 1; }
bin=$ws/target/release/clojure-rs

echo "== profile the workloads"
mkdir -p "$work/cg"
# Each workload runs as programs are run, by a path relative to the
# working directory (#271): a relative script's `*file*` is absolutized
# at load (`current_dir`, `Path::join`), code every real run executes.
# Profiled from absolute paths, it stayed in cold `.text`, where the
# kernel maps executable text in large page-cache folios: those three
# small functions alone kept ~1 MB of cold text resident in every run.
for w in "$here"/workloads/*.clj; do
  name=$(basename "$w" .clj)
  (cd "$here/workloads" && valgrind --tool=callgrind --demangle=no --dump-instr=no \
    --callgrind-out-file="$work/cg/$name.cg" "$bin" run "$name.clj") < /dev/null > /dev/null 2> "$work/$name.err" || {
    tail -n 5 "$work/$name.err" >&2
    echo "regen-run: workload $name failed" >&2
    exit 1
  }
  echo "   $name"
done

echo "== order"
python3 "$here/hot_text.py" "$bin" "$work/new.ld" "$work"/cg/*.cg
# Restore the revision's order, then write OUT (which may be that
# order itself), and drop the restoring trap.
cp "$work/previous.ld" "$order"
cp "$work/new.ld" "$out"
trap 'rm -rf "$work"' EXIT
