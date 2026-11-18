#!/usr/bin/env bash
# Regenerate rust/hot-text-aot.ld for one revision (#271). The order goes
# stale whenever runtime code moves, and a stale order costs peak RSS, so
# CI regenerates it for every AOT build it measures (see the `hot-text`
# job in .github/workflows/h2h-cpu-mem.yml).
#
# Profiles, all on this revision:
#   - which runtime functions each snippet program enters, over every
#     test case input (an entry-count instrumented runtime archive; a
#     program without inputs runs on empty stdin);
#   - the startup set, std's precompiled functions included (callgrind
#     of `(println 1)`).
# Then scripts/bench/hot-text/aot_order.py lays them out.
#
# Needs Linux, valgrind, python3, LLVM for `clojure-rs build`, and the
# toolchain's llvm-tools component (llvm-profdata).
#
# usage: regen-aot.sh RUST_WORKSPACE SNIPPETS_DIR [OUT.ld]
#   SNIPPETS_DIR holds src/**/*.clj and, when present, tests/resources
#   (test_case_*/input.txt per script).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ws=$(cd "$1" && pwd)
snip=$(cd "$2" && pwd)
out=${3:-$ws/hot-text-aot.ld}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

host=$(rustc -vV | sed -n 's/^host: //p')
sysroot=$(rustc --print sysroot)
profdata="$sysroot/lib/rustlib/$host/bin/llvm-profdata"
[ -x "$profdata" ] || { echo "error: llvm-profdata missing (rustup component add llvm-tools)" >&2; exit 1; }

echo "== build $ws"
(cd "$ws" && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > "$work/build.log" 2>&1 \
  || { tail -30 "$work/build.log" >&2; exit 1; }
compiler="$ws/target/release/clojure-rs"

echo "== instrumented runtime"
RUSTFLAGS="-C profile-generate=$work/pgdata -C llvm-args=-pgo-instrument-entry=true" \
  cargo rustc --manifest-path "$ws/Cargo.toml" --release --locked -p clj-rt --crate-type staticlib \
  --target-dir "$work/instr" > "$work/instr.log" 2>&1 || { tail -30 "$work/instr.log" >&2; exit 1; }
archive="$work/instr/release/libcljrt.a"
if ! nm "$archive" 2>/dev/null | grep -Eq ' [TDB] __llvm_profile_runtime$'; then
  # The profiler runtime is not part of a staticlib: add its objects.
  rlib=$(ls "$sysroot/lib/rustlib/$host/lib"/libprofiler_builtins-*.rlib | head -1)
  mkdir -p "$work/pb"
  (cd "$work/pb" && ar x "$rlib" && ar r "$archive" ./*.o) > /dev/null 2>&1
fi

echo "== profile every snippet"
python3 "$here/aot_order.py" cases "$snip/tests/resources" "$snip/src" > "$work/cases.tsv"
find "$snip/src" -name '*.clj' | sort > "$work/scripts.txt"
mkdir -p "$work/fns" "$work/bin" "$work/raw"
profile_one() {  # SCRIPT.clj
  local rel name bin raw
  rel=${1#"$snip/src/"}
  rel=${rel%.clj}
  name=$(printf '%s' "$rel" | sed 's#/#__#g')
  bin="$work/bin/$name"
  raw="$work/raw/$name"
  "$compiler" build "$1" -o "$bin" --runtime-archive "$archive" > /dev/null 2>&1 || return 0
  mkdir -p "$raw"
  local ran=0
  while read -r dir; do
    [ -f "$dir/input.txt" ] || continue
    LLVM_PROFILE_FILE="$raw/%p.profraw" timeout 120 "$bin" < "$dir/input.txt" > /dev/null 2>&1 || true
    ran=1
  done < <(awk -F '\t' -v r="$rel" '$1 == r { print $2 }' "$work/cases.tsv")
  if [ "$ran" = 0 ]; then
    LLVM_PROFILE_FILE="$raw/%p.profraw" timeout 120 "$bin" < /dev/null > /dev/null 2>&1 || true
  fi
  if ls "$raw"/*.profraw > /dev/null 2>&1; then
    "$profdata" merge --text -o - "$raw"/*.profraw 2> /dev/null | python3 "$here/aot_order.py" entered > "$work/fns/$name.txt"
  fi
  rm -rf "$raw" "$bin"
}
export -f profile_one
export work snip compiler archive profdata here
xargs -P "$(nproc)" -I{} bash -c 'profile_one "$1"' _ {} < "$work/scripts.txt"
echo "   $(find "$work/fns" -name '*.txt' -size +0 | wc -l) programs profiled"

echo "== startup set"
printf '(println 1)\n' > "$work/trivial.clj"
"$compiler" build "$work/trivial.clj" -o "$work/trivial" > /dev/null 2>&1
valgrind --tool=callgrind --demangle=no --dump-instr=no --callgrind-out-file="$work/trivial.cg" \
  "$work/trivial" < /dev/null > /dev/null 2>&1 || true
python3 "$here/aot_order.py" callgrind "$work/trivial.cg" > "$work/startup.txt"
echo "   $(wc -l < "$work/startup.txt") startup functions"

echo "== order"
nm -S --size-sort -t d "$ws/target/release/libcljrt.a" > "$work/sizes.txt" 2> /dev/null || true
python3 "$here/aot_order.py" order --fns "$work/fns" --startup "$work/startup.txt" \
  --sizes "$work/sizes.txt" --previous "$ws/hot-text-aot.ld" --out "$out"
