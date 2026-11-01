#!/bin/sh
# Wall-clock micro-bench for the Z3 handle-table FFI (spec §3.4):
# compile + run two loop-heavy subset programs and report wall
# time. Run once per toolchain; before/after numbers land in the
# PR body (new measurement, not a regression gate).
#
# Usage:
#   sh scripts/handle-abi-bench.sh <clojure-rs> <runtime-archive> <workdir>
set -eu

BIN="$1"; ARCHIVE="$2"; WORK="$3"
mkdir -p "$WORK"

# Scalar loop: recursion + int arithmetic + truthiness (the hot
# path — inline scalars, no table traffic post-flip).
cat > "$WORK/sum.clj" <<'EOF'
(defn sum [n acc] (if (= n 0) acc (sum (- n 1) (+ acc n))))
(print (sum 10000 0))
EOF

# Heap loop: persistent-vector conj per iteration (table
# refcount traffic on every step).
cat > "$WORK/build.clj" <<'EOF'
(defn build [n v] (if (= n 0) (count v) (build (- n 1) (conj v n))))
(print (build 3000 []))
EOF

for prog in sum build; do
  echo "== $prog: compile"
  /usr/bin/time -p "$BIN" build "$WORK/$prog.clj" -o "$WORK/$prog" --runtime-archive "$ARCHIVE" 2>&1 | grep -E "real|user|sys"
  echo "== $prog: run x5"
  i=1
  while [ "$i" -le 5 ]; do
    out=$("$WORK/$prog")
    t=$( ( /usr/bin/time -p "$WORK/$prog" >/dev/null ) 2>&1 | awk '/^real/ {print $2}' )
    echo "run $i: ${t}s out=$out"
    i=$((i + 1))
  done
done
