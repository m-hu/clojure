#!/usr/bin/env bash
# Probe (throwaway, #271): where does bucket_sort's AOT binary spend the
# work that `run` does in fewer instructions? Synthetic input shaped like
# the suite's test_case_2 (500 uniform values in [0, 1e6)).
set -euo pipefail
root=$(pwd)
cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib && cd ..
cljrs="$root/rust/target/release/clojure-rs"
mkdir -p /tmp/w && cd /tmp/w
python3 -c "import random; random.seed(7); print(500); print('\n'.join(str(random.randrange(1000000)) for _ in range(500)))" > bs.in
"$cljrs" build "$root/probe/bucket_sort.clj" -o bs_aot
cmp <(./bs_aot < bs.in) <("$cljrs" run "$root/probe/bucket_sort.clj" < bs.in) && echo "outputs match"
valgrind --tool=callgrind --callgrind-out-file=cg.aot ./bs_aot < bs.in > /dev/null 2>&1 || true
valgrind --tool=callgrind --callgrind-out-file=cg.run "$cljrs" run "$root/probe/bucket_sort.clj" < bs.in > /dev/null 2>&1 || true
for v in aot run; do
  echo "== $v total $(grep -E '^(summary|totals):' cg.$v | head -1)"
  echo "-- inclusive"; callgrind_annotate --inclusive=yes cg.$v 2>/dev/null | sed -n '/file:function/,$p' | sed -E 's/ \[[^]]*\]$//' | cut -c1-170 | head -60
  echo "-- self"; callgrind_annotate cg.$v 2>/dev/null | sed -n '/file:function/,$p' | sed -E 's/ \[[^]]*\]$//' | cut -c1-170 | head -45
done
