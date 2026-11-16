#!/usr/bin/env bash
set -euo pipefail
root=$(pwd)
cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib && cd ..
cljrs="$root/rust/target/release/clojure-rs"
mkdir -p /tmp/w && cd /tmp/w
printf '(println 1)\n' > hello.clj
printf '(require (quote [clojure.string :as str]))\n(let [n (Long/parseLong (str/trim (read-line))) xs (mapv #(Long/parseLong %%) (str/split (str/trim (read-line)) #"\\s+"))] (println (reduce + 0 (take n xs))))\n' > sum.clj
printf '5\n1 2 3 4 5\n' > sum.in
: > empty.in
for p in hello sum; do
  "$cljrs" build --report "$p.clj" -o "$p" 2>&1 | tail -1
  file "$p" | sed 's/BuildID[^,]*, //'
done
echo "hello: $(./hello)  sum: $(./sum < sum.in)"
for p in hello sum; do
  in=empty.in; [ "$p" = sum ] && in=sum.in
  rss=$(for _ in 1 2 3; do { /usr/bin/time -f '%M' "./$p" < "$in" > /dev/null; } 2>&1 | tail -1; done | sort -n | head -1)
  valgrind --tool=callgrind --callgrind-out-file="cg.$p" "./$p" < "$in" > /dev/null 2>&1 || true
  ir=$(grep -E '^(summary|totals):' "cg.$p" | head -1 | awk '{print $2}')
  echo "$p ir=$ir maxrss_kb=$rss"
done
