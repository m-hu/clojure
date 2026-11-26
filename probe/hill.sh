#!/usr/bin/env bash
# Probe (throwaway, #271): interpreter peak RSS and smaps of a public
# program (synthetic input), master (A) vs PR head (B), each built as the
# h2h builds it (fresh order from its own tree's training set).
set -euo pipefail
out=/tmp/hill; mkdir -p $out
for v in A B; do
  ref=$(sed -n 1p probe/rss-refs); [ $v = B ] && ref=$(sed -n 2p probe/rss-refs)
  git checkout -q $ref -- rust scripts
  bash scripts/bench/hot-text/regen-run.sh rust > $out/$v-regen.log 2>&1 || { tail -20 $out/$v-regen.log; exit 1; }
  (cd rust && cargo build --release --locked --no-default-features -p clojure-rs) > $out/$v-build.log 2>&1 || { tail -30 $out/$v-build.log; exit 1; }
  cp rust/target/release/clojure-rs $out/$v-clojure-rs
  git checkout -q HEAD -- rust scripts
done
med() { sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }
for p in hill trivial bentley; do
  a=$(for i in 1 2 3 4 5 6 7; do /usr/bin/time -f %M $out/A-clojure-rs run probe/rssprog/$p.clj < probe/rssprog/$p.in 2>&1 >/dev/null | tail -1; done | med)
  b=$(for i in 1 2 3 4 5 6 7; do /usr/bin/time -f %M $out/B-clojure-rs run probe/rssprog/$p.clj < probe/rssprog/$p.in 2>&1 >/dev/null | tail -1; done | med)
  echo "$p A=$a B=$b B-A=$((b-a))"
done
printf '%s\n' '(load-file "probe/rssprog/hill.clj")' '(let [t (System/currentTimeMillis)] (loop [] (when (< (- (System/currentTimeMillis) t) 1500) (recur))))' > $out/hold.clj
for v in A B; do
  $out/$v-clojure-rs run $out/hold.clj < probe/rssprog/hill.in > /dev/null & pid=$!; sleep 1.2
  echo "== $v smaps (KB)"
  [ -r /proc/$pid/smaps ] || { echo "  (exited early)"; continue; }
  awk '/^[0-9a-f]+-[0-9a-f]+ / {name=$6; if (name=="") name="[anon]"; perm=$2} /^Rss:/ {k=name" "perm; s[k]+=$2} END {for (k in s) if (s[k] > 16) printf "  %7d %s\n", s[k], k}' /proc/$pid/smaps | sort -rn | head -8
  wait $pid || true
done
