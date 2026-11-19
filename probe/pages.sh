#!/usr/bin/env bash
# Probe (throwaway, #271): where AOT binaries' resident pages come from.
# For each program (empty stdin, no inputs): per-page residency of its
# mappings at exit with the kernel's default fault-around and with
# fault-around off (the pages it actually touched), and the functions it
# ran (callgrind), so text layouts can be simulated offline.
set -euo pipefail
root=$(pwd)
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q valgrind gdb > /dev/null
ws=$root/rust
bash scripts/bench/hot-text/regen-aot.sh "$ws" > /tmp/regen.log 2>&1 || { tail -30 /tmp/regen.log; exit 1; }
tail -3 /tmp/regen.log
(cd "$ws" && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > /tmp/build.log 2>&1 || { tail -30 /tmp/build.log; exit 1; }
c=$ws/target/release/clojure-rs
out=/tmp/pages
mkdir -p $out/bin $out/fns $out/pm
cp "$ws/hot-text-aot.ld" $out/order.ld
{
  find snip/src -name '*.clj' | sort | sed 's/^/snip /'
  ls docs/spec/i259-native-compiler/corpus/*.clj | sed 's/^/corpus /'
  ls scripts/bench/hot-text/workloads/*.clj | sed 's/^/work /'
} > $out/progs.txt
printf '(println 1)\n' > /tmp/trivial.clj
echo "corpus /tmp/trivial.clj" >> $out/progs.txt
n=0
while read -r group f; do
  k=$group-$(echo "${f%.clj}" | sed 's|^snip/src/||; s|.*/corpus/||; s|.*/workloads/||; s|/|_|g')
  "$c" build "$f" -o "$out/bin/$k" > /dev/null 2>&1 || { echo "$k build-fail"; continue; }
  valgrind --tool=callgrind --dump-instr=no --demangle=no --callgrind-out-file=/tmp/cg.out "$out/bin/$k" < /dev/null > /dev/null 2>&1 || true
  python3 probe/fns.py /tmp/cg.out > "$out/fns/$k.txt"
  n=$((n + 1))
done < $out/progs.txt
echo "built and profiled $n programs"
nm -S --defined-only -n "$out/bin/corpus-trivial" > $out/nm.txt 2> /dev/null || nm -S --defined-only -n "$(ls $out/bin/* | head -1)" > $out/nm.txt
readelf -SW "$out/bin/corpus-trivial" > $out/sections.txt
readelf -lW "$out/bin/corpus-trivial" > $out/segments.txt
sudo mount -t debugfs none /sys/kernel/debug 2> /dev/null || true
fa=/sys/kernel/debug/fault_around_bytes
echo "fault_around_bytes: $(sudo cat $fa)"
measure() {  # TAG
  for b in "$out"/bin/*; do
    k=$(basename "$b")
    PM_OUT="$out/pm/$k.$1.json" gdb -q -batch -ex 'catch syscall exit_group' -ex 'run < /dev/null > /dev/null 2>&1' \
      -ex "python exec(open('$root/probe/pm.py').read())" -ex kill "$b" > /dev/null 2>&1 || echo "$k $1 gdb-fail"
  done
}
measure fa
echo 4096 | sudo tee $fa > /dev/null && echo "fault_around_bytes now: $(sudo cat $fa)" && measure nofa
python3 probe/pagesum.py $out
tar czf /tmp/pages.tgz -C /tmp pages/fns pages/pm pages/nm.txt pages/sections.txt pages/segments.txt pages/order.ld pages/progs.txt
