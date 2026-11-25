#!/usr/bin/env bash
# Probe (throwaway, #271): glibc vs mimalloc for Linux AOT binaries.
# Per variant: regenerate the AOT hot-text order (as CI does), build,
# then per program (public snippet sources, synthetic inputs written for
# this probe) the callgrind instruction total of one run and the median
# peak RSS of 7 runs, on its input and on empty stdin.
set -euo pipefail
root=$(pwd)
out=/tmp/alloc; mkdir -p $out
variant() {
  git checkout -q -- rust
  case $1 in
    glibc) ;;
    mi3|mi2)
      feats='"no_thp"'; [ "$1" = mi2 ] && feats='"no_thp", "v2"'
      python3 - "$feats" <<'PY'
import re, sys
p = 'rust/crates/clj-eval/Cargo.toml'; s = open(p).read()
s = s.replace("[target.'cfg(not(target_os = \"linux\"))'.dependencies]\nmimalloc = { version = \"0.1.52\", default-features = false }",
              "[target.'cfg(target_os = \"linux\")'.dependencies]\nmimalloc = { version = \"0.1.52\", default-features = false, features = [%s] }\n[target.'cfg(not(target_os = \"linux\"))'.dependencies.mimalloc]\nversion = \"0.1.52\"\ndefault-features = false" % sys.argv[1])
open(p, 'w').write(s)
p = 'rust/crates/clj-eval/src/lib.rs'; s = open(p).read()
s = s.replace('#[cfg(not(target_os = "linux"))]\n#[global_allocator]', '#[global_allocator]')
open(p, 'w').write(s)
PY
      git diff --stat rust ;;
  esac
}
for v in glibc mi3 mi2; do
  echo "######## $v"
  variant $v
  bash scripts/bench/hot-text/regen-aot.sh rust > $out/$v-regen.log 2>&1 || { tail -20 $out/$v-regen.log; continue; }
  (cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > $out/$v-build.log 2>&1 || { tail -30 $out/$v-build.log; continue; }
  c=$root/rust/target/release/clojure-rs
  for f in probe/prog/*.clj; do
    k=$(basename $f .clj)
    $c build $f -o $out/$v-$k > /dev/null 2>&1 || { echo "$v $k build-fail"; continue; }
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
for v in ('glibc', 'mi3', 'mi2'):
    try: lines = open(f'/tmp/alloc/{v}.txt').read().splitlines()
    except FileNotFoundError: continue
    for l in lines:
        m = re.match(r'(\S+) (\S+) (\S+) ins=(\d+) rss=(\d+)', l)
        if m: rows[(m.group(2), m.group(3))][v] = (int(m.group(4)), int(m.group(5)))
print(f"{'program':16} {'tag':6} {'glibc ins':>12} {'mi3/g':>7} {'mi2/g':>7} | {'glibc KB':>9} {'mi3-g':>7} {'mi2-g':>7}")
for (k, t), d in sorted(rows.items()):
    g = d.get('glibc')
    if not g: continue
    f = lambda v, i: (f"{d[v][0]/g[0]:.3f}" if i == 0 else f"{d[v][1]-g[1]:+d}") if v in d else '-'
    print(f"{k:16} {t:6} {g[0]:12d} {f('mi3',0):>7} {f('mi2',0):>7} | {g[1]:9d} {f('mi3',1):>7} {f('mi2',1):>7}")
PY
