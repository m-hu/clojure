#!/usr/bin/env bash
# Probe (throwaway, #271): which runtime functions native (AOT) runs of
# public programs (synthetic inputs) execute outside .text.hot, built as
# CI builds them (fresh AOT order from this tree's training set).
set -euo pipefail
out=/tmp/coldaot; mkdir -p $out
ref=$(sed -n 2p probe/rss-refs)
git checkout -q $ref -- rust
bash scripts/bench/hot-text/regen-aot.sh rust > $out/regen.log 2>&1 || { tail -20 $out/regen.log; exit 1; }
(cd rust && cargo build --release --locked -p clojure-rs && cargo rustc --release --locked -p clj-rt --crate-type staticlib) > $out/build.log 2>&1 || { tail -30 $out/build.log; exit 1; }
c=$(pwd)/rust/target/release/clojure-rs
for f in probe/prog/*.clj; do
  k=$(basename $f .clj)
  $c build $f -o $out/$k --keep-symbols > /dev/null 2>&1 || { echo "$k build-fail"; continue; }
  readelf -SW $out/$k | grep -E " \.text(\.hot)? " > $out/$k.sec
  nm -S --defined-only $out/$k | awk '$3 ~ /[tTwW]/' > $out/$k.syms
  valgrind --tool=callgrind --demangle=no --callgrind-out-file=$out/$k.cg $out/$k < probe/prog/$k.in > /dev/null 2> $out/$k.vg || true
  echo "$k rss $(/usr/bin/time -f %M $out/$k < probe/prog/$k.in 2>&1 >/dev/null | tail -1)"
done
python3 - <<'PY'
import re, glob, os, collections
agg = collections.Counter()
for cg in sorted(glob.glob('/tmp/coldaot/*.cg')):
    k = os.path.basename(cg)[:-3]
    secs = {}
    for l in open(f'/tmp/coldaot/{k}.sec'):
        m = re.search(r'(\.text(?:\.hot)?)\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)', l)
        if m: secs[m.group(1)] = (int(m.group(2), 16), int(m.group(4), 16))
    if '.text.hot' not in secs: print(k, 'no .text.hot'); continue
    hlo, hsz = secs['.text.hot']; tlo, tsz = secs['.text']
    syms = {}
    for l in open(f'/tmp/coldaot/{k}.syms'):
        p = l.split()
        if len(p) >= 4: syms[p[3]] = (int(p[0], 16), int(p[1], 16))
    fns = set()
    for l in open(cg):
        m = re.match(r'^c?fn=\(\d+\) (\S+)', l.strip())
        if m: fns.add(m.group(1))
    hot = 0; cold = []; win = set()
    for fn in fns:
        if fn not in syms: continue
        a, sz = syms[fn]
        if hlo <= a < hlo + hsz: hot += 1
        elif tlo <= a < tlo + tsz and not fn.startswith('cljfn') and not fn.startswith('cljtop'):
            cold.append((sz, fn)); win.add((a - tlo) // 65536)
    print(f"== {k}: hot {hot}, cold runtime fns {len(cold)} ({sum(s for s,_ in cold)//1024} KB), cold 64K windows {len(win)} of {tsz//65536}")
    for sz, fn in cold: agg[fn] += 1
for fn, c in agg.most_common(120): print(f"AGG {c} {fn}")
PY
