#!/usr/bin/env bash
# Probe (throwaway, #271): which functions an interpreter run of a public
# program executes outside the hot-text section (fresh order, as CI
# builds it), grouped by crate/module, with their sizes and the 64 KiB
# windows of .text they touch.
set -euo pipefail
out=/tmp/cold; mkdir -p $out
ref=$(sed -n 2p probe/rss-refs)
git checkout -q $ref -- rust
bash scripts/bench/hot-text/regen-run.sh rust > $out/regen.log 2>&1 || { tail -20 $out/regen.log; exit 1; }
(cd rust && CARGO_PROFILE_RELEASE_STRIP=false cargo build --release --locked --no-default-features -p clojure-rs) > $out/build.log 2>&1 || { tail -30 $out/build.log; exit 1; }
bin=rust/target/release/clojure-rs
readelf -SW $bin | grep -E " \.text(\.hot)? " | tee $out/sections.txt
nm -S --defined-only $bin | awk '$3 ~ /[tTwW]/' > $out/syms.txt
for p in bentley trivial; do
  valgrind --tool=callgrind --demangle=no --callgrind-out-file=$out/$p.cg $bin run probe/rssprog/$p.clj < probe/rssprog/$p.in > /dev/null 2> $out/$p.vg || true
  /usr/bin/time -f "$p rss %M" $bin run probe/rssprog/$p.clj < probe/rssprog/$p.in > /dev/null
done
python3 - <<'PY'
import re, collections
secs = {}
for l in open('/tmp/cold/sections.txt'):
    m = re.search(r'(\.text(?:\.hot)?)\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)', l)
    if m: secs[m.group(1)] = (int(m.group(2), 16), int(m.group(4), 16))
hot_lo, hot_sz = secs['.text.hot']; txt_lo, txt_sz = secs['.text']
syms = {}
for l in open('/tmp/cold/syms.txt'):
    p = l.split()
    if len(p) >= 4: syms[p[3]] = (int(p[0], 16), int(p[1], 16))
for prog in ('bentley', 'trivial'):
    fns = set()
    for l in open(f'/tmp/cold/{prog}.cg'):
        m = re.match(r'^c?fn=\(\d+\) (\S+)', l.strip())
        if m: fns.add(m.group(1))
    hot = 0; cold = []; windows = set(); unknown = 0
    for fn in fns:
        if fn not in syms: unknown += 1; continue
        a, sz = syms[fn]
        if hot_lo <= a < hot_lo + hot_sz: hot += 1
        elif txt_lo <= a < txt_lo + txt_sz:
            cold.append((sz, fn)); windows.add((a - txt_lo) // 65536)
    print(f"== {prog}: executed fns {len(fns)} (not in nm {unknown}), in .text.hot {hot}, in cold .text {len(cold)} ({sum(s for s,_ in cold)//1024} KB), cold 64K windows {len(windows)} of {txt_sz//65536}")
    with open(f'/tmp/cold/{prog}-cold.txt', 'w') as out:
        for sz, k in sorted(cold, reverse=True): out.write(f"{sz} {k}\n")
PY
for p in bentley trivial; do echo "== $p cold (size name)"; cat /tmp/cold/$p-cold.txt; done
