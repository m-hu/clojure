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
  valgrind --tool=callgrind --callgrind-out-file=$out/$p.cg $bin run probe/rssprog/$p.clj < probe/rssprog/$p.in > /dev/null 2> $out/$p.vg || true
  /usr/bin/time -f "$p rss %M" $bin run probe/rssprog/$p.clj < probe/rssprog/$p.in > /dev/null
done
python3 - <<'PY'
import re, subprocess, collections
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
        m = re.match(r'^fn=\(\d+\)(?: (.+))?$', l.strip())
        if m and m.group(1): fns.add(m.group(1))
    # callgrind names are demangled; map via nm demangled names
    dem = subprocess.run(['c++filt'], input='\n'.join(syms), capture_output=True, text=True).stdout.splitlines()
    bydem = {}
    for raw, d in zip(syms, dem): bydem.setdefault(re.sub(r'::h[0-9a-f]{16}$', '', d), []).append(raw)
    cold = []; hot = 0; windows = set()
    for fn in fns:
        key = re.sub(r'::h[0-9a-f]{16}$', '', fn)
        for raw in bydem.get(key, []):
            a, sz = syms[raw]
            if hot_lo <= a < hot_lo + hot_sz: hot += 1
            elif txt_lo <= a < txt_lo + txt_sz:
                cold.append((sz, key)); windows.add((a - txt_lo) // 65536)
    print(f"== {prog}: executed fns {len(fns)}, in .text.hot {hot}, in cold .text {len(cold)} ({sum(s for s,_ in cold)//1024} KB), cold 64K windows {len(windows)} of {txt_sz//65536}")
    groups = collections.Counter()
    for sz, k in cold:
        g = re.sub(r'<|>|\{.*', '', k).split('::')
        groups['::'.join(g[:3])] += 1
    for g, c in groups.most_common(30): print(f"   {c:4d} {g}")
    if prog == 'bentley':
        for sz, k in sorted(cold, reverse=True)[:60]: print(f"      {sz:7d} {k[:150]}")
PY
