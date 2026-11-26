#!/usr/bin/env bash
# Probe (throwaway, #271): for master (A) and the PR head (B) interpreters
# built as the h2h builds them, the functions a run executes, the bytes
# they span, and the 64 KiB windows they touch in .text.hot and .text.
set -euo pipefail
out=/tmp/win; mkdir -p $out
for v in A B; do
  ref=$(sed -n 1p probe/rss-refs); [ $v = B ] && ref=$(sed -n 2p probe/rss-refs)
  git checkout -q $ref -- rust scripts
  bash scripts/bench/hot-text/regen-run.sh rust > $out/$v-regen.log 2>&1 || { tail -20 $out/$v-regen.log; exit 1; }
  (cd rust && CARGO_PROFILE_RELEASE_STRIP=false cargo build --release --locked --no-default-features -p clojure-rs) > $out/$v-build.log 2>&1 || { tail -30 $out/$v-build.log; exit 1; }
  cp rust/target/release/clojure-rs $out/$v-bin
  readelf -SW $out/$v-bin | grep -E " \.text(\.hot)? " > $out/$v.sec
  nm -S --defined-only $out/$v-bin | awk '$3 ~ /[tTwW]/' > $out/$v.syms
  for p in trivial hill; do
    valgrind --tool=callgrind --demangle=no --callgrind-out-file=$out/$v-$p.cg $out/$v-bin run probe/rssprog/$p.clj < probe/rssprog/$p.in > /dev/null 2> /dev/null || true
  done
  git checkout -q HEAD -- rust scripts
done
python3 - <<'PY'
import re, collections
for v in ('A', 'B'):
    secs = {}
    for l in open(f'/tmp/win/{v}.sec'):
        m = re.search(r'(\.text(?:\.hot)?)\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)', l)
        if m: secs[m.group(1)] = (int(m.group(2), 16), int(m.group(4), 16))
    syms = {}
    for l in open(f'/tmp/win/{v}.syms'):
        p = l.split()
        if len(p) >= 4: syms[p[3]] = (int(p[0], 16), int(p[1], 16))
    hlo, hsz = secs['.text.hot']; tlo, tsz = secs['.text']
    print(f"== {v}: .text.hot {hsz//1024} KB, .text {tsz//1024} KB")
    for p in ('trivial', 'hill'):
        fns = set()
        for l in open(f'/tmp/win/{v}-{p}.cg'):
            m = re.match(r'^c?fn=\(\d+\) (\S+)', l.strip())
            if m: fns.add(m.group(1))
        hw = set(); cw = set(); hb = cb = 0; nh = nc = 0; cold = []
        for fn in fns:
            if fn not in syms: continue
            a, sz = syms[fn]
            if hlo <= a < hlo + hsz:
                nh += 1; hb += sz; hw.update(range((a - hlo) // 65536, (a + sz - hlo) // 65536 + 1))
            elif tlo <= a < tlo + tsz:
                nc += 1; cb += sz; cw.update(range((a - tlo) // 65536, (a + sz - tlo) // 65536 + 1)); cold.append((sz, fn))
        print(f"  {p}: hot fns {nh} ({hb//1024} KB) in {len(hw)} windows; cold fns {nc} ({cb//1024} KB) in {len(cw)} windows")
        if v == 'B' or p == 'trivial':
            for sz, fn in sorted(cold, reverse=True)[:25]: print(f"     COLD {v} {p} {sz} {fn}")
PY
