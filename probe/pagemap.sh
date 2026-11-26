#!/usr/bin/env bash
# Probe (throwaway, #271): which pages of the interpreter's executable
# mapping are resident at the end of a run (pagemap), per section, for
# master (A) and the PR head (B) built as the h2h builds them.
set -euo pipefail
out=/tmp/pm; mkdir -p $out
for v in A B; do
  ref=$(sed -n 1p probe/rss-refs); [ $v = B ] && ref=$(sed -n 2p probe/rss-refs)
  git checkout -q $ref -- rust scripts
  bash scripts/bench/hot-text/regen-run.sh rust > $out/$v-regen.log 2>&1 || { tail -20 $out/$v-regen.log; exit 1; }
  (cd rust && CARGO_PROFILE_RELEASE_STRIP=false cargo build --release --locked --no-default-features -p clojure-rs) > $out/$v-build.log 2>&1 || { tail -30 $out/$v-build.log; exit 1; }
  cp rust/target/release/clojure-rs $out/$v-bin
  readelf -SW $out/$v-bin > $out/$v.sec
  readelf -lW $out/$v-bin > $out/$v.seg
  nm -n --defined-only $out/$v-bin | awk '$2 ~ /[tTwW]/ {print $1, $3}' > $out/$v.nm
  git checkout -q HEAD -- rust scripts
done
cat > $out/res.py <<'PY'
import sys, re, struct
pid, binpath, secfile = int(sys.argv[1]), sys.argv[2], sys.argv[3]
secs = []
for l in open(secfile):
    m = re.search(r'\]\s+(\S+)\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)', l)
    if m and int(m.group(2), 16): secs.append((m.group(1), int(m.group(2), 16), int(m.group(4), 16)))
maps = []
for l in open(f'/proc/{pid}/maps'):
    p = l.split()
    if len(p) >= 6 and p[5] == binpath:
        lo, hi = (int(x, 16) for x in p[0].split('-')); maps.append((lo, hi, p[1]))
res = {}
with open(f'/proc/{pid}/pagemap', 'rb') as pm:
    for lo, hi, perm in maps:
        for a in range(lo, hi, 4096):
            pm.seek(a // 4096 * 8)
            e = struct.unpack('<Q', pm.read(8))[0]
            if e >> 63: res[a] = perm
by = {}
for a in res:
    name = next((n for n, s, z in secs if s <= a < s + z), '?')
    by[name] = by.get(name, 0) + 4
print(' '.join(f"{k}={v}" for k, v in sorted(by.items(), key=lambda kv: -kv[1])))
if len(sys.argv) > 4:
    import bisect
    nm = [(int(a, 16), n) for a, n in (l.split()[:2] for l in open(sys.argv[4]) if len(l.split()) >= 2)]
    addrs = [a for a, _ in nm]
    txt = next((s, z) for n, s, z in secs if n == '.text')
    pages = sorted(a for a in res if txt[0] <= a < txt[0] + txt[1])
    runs = []
    for a in pages:
        if runs and a == runs[-1][1]: runs[-1][1] = a + 4096
        else: runs.append([a, a + 4096])
    print(f"  .text resident runs: {len(runs)}")
    for lo, hi in runs[:40]:
        i = bisect.bisect_right(addrs, lo) - 1
        print(f"    +{(lo - txt[0]) // 1024:6d} KB len {(hi - lo) // 1024:4d} KB at {nm[i][1][:90] if i >= 0 else '?'}")
PY
for p in trivial hill; do
  for v in A B; do
    printf '%s\n' "(load-file \"probe/rssprog/$p.clj\")" '(let [t (System/currentTimeMillis)] (loop [] (when (< (- (System/currentTimeMillis) t) 1500) (recur))))' > $out/hold-$p.clj
    $out/$v-bin run $out/hold-$p.clj < probe/rssprog/$p.in > /dev/null & pid=$!; sleep 1.2
    echo "$p $v resident KB by section:"; sudo python3 $out/res.py $pid $out/$v-bin $out/$v.sec $out/$v.nm
    wait $pid || true
  done
done
