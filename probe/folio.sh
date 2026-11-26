#!/usr/bin/env bash
# Probe (throwaway, #271): is the stripped interpreter's higher RSS the
# strip, or per-file page-cache folio luck? One PR-head build, unstripped
# (U) and objcopy-stripped (S): six fresh copies of each, written two
# ways, each measured (median of 5) on two programs run by absolute path,
# plus the folio makeup of one copy's resident text.
set -euo pipefail
out=/tmp/folio; mkdir -p $out
uname -r; cat /sys/kernel/mm/transparent_hugepage/enabled; cat /sys/kernel/debug/fault_around_bytes 2>/dev/null || sudo cat /sys/kernel/debug/fault_around_bytes 2>/dev/null || echo "fault_around_bytes: n/a"
ref=$(sed -n 2p probe/rss-refs)
git checkout -q $ref -- rust scripts
bash scripts/bench/hot-text/regen-run.sh rust > $out/regen.log 2>&1 || { tail -20 $out/regen.log; exit 1; }
(cd rust && CARGO_PROFILE_RELEASE_STRIP=false cargo build --release --locked --no-default-features -p clojure-rs) > $out/build.log 2>&1 || { tail -30 $out/build.log; exit 1; }
cp rust/target/release/clojure-rs $out/U.src
objcopy --strip-all $out/U.src $out/S.src
git checkout -q HEAD -- rust scripts
root=$(pwd)
med() { sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }
rss() { for i in 1 2 3 4 5; do /usr/bin/time -f %M $1 run $root/probe/rssprog/$2.clj < $root/probe/rssprog/$2.in 2>&1 >/dev/null | tail -1; done | med; }
for how in cat dropped; do
  for k in 1 2 3 4 5 6; do
    for v in U S; do
      f=$out/$v-$how-$k; cat $out/$v.src > $f; chmod +x $f
      if [ $how = dropped ]; then sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null; fi
      echo "$how $k $v trivial=$(rss $f trivial) hill=$(rss $f hill)"
    done
  done
done | tee $out/rows.txt
python3 - $out/rows.txt <<'PY'
import sys, statistics as st, collections
d = collections.defaultdict(list)
for l in open(sys.argv[1]):
    how, k, v, *kv = l.split()
    for x in kv:
        p, n = x.split('='); d[(how, v, p)].append(int(n))
for how in ('cat', 'dropped'):
    for p in ('trivial', 'hill'):
        u, s = d[(how, 'U', p)], d[(how, 'S', p)]
        print(f"{how:8s} {p:8s} U med {st.median(u):7.0f} [{min(u)}..{max(u)}]  S med {st.median(s):7.0f} [{min(s)}..{max(s)}]  S-U {st.median(s)-st.median(u):+.0f}")
PY
# Folio makeup of one copy's resident executable pages (pagemap PFN + kpageflags).
cat > $out/folio.py <<'PY'
import sys, struct
pid, path = int(sys.argv[1]), sys.argv[2]
heads = tails = small = 0
with open(f'/proc/{pid}/pagemap', 'rb') as pm, open('/proc/kpageflags', 'rb') as kf:
    for l in open(f'/proc/{pid}/maps'):
        p = l.split()
        if len(p) < 6 or p[5] != path or 'x' not in p[1]: continue
        lo, hi = (int(x, 16) for x in p[0].split('-'))
        for a in range(lo, hi, 4096):
            pm.seek(a // 4096 * 8); e = struct.unpack('<Q', pm.read(8))[0]
            if not e >> 63: continue
            pfn = e & ((1 << 55) - 1)
            kf.seek(pfn * 8); f = struct.unpack('<Q', kf.read(8))[0]
            if f >> 15 & 1: heads += 1
            elif f >> 16 & 1: tails += 1
            else: small += 1
print(f"resident exec pages: {heads + tails + small} ({(heads + tails + small) * 4} KB); in large folios {heads + tails} (heads {heads}), order-0 {small}")
PY
for v in U S; do
  f=$out/$v-cat-1
  printf '%s\n' "(load-file \"$root/probe/rssprog/hill.clj\")" '(let [t (System/currentTimeMillis)] (loop [] (when (< (- (System/currentTimeMillis) t) 1500) (recur))))' > $out/hold.clj
  $f run $out/hold.clj < probe/rssprog/hill.in > /dev/null & pid=$!; sleep 1.2
  echo "$v hill: $(sudo python3 $out/folio.py $pid $f)"
  wait $pid || true
done
