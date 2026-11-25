"""Aggregate prog-delta summaries: each program's share by function bucket (throwaway, #271)."""
import collections, re, sys
cur = None; tot = {}; per = collections.defaultdict(collections.Counter)
for line in open(sys.argv[1]):
    m = re.match(r'== (\S+): (\d+) per run', line)
    if m: cur = m.group(1); tot[cur] = int(m.group(2)); continue
    m = re.match(r'\s+([\d.]+)\s+(.*)$', line)
    if m and cur: per[cur][m.group(2)] += float(m.group(1))
def bucket(fn):
    f = fn.lower()
    if re.search(r'malloc|free|_int_|tcache|realloc|calloc|unlink_chunk|consolidate|sbrk|mremap|memalign', f): return 'alloc'
    if 'drop' in f: return 'drop'
    if 'memcpy' in f or 'memmove' in f or 'memset' in f: return 'memcpy/set'
    if 'clone' in f: return 'clone'
    if 'hash' in f or 'sip' in f: return 'hash'
    if 'cljfn' in f or 'cljtop' in f: return 'compiled'
    if 'lazy' in f or 'seq' in f: return 'seq'
    if 'nabi' in f or 'rt_n_' in f or 'core_' in f or 'native_abi' in f: return 'core-call'
    return 'other'
agg = collections.Counter()
for k, c in per.items():
    b = collections.Counter()
    for fn, v in c.items(): b[bucket(fn)] += v
    t = tot[k] or 1
    print(k, tot[k], ' '.join(f'{x}={v / t:.0%}' for x, v in b.most_common()))
    for x, v in b.items(): agg[x] += v / t
print('MEAN SHARE', ' '.join(f'{x}={v / len(per):.0%}' for x, v in agg.most_common()))
