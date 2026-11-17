"""Summarize a smaps dump (throwaway probe, #271): Rss in KB by mapping
kind, plus the binary's text/data section sizes."""
import collections, re, subprocess, sys

path, name, binary = sys.argv[1], sys.argv[2], sys.argv[3]
kinds = collections.Counter()
cur = None
for line in open(path, errors='replace'):
    m = re.match(r'^[0-9a-f]+-[0-9a-f]+ (\S+) \S+ \S+ \S+\s*(.*)$', line)
    if m:
        perms, pathname = m.group(1), m.group(2).strip()
        if pathname == binary or pathname.endswith('/' + binary.split('/')[-1]):
            cur = {'r-xp': 'text', 'r--p': 'rodata'}.get(perms, 'data')
        elif pathname in ('[heap]', '[stack]', '[vdso]', '[vvar]', '[vsyscall]'):
            cur = pathname.strip('[]')
        elif pathname == '':
            cur = 'anon'
        else:
            cur = 'other:' + pathname.split('/')[-1]
        continue
    m = re.match(r'^Rss:\s+(\d+) kB', line)
    if m and cur:
        kinds[cur] += int(m.group(1))
total = sum(kinds.values())
size = subprocess.run(['size', binary], capture_output=True, text=True).stdout.splitlines()[-1].split()
print(f'{name}: rss {total} KB = ' + ' '.join(f'{k} {v}' for k, v in kinds.most_common()) + f' | size text {size[0]} data {size[1]} bss {size[2]}')
