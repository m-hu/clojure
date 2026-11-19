import json
import os

import gdb

pid = gdb.selected_inferior().pid
res = {'vmas': []}
for line in open(f'/proc/{pid}/status'):
    if line.startswith(('VmHWM:', 'VmRSS:', 'RssAnon:', 'RssFile:')):
        k, v = line.split(':', 1)
        res[k] = int(v.split()[0])
with open(f'/proc/{pid}/pagemap', 'rb') as pm:
    for line in open(f'/proc/{pid}/maps'):
        parts = line.split()
        lo, hi = (int(x, 16) for x in parts[0].split('-'))
        if lo >= 1 << 47:
            continue
        n = (hi - lo) >> 12
        pm.seek((lo >> 12) * 8)
        data = pm.read(n * 8)
        present = [i for i in range(n) if data[i * 8 + 7] & 0x80]
        res['vmas'].append({'lo': lo, 'hi': hi, 'perm': parts[1], 'off': int(parts[2], 16),
                            'path': parts[5] if len(parts) > 5 else '', 'present': present})
json.dump(res, open(os.environ['PM_OUT'], 'w'))
