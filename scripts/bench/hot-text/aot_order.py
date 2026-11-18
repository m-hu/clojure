"""Hot-text order for AOT binaries from Linux profiles (#271).

`regen-aot.sh` drives three steps:

  aot_order.py cases RESOURCES SRC
      `script<TAB>case dir` for every test case directory: a script
      `SRC/REL.clj` owns `RESOURCES[/top]/REL/test_case_N`.
  aot_order.py entered            < llvm-profdata merge --text output
      The functions an entry-count instrumented runtime entered (one
      program's runs over its test inputs), one mangled name per line.
  aot_order.py callgrind FILE.cg
      Every Rust function a callgrind run executed (the startup set:
      std's precompiled functions carry no instrumentation).
  aot_order.py order --fns DIR --startup FILE --sizes FILE [--previous LD] --out LD
      The layout: the startup set first (in the previous order's
      sequence), then 62 KiB windows filled with functions programs use
      together (each window seeded with the most used unplaced function,
      then grown by the function whose set of using programs is most
      similar to the window's), so each program touches few 64 KiB
      windows of the hot section and so few pages.

A function the order misses still links, in the cold `.text`."""
import collections
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hot_text  # noqa: E402

WINDOW = 64 * 1024
CAP = 62 * 1024


def rust_name(name):
    """Whether `name` is a Rust mangled symbol (v0 or legacy)."""
    return re.fullmatch(r'(_R|_ZN)[A-Za-z0-9_.$]+', name) is not None


def cases(resources, src):
    """`(script rel, case dir)` pairs: a case dir belongs to the script
    whose relative path it spells under the resources root, directly or
    under one top-level directory."""
    scripts = set()
    for root, _, files in os.walk(src):
        for f in files:
            if f.endswith('.clj'):
                scripts.add(os.path.relpath(os.path.join(root, f), src)[:-4])
    pairs = []
    for root, dirs, _ in os.walk(resources):
        for d in sorted(dirs):
            if not d.startswith('test_case_'):
                continue
            rel = os.path.relpath(root, resources)
            owner = rel if rel in scripts else rel.split('/', 1)[-1] if '/' in rel else None
            if owner in scripts:
                pairs.append((owner, os.path.join(root, d)))
    return sorted(pairs)


def entered(lines):
    """Names with a non-zero entry count in `llvm-profdata merge --text`
    output (a local function's name carries its `file;` prefix)."""
    lines = list(lines)
    out, i = set(), 0
    while i < len(lines):
        if i + 1 < len(lines) and lines[i + 1] == '# Func Hash:' and lines[i] and not lines[i].startswith('#'):
            counters = int(lines[i + 4])
            if counters and int(lines[i + 6]) > 0:
                name = lines[i].split(';', 1)[-1]
                if rust_name(name):
                    out.add(name)
            i += 6 + counters
        else:
            i += 1
    return sorted(out)


def callgrind(path):
    """Rust functions with any cost in a callgrind output, in first-seen
    order (`fn=` ids resolve through their first definition)."""
    names, seen = {}, {}
    fn, after_call = None, False
    for line in open(path, errors='replace'):
        line = line.rstrip('\n')
        if line.startswith(('fn=', 'cfn=')):
            spec = line.split('=', 1)[1]
            m = re.match(r'\((\d+)\)\s*(.*)$', spec)
            name = spec
            if m:
                if m.group(2):
                    names[m.group(1)] = m.group(2)
                name = names.get(m.group(1), '?')
            if line.startswith('fn='):
                fn = name
        elif line.startswith('calls='):
            after_call = True
        elif line[:1].isdigit() or line[:1] in '+-*':
            if after_call:
                after_call = False
            elif fn and rust_name(fn):
                seen.setdefault(fn, None)
    return list(seen)


def load_sizes(path):
    """Symbol sizes by glob from `nm -S --size-sort -t d` output."""
    sizes = {}
    for line in open(path, errors='replace'):
        parts = line.split()
        if len(parts) == 4 and parts[2] in 'tTwW':
            sizes.setdefault(hot_text.glob(parts[3]), int(parts[1]))
    return sizes


def previous_rank(path):
    """Each glob's position in an existing order (`*(.text*.GLOB)` lines)."""
    rank = {}
    if path and os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line.startswith('*(.text*.') and line.endswith(')'):
                rank.setdefault(line[len('*(.text*.'):-1], len(rank))
    return rank


def layout(progs, startup, size, rank):
    """The function order: startup set first, then the filled windows."""
    count = collections.Counter(f for fns in progs.values() for f in fns)
    first = {}
    for name in sorted(progs):
        for f in sorted(progs[name]):
            first.setdefault(f, len(first))
    ids = {name: i for i, name in enumerate(sorted(progs))}
    users = collections.defaultdict(int)
    for name, fns in progs.items():
        for f in fns:
            users[f] |= 1 << ids[name]
    head = sorted(set(startup), key=lambda f: (rank.get(hot_text.glob(f), 1 << 30), startup.index(f)))
    placed = set(head)
    order = list(head)
    unplaced = set(count) - placed
    by_count = sorted(unplaced, key=lambda f: (-count[f], first[f]))
    while unplaced:
        seed = next(f for f in by_count if f in unplaced)
        window, used, union = [seed], size(seed), users[seed]
        unplaced.discard(seed)
        while True:
            best, score = None, 0.0
            for f in unplaced:
                if used + size(f) > CAP:
                    continue
                both = bin(users[f] & union).count('1')
                either = bin(users[f] | union).count('1')
                s = both / either if either else 0.0
                if s > score:
                    best, score = f, s
            if best is None:
                break
            window.append(best)
            used += size(best)
            union |= users[best]
            unplaced.discard(best)
        order += window
    return order


def windows(order, fns, size):
    """How many 64 KiB windows of the hot section `fns` touch."""
    pos, off = {}, 0
    for f in order:
        pos[f] = off
        off += size(f)
    touched = set()
    for f in fns:
        if f in pos:
            touched.update(range(pos[f] // WINDOW, (pos[f] + size(f) - 1) // WINDOW + 1))
    return len(touched)


def write_order(order, path, programs):
    globs, done = [], set()
    for f in order:
        g = hot_text.glob(f)
        if g not in done:
            done.add(g)
            globs.append(g)
    with open(path, 'w') as out:
        out.write('/* Generated by scripts/bench/hot-text/regen-aot.sh from %d program\n' % programs)
        out.write('   profiles; see scripts/bench/hot-text/aot_order.py. Do not edit. */\n')
        out.write('SECTIONS\n{\n  .text.hot :\n  {\n')
        for g in globs:
            # `.text*.` also takes `.text.unlikely.` (`#[cold]` fns).
            out.write('    *(.text*.%s)\n' % g)
        out.write('  }\n}\nINSERT BEFORE .text;\n')
    return len(globs)


def order_main(args):
    opts = dict(zip(args[::2], args[1::2]))
    fns_dir, out = opts['--fns'], opts['--out']
    progs = {}
    for f in sorted(os.listdir(fns_dir)):
        if f.endswith('.txt'):
            fns = [l.strip() for l in open(os.path.join(fns_dir, f)) if l.strip()]
            if fns:
                progs[f[:-4]] = set(fns)
    startup = [l.strip() for l in open(opts['--startup']) if l.strip()]
    sizes = load_sizes(opts['--sizes'])
    median = int(statistics.median(sizes.values())) if sizes else 256
    size = lambda f: sizes.get(hot_text.glob(f)) or median
    order = layout(progs, startup, size, previous_rank(opts.get('--previous')))
    n = write_order(order, out, len(progs))
    per = sorted(windows(order, fns | set(startup), size) for fns in progs.values())
    print('%d programs, %d functions -> %d globs in %s; windows per program median %d, p90 %d'
          % (len(progs), len(order), n, out, statistics.median(per), per[9 * len(per) // 10]))


def main():
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == 'cases':
        print('\n'.join('%s\t%s' % p for p in cases(rest[0], rest[1])))
    elif cmd == 'entered':
        print('\n'.join(entered(sys.stdin.read().split('\n'))))
    elif cmd == 'callgrind':
        print('\n'.join(callgrind(rest[0])))
    elif cmd == 'order':
        order_main(rest)
    else:
        sys.exit('usage: aot_order.py cases RES SRC | entered | callgrind FILE.cg | order --fns DIR --startup FILE --sizes FILE [--previous LD] --out LD')


if __name__ == '__main__':
    main()
