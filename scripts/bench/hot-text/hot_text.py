#!/usr/bin/env python3
"""Generate `rust/hot-text.ld` (#226): the functions a typical run
executes, packed at the front of `.text`.

usage: hot_text.py BINARY OUT.ld CALLGRIND_OUT...

A run's peak RSS counts every binary page it touches, and the kernel
maps file pages in 64 KB windows. The functions a trivial run executes
total about 1 MB, but the linker scatters them over most of the 6 MB
`.text`, so a run touched 5 MB of it. The script lists the executed
functions (most workloads first) as input-section globs, so the linker
places them in one `.text.hot` output section ahead of `.text`.

Each glob keeps the symbol's path identifiers as mangled (length
prefix included) and wildcards everything between them, so the crate
disambiguators (which hash the compiler version), v0 backreference
offsets and the legacy hash suffix drop out and the script survives
toolchain updates:

    _RNvMs_NtCscx6VyDXTacY_8clj_eval4exprNtB4_8Analyzer15analyze_special
    -> .text.*8clj_eval*4expr*8Analyzer*15analyze_special*

A glob can also match other instantiations of a generic function; they
join the hot section too (small, and harmless). Functions that a
workload never ran stay in `.text`.
"""
import collections
import os
import re
import sys

B62 = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
BASIC = set('abcdefhijklmnopstuvxyz')


class Bad(Exception):
    pass


class V0:
    def __init__(self, s):
        self.s, self.i, self.out = s, 0, []

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else ''

    def eat(self, c):
        if self.peek() == c:
            self.i += 1
            return True
        return False

    def b62(self):
        # <base-62-number> = { [0-9a-zA-Z] } "_"
        while self.peek() and self.peek() != '_':
            if self.peek() not in B62:
                raise Bad('b62')
            self.i += 1
        if not self.eat('_'):
            raise Bad('b62 end')

    def disambiguator(self):
        # Only called where a path or identifier starts, which never
        # begin with `s` themselves (the `i16` basic type `s` only
        # appears in type position).
        if self.eat('s'):
            self.b62()

    def ident(self):
        self.disambiguator()
        start = self.i
        self.eat('u')
        # <decimal-number> = "0" | [1-9] {[0-9]} (no leading zeros:
        # nested closures mangle as `...00`)
        m = re.match(r'0|[1-9]\d*', self.s[self.i:])
        if not m:
            raise Bad('ident')
        n = int(m.group(0))
        self.i += m.end()
        self.eat('_')
        body = self.s[self.i:self.i + n]
        if len(body) != n:
            raise Bad('ident len')
        self.i += n
        if n:
            self.out.append(self.s[start:self.i])

    def path(self):
        c = self.peek()
        self.i += 1
        if c == 'C':
            self.ident()
        elif c == 'M':
            self.disambiguator()
            self.path()
            self.type_()
        elif c == 'X':
            self.disambiguator()
            self.path()
            self.type_()
            self.path()
        elif c == 'Y':
            self.type_()
            self.path()
        elif c == 'N':
            if not self.peek().isalpha():
                raise Bad('ns')
            self.i += 1
            self.path()
            self.ident()
        elif c == 'I':
            self.path()
            while not self.eat('E'):
                self.generic_arg()
        elif c == 'B':
            self.b62()
        else:
            raise Bad('path ' + c)

    def generic_arg(self):
        if self.eat('L'):
            self.b62()
        elif self.eat('K'):
            self.const()
        else:
            self.type_()

    def const(self):
        if self.eat('p'):
            return
        if self.eat('B'):
            self.b62()
            return
        self.type_()
        self.eat('n')
        while self.peek() and self.peek() in '0123456789abcdef':
            self.i += 1
        if not self.eat('_'):
            raise Bad('const')

    def fn_sig(self):
        if self.eat('G'):
            self.b62()
        self.eat('U')
        if self.eat('K'):
            if self.peek() == 'C':
                self.i += 1
            else:
                self.ident()
        while not self.eat('E'):
            self.type_()
        self.type_()

    def type_(self):
        c = self.peek()
        if c in BASIC:
            self.i += 1
        elif c in 'CMXYNI':
            self.path()
        elif c == 'B':
            self.i += 1
            self.b62()
        elif c == 'A':
            self.i += 1
            self.type_()
            self.const()
        elif c == 'S':
            self.i += 1
            self.type_()
        elif c in 'RQ':
            self.i += 1
            if self.eat('L'):
                self.b62()
            self.type_()
        elif c in 'PO':
            self.i += 1
            self.type_()
        elif c == 'F':
            self.i += 1
            self.fn_sig()
        elif c == 'D':
            self.i += 1
            if self.eat('G'):
                self.b62()
            while not self.eat('E'):
                self.path()
                while self.eat('p'):
                    self.ident()
                    self.type_()
            if not self.eat('L'):
                raise Bad('dyn lifetime')
            self.b62()
        elif c == 'T':
            self.i += 1
            while not self.eat('E'):
                self.type_()
        else:
            raise Bad('type ' + c)


def tokens(sym):
    """The path identifiers of a mangled symbol, as mangled (or None)."""
    base = re.sub(r'\.llvm\.\d+$', '', sym)
    if base.startswith('_R'):
        p = V0(base[2:])
        m = re.match(r'\d+', p.s)
        if m:
            p.i = m.end()
        try:
            p.path()
            if p.i < len(p.s) and p.peek() in 'CMXYNIB':
                p.path()  # instantiating crate
        except (Bad, IndexError, RecursionError):
            return None
        return p.out
    if base.startswith('_ZN'):
        s = re.sub(r'17h[0-9a-f]{16}E$', '', base[3:])
        out, i = [], 0
        while i < len(s):
            m = re.match(r'\d+', s[i:])
            if not m:
                return None
            n = int(m.group(0))
            j = i + m.end()
            out.append(s[i:j + n])
            i = j + n
        return out
    return None


def glob(sym):
    """The toolchain-stable glob for `sym` (exact when unparsed, or
    when one identifier alone would match a whole crate)."""
    t = tokens(sym)
    if not t or len(t) < 2:
        return sym
    return '*' + '*'.join(t) + '*'


def resolve(table, spec):
    """A callgrind name spec: `(id) name` defines, `(id)` refers."""
    m = re.match(r'\((\d+)\)\s*(.*)$', spec)
    if not m:
        return spec
    if m.group(2):
        table[m.group(1)] = m.group(2)
    return table.get(m.group(1), '?')


def executed(path, base):
    """Functions of the object named `base` with self cost in a
    callgrind output, in order of appearance."""
    obs, fns, seen = {}, {}, {}
    ob = fn = None
    skip = False
    with open(path, errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('ob='):
                ob = resolve(obs, line[3:])
            elif line.startswith('cob='):
                resolve(obs, line[4:])
            elif line.startswith('fn='):
                fn = resolve(fns, line[3:])
            elif line.startswith('cfn='):
                resolve(fns, line[4:])
            elif line.startswith('calls='):
                skip = True  # the next cost line is the call's inclusive cost
            elif line[:1].isdigit() or line[:1] in '+-*':
                if skip:
                    skip = False
                elif fn and ob and os.path.basename(ob) == base:
                    seen.setdefault(fn, len(seen))
    return seen


def main():
    binary, out, profiles = sys.argv[1], sys.argv[2], sys.argv[3:]
    base = os.path.basename(binary)
    runs = collections.Counter()
    first = {}
    for path in profiles:
        for name in executed(path, base):
            runs[name] += 1
            first.setdefault(name, len(first))
    names = sorted(runs, key=lambda n: (-runs[n], first[n]))
    globs, done = [], set()
    for name in names:
        if not re.fullmatch(r'[A-Za-z0-9_.$]+', name):
            continue  # callgrind pseudo-entries such as `(below main)`
        g = glob(name)
        if g not in done:
            done.add(g)
            globs.append(g)
    with open(out, 'w') as f:
        f.write('/* Generated by scripts/bench/hot-text/run.sh from %d workload\n' % len(profiles))
        f.write('   profiles; see scripts/bench/hot-text/hot_text.py. Do not edit. */\n')
        f.write('SECTIONS\n{\n  .text.hot :\n  {\n')
        for g in globs:
            f.write('    *(.text.%s)\n' % g)
        f.write('  }\n}\nINSERT BEFORE .text;\n')
    print('%d functions from %d profiles -> %d globs in %s' % (len(names), len(profiles), len(globs), out))


if __name__ == '__main__':
    main()
