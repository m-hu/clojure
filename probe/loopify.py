"""Make a snippet loop its driver: (slurp *in*) -> INPUT (read once), and
the last top-level form repeated N times. usage: loopify.py SRC OUT N"""
import re, sys
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
s = open(src).read()
# top-level form spans (rough reader: strings, chars, comments)
forms, depth, start, i = [], 0, None, 0
while i < len(s):
    c = s[i]
    if c == ';':
        while i < len(s) and s[i] != '\n': i += 1
        continue
    if c == '"':
        i += 1
        while i < len(s) and s[i] != '"': i += 2 if s[i] == '\\' else 1
    elif c == '\\': i += 2; continue
    elif c in '([{':
        if depth == 0: start = i
        depth += 1
    elif c in ')]}':
        depth -= 1
        if depth == 0: forms.append((start, i + 1))
    i += 1
if not forms or '(slurp *in*)' not in s:
    sys.exit(2)
a, b = forms[-1]
last = s[a:b].replace('(slurp *in*)', 'INPUT')
body = s[:a].replace('(slurp *in*)', 'INPUT')
# INPUT goes after the ns/require form (the first form)
fa, fb = forms[0]
first = body[fa:fb]
if first.startswith('(ns ') or first.startswith('(require'):
    body = body[:fb] + '\n(def INPUT (slurp *in*))\n' + body[fb:]
else:
    body = '(def INPUT (slurp *in*))\n' + body
open(out, 'w').write(body + '(dotimes [_ %d] %s)\n' % (n, last) + s[b:])
