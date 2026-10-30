# T1174 S2 reader contract: JVM `LispReader` behavior to replicate

Status: normative for slice S2 (`clj-reader` + `clj-value` printer/value
changes + differential corpus harness). Every rule below is backed by JVM
source reads (`src/jvm/clojure/lang/LispReader.java`, plus `RT.java`,
`Util.java`, `Compiler.java`, `LineNumberingPushbackReader.java`,
`PersistentArrayMap.java`, `PersistentHashSet.java`,
`src/clj/clojure/core_print.clj`) and by oracle probes
(`docs/t1174/probes/probe*.clj`, run via `docs/t1174/probes/probe.sh`
against `target/clojure-1.13.0-master-SNAPSHOT.jar`). Probe citations look
like `(probe3 POS unread-nl)`.

Conventions: `L` = 1-based line, `C` = 1-based column of the next char to
read (col 0 is observable after `unread('\n')`, see §1). `(L,C)` after an
input is the `#:clojure.error{:line :column}` ex-data of the top-level
`ReaderException`. `[NFE]` = root cause `NumberFormatException`, `[IAE]` =
`IllegalArgumentException`, `[ISE]` = `IllegalStateException`, `[RE]` =
`RuntimeException`, `[AE]` = `ArithmeticException`. The differential
harness compares **root-cause message + root-cause simple class name**;
our `ReaderError` must expose both (`message()` and `jvm_class()`).

S2 default read options are **load semantics**: `:read-cond :allow`,
`:features #{:clj}` (JVM always conjs `:clj`, `installPlatformFeature`).
S5's `read-string` will pass deny explicitly. S2 honored `*read-eval*`
only as: `#=` always errored (N5 deferral, §16); the `:unknown` check
and `false` branch had no S2 equivalent (no dynamic bindings yet).
S7+ threads the live binding through `ReaderOpts.read_eval`: closed
⇒ the JVM gate triple, open ⇒ evaluator-hook application (§16).
The harness reads with the gate closed, like the driver's
`(binding [*read-eval* false] …)` (#125).

## §1 Char stream, positions, unread

`LineNumberingPushbackReader` wraps `LineNumberReader`:

- **Normalization**: CR, LF, CRLF all yield a single `\n`
  (probe3 `POS crlf/cr/a-cr-b/a-crlf-b`). `"\r"` never reaches the reader:
  `"a\rb"` reads as `"a\nb"`; `;c\rd` ends the comment, then reads `d`
  (probe3). Our char iterator must collapse `\r\n` and lone `\r` to `\n`.
- **Line** starts 1, increments when a `\n` is *read from the underlying
  stream* (re-reads from the pushback buffer do not bump the line).
- **Column** starts 1. `read(c)`: `c == '\n'` or EOF sets col 1, else
  col++. `unread(c)` (never called with -1; at most one char pending)
  does col-- **unconditionally, even for `\n`** (line unchanged).
  Observable: `foo:\n1` errors at (2,0), `1a\n` at (2,0) (probe3).
  List/meta colons capture `col_after_open - 1` (1-based col of `(`/`^`).
- **EOF bump**: reading EOF sets col 1 and bumps the line **iff a
  non-`\n` char was read since the last `\n`/EOF/start** (i.e. once per
  content run; repeated EOF reads do not bump again). So `""`→(1,1),
  `"   "`→(2,1), `"(a\nb"`→(3,1), `"a\r\nb`(unterminated)→(3,1)
  (probe3 `POS *`, probe4). Re-reads from pushback never set the
  bumpable flag. (This quirk lives in `LineNumberReader`; replicate the
  observed state machine, not the mechanism.)
- Our `Reader` tracks `(line, col, bumpable, pushback: Option<char>)`
  exactly per the above. `atLineStart` is only used by `clojure.main`
  echo, not the read path: do not port it.

## §2 Whitespace

`isWhitespace(ch) = Character.isWhitespace(ch) || ch == ','`, where
`Character.isWhitespace` is **not** Unicode `White_Space`: excluded are
U+0085, U+00A0, U+2007, U+202F (probe1 `ISWS`, probe2/4). Exact set:

```
09 0A 0B 0C 0D  1C 1D 1E 1F  20
1680  2000 2001 2002 2003 2004 2005 2006 2008 2009 200A
2028 2029  205F  3000  002C (,)
```

plus EOF handling. Behavioral check `1<ch>2` splits into two forms iff
`ch` is in the set, else merges into one token (`Invalid number: 1<ch>2`,
probe2 `WS-*`). Implement as an explicit table (not
`char::is_whitespace`, which wrongly includes 0085/00A0/2007/202F and
excludes nothing here).

## §3 `read()` core loop

Per call (recursive; `pending` is a queue on our `Reader`):

1. If `pending` non-empty, pop front and return it (before touching chars).
2. Skip whitespace (comment/discard/`#?`-no-match return the reader
   sentinel ⇒ `continue`, i.e. recur for the next form).
3. EOF: `eofIsError` ⇒ `[RE] "EOF while reading"`, else return the EOF
   value. Multi-form `read_all` loops with a private EOF sentinel.
4. `returnOn` delim char (inside delimited lists) ⇒ return its value.
5. `Character.isDigit(ch)` ⇒ `readNumber`. Note: this is Unicode Nd, so
   non-ASCII decimal digits take the number path and then fail with
   `Invalid number`. Our port uses `is_ascii_digit || is_numeric`
   (documented edge: Nl/No chars like Roman numerals diverge, §24).
6. Macro char ⇒ invoke with `(reader, ch, opts, pending)`; reader
   sentinel ⇒ `continue`, else return.
7. `+`/`-` followed by `isDigit` ⇒ `readNumber` (unread the digit
   first); else fall through to token.
8. Else `readToken` + `interpretToken`.

`ReaderException(line, col, cause)` wrapping happens only at the
non-recursive top level; recursive frames rethrow untouched. Our port:
every `ReaderError` carries its span directly; no wrapper type. The
harness asserts spans (line exact, col exact — the tracker above makes
them match, including col 0 and the EOF bump).

## §4 Tokens, symbols, keywords

`readToken(r, initch)`: accumulate until EOF/whitespace/**terminating**
macro; `#`, `'`, `%` are macros but **non-terminating**, so `a'b`, `a#b`
are single tokens ⇒ symbols (probe1).

`interpretToken(s)`: `nil`⇒nil, `true`/`false`⇒bools, else
`matchSymbol(s)` **only** (numbers never come through here — they use
`readNumber`). Failure ⇒ `[RE] "Invalid token: {s}"`.

`matchSymbol(s)` replicates, in order (probes: `a/b/c`, `foo:/bar`,
`:a::b`, `ab/12`, `::`, `:`, `a:`, `:a:`, `a::b`, `//`, `a/`, `/a`,
`/1`, `a/1`, `:1`, `:1a`, `/.5`, `a/b/`, `:a/`, `1:2`):

1. `symbolPat = [:]?([\D&&[^/]].*/)?(/|[\D&&[^/]][^/]*)` full match,
   with Java `\D` = ASCII `[^0-9]` (so the first char class admits `:`
   and non-ASCII, but not ASCII digits or `/`; the tail `[^/]*` admits
   digits and colons — hence `:1`, `:1a`, `a1:b2` are legal).
   Hand-roll this matcher; do not pull a regex engine for it.
2. Reject (return null ⇒ `Invalid token`) if: `ns` group ends with `:/`
   (e.g. `foo:/bar`); `name` ends with `:`; or `s.indexOf("::", 1) != -1`.
3. `s.startsWith("::")`: strip `::`, split on first `/` into
   `alias/name` or `name`. Resolve: alias via alias map, else current
   ns; unknown ⇒ `Invalid token` (e.g. `::nope/x`, `::clojure.string/foo`
   — a qualified name is looked up as an *alias*, probe1).
   Note `::` alone is rejected by rule 2 first.
4. Else `:`-leading ⇒ keyword, else symbol, via **`Symbol::intern`
   semantics: split on the FIRST `/`** (`a/b/c` ⇒ ns `a`, name `b/c`;
   `/` alone ⇒ no-ns name `/`). Our `Symbol::new` stays strict for API
   use; add `Symbol::intern(&str)` with exactly this split (no `/`
   rejection), and build keywords from it.
5. If `symbolPat` fails, try
   `arraySymbolPat = ([\D&&[^/:]].*)/([1-9])` (single trailing digit
   1-9): `a/1` ⇒ symbol, `ab/12` ⇒ `Invalid token` (probe1 — the greedy
   `.*` cannot donate its `/`).

## §5 Numbers

`readNumber(r, initch)`: accumulate until EOF/whitespace/**any** macro
(including `(`, `[`, `{`, `#`, `'`, `%` — so `1(2)` reads `1` then a
list), then `matchNumber(s)`; null ⇒ `[NFE] "Invalid number: {s}"`.

`matchNumber(s)` — order matters, and the int-pattern match
**short-circuits** (no float/ratio retry once `intPat` matches):

- `intPat = ([-+]?)(?:(0)|([1-9][0-9]*)|0[xX]([0-9A-Fa-f]+)|0([0-7]+)|([1-9][0-9]?)[rR]([0-9A-Za-z]+)|0[0-9]+)(N)?`
  - `(0)` alone ⇒ 0 (our `Int(0)`); `(0)` + `N` ⇒ BigInt deferral (§22).
  - Decimal/hex/octal/radix digits resolve per radix; `N` ⇒ BigInt
    deferral. Radix prefix `[1-9][0-9]?` (1–99); radix outside 2–36 ⇒
    `[NFE] "Radix out of range"` (probe1 `37r10`, `1r1` — exact JDK
    `BigInteger` message). Bad digit ⇒
    `[NFE] "For input string: \"{digits}\" under radix {radix}"`
    (digits part only, probe1 `2r2`). Both replicate JDK messages
    exactly — replicate, do not reuse a bignum crate's errors.
  - The bare `0[0-9]+` alternative is **poison**: it matches invalid
    octal (`08`, `09`, `018`, `08N`) so the token fails as
    `Invalid number` instead of parsing as float (probe1). `00`, `007`,
    `00N` take the *octal* alternative instead (`00`⇒0, `00N`⇒BigInt
    deferral).
  - Overflow: magnitude past i64 (`> i64::MAX` positive, `> 2^63`
    negative incl. `== 2^63` ⇒ `i64::MIN` ok) ⇒ BigInt deferral
    (probe3 boundaries, probe5/8 radix boundaries). Equivalently
    `bitLength < 64`. Accumulate in u128 with early-out; never panic.
- Else `floatPat = ([-+]?[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?)(M)?`:
  `M` ⇒ BigDecimal deferral; else `f64` via Rust `parse` (matches
  `Double.parseDouble` on this grammar, incl. `5.`⇒5.0, `1e309`⇒inf,
  subnormals; probe1). Leading zeros fine (`01.5`, `00.5`).
- Else `ratioPat = ([-+]?[0-9]+)/([0-9]+)` ⇒ ratio deferral (§22).
  Recognize the shape (do not report `Invalid number`): `1/2`, `+1/2`,
  `007/2` all defer. (JVM would reduce/construct; `5/0` ⇒ `[AE]
  "Divide by zero"`, `0/5`⇒0, `2/4`⇒`1/2` — pinned by probes for the
  follow-up, but S2 errors uniformly.)
- Else `[NFE] "Invalid number: {s}"` (`1a`, `1e`, `1e+`, `5.e`, `1N2`,
  `0x`, `0xG`, `10r`, `1r`, `200r10`, `-5-`, `1/2x`, `5.5.5`, `1:2`).

S2 produces `Int(i64)` for all in-range integers (JVM's Integer-vs-Long
split is invisible to printing; S3 introduces `Integer(i32)` for `type`
fidelity — the reader call site must be easy to find).

## §6 Characters (`\`)

`CharacterReader`: `read1` (EOF ⇒ `[RE] "EOF while reading character"`),
then `readToken` from that char:

- Length 1 ⇒ the char (`\o`⇒`o`, `\9`, `\,`, `\ `⇒space via token `" "`,
  probe1/3/4). **The length check comes before `u`/`o` handling**, so
  bare `\u`⇒`u` and bare `\o`⇒`o` (probe1).
- Named: `newline space tab backspace formfeed return` ⇒ `\n \  \t \b
  \f \r`.
- `u` + exactly 4 hex (`readUnicodeChar`, length must match exactly):
  `\u0041`⇒`A`; short/long ⇒ `[IAE] "Invalid unicode character: \{token}"`
  (`\u12`, `\u004`, `\u0041x`); bad digit ⇒ `[IAE] "Invalid digit: {c}"`.
  Surrogate halves ⇒ `[RE] "Invalid character constant: \u{hex}"` with
  **lowercase** hex (`\uD800` ⇒ `...\ud800`, probe1).
- `o` + up to 3 octal digits: `\o12`⇒newline, `\o0`⇒NUL, `\o7`⇒BEL,
  `\o123`⇒`S`; length 4+ ⇒ `[RE] "Invalid octal escape sequence length:
  {len}"` (len counts digits after `o`, probe1 `\o1234`); bad digit ⇒
  `[IAE] "Invalid digit: {c}"` (`\o8`); value > 0o377 ⇒
  `[RE] "Octal escape sequence must be in range [0, 377]."`.
- Else `[RE] "Unsupported character: \{token}"` (`\ab`, `\1234`).

## §7 Strings (`"`)

`StringReader` loop with escapes; unterminated ⇒ `[RE] "EOF while
reading string"` (also right after a backslash at EOF):

- Simple: `t r n \ " b f` ⇒ control chars. Anything else:
- `u`: first hex digit must satisfy `Character::digit(ch,16)` else
  `[RE] "Invalid unicode escape: \u{c}"` showing **only that char**
  (`\uZ`; EOF after `\u` renders U+FFFF: `"a\u` ⇒ `Invalid unicode
  escape: \u￿`, probe1). Then 3 more hex via the pushback rule below
  with exact=true: short ⇒ `[IAE] "Invalid character length: {i},
  should be: 4"` (`"\u12"` ⇒ length 2, col confirmed); bad digit ⇒
  `[IAE] "Invalid digit: {c}"`.
- ASCII-decimal digit ⇒ octal via pushback rule, base 8, max 3 digits,
  exact=false: `"\1"`⇒SOH, `"\0"`⇒NUL; `"\18"` ⇒ `[IAE] "Invalid digit:
  8"` (a non-octal digit *errors*, it does not terminate the escape);
  value > 0o377 ⇒ `[RE] "Octal escape sequence must be in range
  [0, 377]."` (`"\400"`).
- Else `[RE] "Unsupported escape character: \{c}"` (`\z`).
- Pushback escape rule: the first digit is already consumed; further
  chars stop (unread, break) at EOF/whitespace/macro; a non-digit
  errors `Invalid digit`. Note `"` is a macro, so `"\u12"` stops
  correctly (probe1 col 6 ⇒ unread-then-throw discipline, §1).
- **Lone surrogates diverge**: JVM accepts `"\ud800"` (1 char) and
  `"\ud800\udc00"` (2 chars, probe1 SURR-COUNT). Rust `str` cannot
  represent them: S2 raises our loud error (§22), catalogued in §24.
  No CESU-8 pairing.
- CR/CRLF inside strings arrive as `\n` via §1 (`"a\rb"`⇒`"a\nb"`).
- Unprintable ASCII passes through raw (`"\1"` prints raw SOH, S1
  `print_string` already matches).

## §8 Regex (`#"`)

`RegexReader` does **not** reuse `StringReader`: raw loop appending chars
verbatim; `\` + next char appended as a verbatim pair (so `#"\n"` holds
backslash+`n`, never newline). EOF ⇒ `[RE] "EOF while reading regex"`
(distinct message, probe1). S2 stores the source **without compiling**
(no `regex` dependency): invalid patterns like `#"(("` read fine here
but throw `PatternSyntaxException` (`Unclosed group near index 2\n((`)
on the JVM — catalogued divergence, §24. Rationale: a Rust regex engine
would falsely reject Java-valid patterns (possessives, backrefs).

## §9 Comments and discard

- `;` and `#!`: skip to `\n` (stream never yields `\r`, §1) or EOF,
  return the reader sentinel (recur). `#!sheb\n1`⇒`1`; `#!sheb` at EOF
  ⇒ `EOF while reading`; `1 #!c\r2`⇒`1 2` (probe1/4).
- `#_`: read one form with `eofIsError=true` (`#_` at EOF ⇒
  `EOF while reading`), discard, return the sentinel (recur).
  `#_ #_ 1 2 3`⇒`3`; `#_(`⇒EOF-starting-line-1 (probe1).

## §10 Wrapping readers

Each reads one form (`eofIsError=true`, recursive) and wraps:

- `'` ⇒ `(quote X)` — `quote` **unqualified**.
- `@` ⇒ `(clojure.core/deref X)` — **qualified** (the `@!`TentativeReader
  is commented out upstream: `@!x` ⇒ `(clojure.core/deref !x)`, probe1).
- `#'` ⇒ `(var X)` — unqualified.
- `~X` ⇒ `(clojure.core/unquote X)`; `~@X` ⇒
  `(clojure.core/unquote-splicing X)` — qualified. `~` at EOF ⇒
  `[RE] "EOF while reading character"`; `~@` at EOF ⇒
  `[RE] "EOF while reading"` (probe1). Outside syntax-quote these read
  fine (no error).

## §11 Metadata (`^`, `#'` shares nothing)

`MetaReader`: capture `(line, col_after_^ - 1)`; read meta form, coerce:

- Symbol/String ⇒ `{:tag m}`; Keyword ⇒ `{m true}`;
  Vector ⇒ `{:param-tags m}`; Map ⇒ as-is.
- Else `[IAE] "Metadata must be Symbol,Keyword,String,Vector or Map"`
  (`^1 'x`, probe1).
- Read the object; attach only if it supports meta, else
  `[IAE] "Metadata can only be applied to IMetas"` (`^:a 1`, `^:a :k`,
  `^:a "s"` — **keywords and strings do NOT take meta**, probe4/5).
  Meta lives on: Symbol, List, Vector, Map, Set (§21).
- If the object is a seq (list), assoc `:line`/`:column` (captured
  above) unless already present. `^String [x]` gets no line/col;
  `(a)` reads with `{:line 1, :column 1}` (probe1 META).
- Merge: new entries assoc over the object's existing meta, so
  `^:a ^:b 'x` ⇒ `{:b true, :line 1, :column 1, :a true}` (inner first;
  key order is insertion (array-map) on the JVM vs Fx-hash on ours —
  the harness canonicalizes map order, §23). No `IReference` in S2.

## §12 Fn literals (`#(...)`)

- Nested `#()` (ARG_ENV bound) ⇒ `[ISE] "Nested #()s are not allowed"`
  (`#(#(%))`, col after the inner `#(`).
- Bind fresh ARG_ENV (empty); unread `(`; read the list form.
- `%` tokens inside: `argPat = %([&]|[1-9][0-9]*)?`. `%`⇒arg 1,
  `%&`⇒rest(-1), `%12`⇒arg 12; anything else (`%0`, `%00`, `%1x`)
  ⇒ `[ISE] "arg literal must be %, %& or %integer"`.
  `registerArg(n)`: memo per number; fresh ⇒
  `p{1..}#` / `rest#` as `Symbol(null, "p{n}__{id}#")` /
  `Symbol(null, "rest__{id}#")` with the global id counter (§13 ids).
- Outside `#()`, `%...` is an ordinary symbol (`%`, `%&`, `%1`, `%00`
  all read as symbols, probe1/4).
- Expansion: `(fn* [args] form)` — head is **`fn*` unqualified**
  (`Compiler.FN`). Args: `1..higharg` (gaps filled with fresh `garg(i)`
  **after** the read, so gap ids sort last), then `& restsym` if `%&`
  present. `#(f %3 %)` ⇒
  `(fn* [p1__245# p2__246# p3__244#] (f p3__244# p1__245#))` (probe1 id
  order); `#(%2)` ⇒ `[p1 p2]`; `#(%&)` ⇒ `[& rest]`; `#()` ⇒ `(fn* [] ())`.
- Gensym ids: process-global `AtomicU64` starting at **1**
  (`RT.id = new AtomicInteger(1)`, `getAndIncrement`; first observed
  `#(f %)` id varies by session since boot consumes ids). The harness
  normalizes `__\d+#` and `__\d+__auto__`. Meta inside the body (e.g.
  `#(^:k %)`) attaches to the body occurrence only; the argvec keeps
  the un-meta'd registered symbol (probe4 PRINTMETA4).
- `x#` inside `#()` is a plain symbol (no gensym interaction, probe4).

## §13 Syntax-quote (`` ` ``)

Fresh gensym env per `` ` `` form (nesting rebinds: ``` ``x ```
inner-expands first with its own env). `syntaxQuote(form)`:

- Specials (exact set — `Compiler.specials` keys) ⇒ `(quote sym)`
  **without** qualification: `def loop* recur if case* let* letfn*
  do fn* quote var clojure.core/import* . set! deftype* reify* try
  catch finally throw monitor-enter monitor-exit new &`.
  Note `fn` is NOT special (`` `fn `` ⇒ `(quote clojure.core/fn)` via
  refers, probe1); `import*` alone is not special (only qualified).
- Symbol `s#` (unqualified, trailing `#`) ⇒ memoized gensym
  `s__{id}__auto__` (same `s#` twice ⇒ same gensym, probe4);
  outside syntax-quote `x#` is a plain symbol.
- Unqualified `C.` ⇒ resolve class (S2: none) then re-add `.`, quote
  (`C.` ⇒ `(quote C.)`, probe1). Leading-dot `.m` ⇒ quote as-is.
- Unqualified other ⇒ resolve class, then var (S2: harness-provided
  maps, §23; **empty in native unit tests**), else qualify with
  current ns (`x` ⇒ `(quote user/x)`). The JVM's null-resolver path
  additionally resolves referred vars (`map`⇒`clojure.core/map`) and
  imported classes (`` `String `` ⇒ `(quote java.lang.String)`,
  probe2) — the harness supplies these maps from the live `*ns*`,
  so S2 matches with no divergence (§23). Without maps (native
  tests), everything unqualified lands in the current ns.
- Qualified `a/b` ⇒ resolve class/alias on `a`; hit ⇒ rewrite the ns
  part, miss ⇒ leave as-is (`a/b` ⇒ `(quote a/b)`, probe1).
- `~X` (list headed `clojure.core/unquote`) ⇒ `X` raw (`` `~x `` ⇒ `x`);
  top-level `~@X` ⇒ `[ISE] "splice not in list"`.
- Map ⇒ `(clojure.core/apply clojure.core/hash-map (clojure.core/seq
  (clojure.core/concat ...)))`; vector ⇒ `.../vector ...`; set ⇒
  `.../hash-set ...`; seq/list ⇒ `(clojure.core/seq
  (clojure.core/concat ...))`; empty list ⇒ `(clojure.core/list)`.
  Items via `sqExpandList`: `~i` ⇒ `(clojure.core/list {second})`,
  `~@i` ⇒ `{second}` as-is, else `(clojure.core/list {sq(i)})`
  (`` `(a ~b ~@c) ``, `` `[a ~b] ``, `` `{:a ~b} ``, `` `#{a} ``,
  `` `() ``, `` `[~@xs] `` exact shapes in probe1/4).
- Atoms: Keyword/Number/Char/String pass through untouched (incl.
  non-finite doubles: `` `##Inf `` ⇒ `##Inf`, probe4). **Everything
  else ⇒ `(quote X)`**: booleans, nil, regexes
  (`` `true `` ⇒ `(quote true)`, `` `nil `` ⇒ `(quote nil)`,
  `` `#"a" `` ⇒ `(quote #"a")`, probe2).
- Records pass through (no records in S2 — unreachable).
- If the form carries non-line/col meta, wrap:
  `(clojure.core/with-meta {ret} {sq(meta-minus-line-col)})`
  (`` `^:k x `` ⇒ `(with-meta (quote user/x) (apply hash-map ...))`
  with `(quote true)` inside, probe1).

## §14 Reader conditionals (`#?`, `#?@`)

Options: `:read-cond` must be `:allow` or `:preserve`, else
`[RE] "Conditional read not allowed"` (thrown **before** reading the
body, col after `#?`). Then optional `@` (splicing), skip whitespace
(`#? (:clj 1)` ok), require `(` else
`[RE] "read-cond body must be a list"`.

`readCondDelimited` — replicate the loop shape, not "collect pairs":

```
result = STARTED
loop {
    if result == STARTED {
        f = read-or-FINISHED-or-EOF
        EOF ⇒ "EOF while reading[, starting at line N]"
        FINISHED ⇒ break                       // odd tail silently dropped
        f in {:else, :none} ⇒ [RE] "Feature name {f} is reserved."
        f not keyword ⇒ [RE] "Feature should be a keyword: {dbg(f)}"
        if f == :default || features.contains(f) {
            v = read-or-FINISHED-or-EOF
            EOF ⇒ EOF error; FINISHED ⇒ even-forms error (§14a)
            result = v
        }
    }
    // discard next form under suppression (runs ALWAYS, even right
    // after a match — this is what drops `:dangling` in
    // `#?(:clj 1 :dangling)` ⇒ 1, probe1)
    d = read-or-FINISHED-or-EOF   // same EOF/FINISHED handling
    FINISHED ⇒ break
}
no match ⇒ return reader sentinel (recur: `#?(:cljs 2)` alone ⇒ EOF error)
splicing: result must be List or Vector (vectors ARE java.util.List:
`[1 #?@(:clj [2])]`⇒`[1 2]`, probe1), else
[RE] "Spliced form list in read-cond-splicing must implement java.util.List";
at top level (no enclosing delimited list) ⇒
[RE] "Reader conditional splicing not allowed at the top level."
(prefer the no-match check first: top-level `#?@(:cljs (1))` ⇒ EOF, probe4).
splice ⇒ prepend items to pending; non-splice ⇒ return result.
```

§14a even-forms message: without line info
`"read-cond requires an even number of forms."` (period); with line
info `"read-cond starting on line {N} requires an even number of
forms"` (no period) — `#?(:clj)` ⇒ line variant (probe1).

`{dbg(f)}` uses java-debug rendering (§22), which for collections is
readable (`[1]`, `{:a 1}`, `#{1}`, `(1)`, `(quote :clj)`, probe1/2/4),
for strings/chars raw (`s`, real newline for `\newline`, probe2),
nil⇒`null`, doubles⇒`Double.toString` (`-Infinity`, `NaN`, probe3),
regex⇒raw source (probe3). **Corpus rule**: avoid multi-entry
collections in this position (JVM prints insertion order, ours Fx —
pinned behaviorally identical but textually different, §24).

`:default` always matches, even non-first (`#?(:cljs 1 :default 2)`⇒2).
`:else`/`:none` are reserved (`#?(:else 1)` errors). `#?()`/`#?(:a)`
with no match ⇒ recur-then-EOF. `[1 #?@(:cljs (2)) 3]`⇒`[1 3]`;
`[#?@(:cljs (1))]`⇒`[]`; `{:a 1 #?@(:clj (:b 2))}`⇒`{:a 1, :b 2}`;
odd splice into map ⇒ even-forms map error at end (probe4).

`:preserve`: reads the raw list (with line/col metas as usual) and
returns `ReaderConditional{form, splicing?}`, printing `#?`/`#?@` +
form (`#?(:a 1)`⇒`#?(:a 1)`, probe1). Tags nested inside a preserved
cond become `TaggedLiteral`s (`#?(:a #inst "2020-01-01")`⇒
`#?(:a #inst "2020-01-01")`, probe2). **Preserve only applies inside a
read-cond**: top-level `#inst` with `:preserve` still invokes the data
reader (`isPreserveReadCond` requires the cond env, probe1).

## §15 Tagged literals / ctor (`#tag form`)

`#` + char with no dispatch entry (any char incl. non-ASCII, e.g.
`#é 1`, probe4; EOF ⇒ `[RE] "EOF while reading character"`): unread
the char and run the ctor path (so `No dispatch macro for:` is
unreachable in S2 — the ctor path always throws or returns):

1. Read the tag (recursive, `eofIsError=true`); non-symbol ⇒
   `[RE] "Reader tag must be a symbol"` (`#123`, probe1).
2. Read the form (same). Order matters: `#unknown (unclosed` ⇒ EOF
   error, not unknown-tag.
3. Preserve-in-cond (§14) ⇒ `TaggedLiteral{tag, form}`.
4. Tag **name** (not ns) contains `.` ⇒ record-deferral (§22)
   (`#foo.bar [1]`, `#foo/bar.baz 1`; but `#foo.bar/baz 1` has no dot
   in the name ⇒ step 5, probe4).
5. Else tag-table lookup: `inst`/`uuid` ⇒ N5 deferrals (§22);
   anything else ⇒ `[RE] "No reader function for tag {tag}"`
   (`#a 1`, `#foo/bar 1`, probe1/4).

## §16 Eval reader (`#=`)

S2 had no evaluator: `#=` ⇒ immediate N5-deferral error (§22)
**without consuming the following form**. S7+ honors `*read-eval*`
(issue #125): closed ⇒ the JVM gate error —
`[RE] "EvalReader not allowed when *read-eval* is false."` at the
entry span (col right after `#=`, probe1 `#=(+ 1 2)`), compared
conformant (§24 item 2); open ⇒ read the operand, then apply the
evaluator hook at the post-operand span (hookless contexts report
the `ReadEval` deferral instead). Application semantics (head
resolution over raw operand forms, never full eval) are
oracle-pinned in the S7 spec, not here.

## §17 Unreadable (`#<`)

`#<` ⇒ `[RE] "Unreadable form"` (col after `#<`, probe1).

## §18 Namespaced maps (`#:`)

`#:` then optional second `:` (auto). Then:

- Whitespace: auto + `{` (after skipping ws) ⇒ current ns
  (`#:: {:a 1}` ⇒ `#:user{:a 1}`); auto + non-`{`, or plain + ws ⇒
  `[RE] "Namespaced map must specify a namespace"` (`#:  {}`, probe1).
- Else unread, read the ns symbol, skip ws, require `{` else
  `[RE] "Namespaced map must specify a map"` (`#:a 1`, probe1).
- Resolve: auto + no sym ⇒ current ns; auto + sym ⇒ sym must be a
  simple symbol else `[RE] "Namespaced map must specify a valid
  namespace: {dbg}"`, then alias lookup else
  `[RE] "Unknown auto-resolved namespace alias: {sym}"` (col right
  after the sym, probe1 `#::nope{:b 1}`); plain ⇒ sym must be a simple
  symbol (`#:1{}`, `#:a/b{}` errors, probe1).
- Read `}`-delimited kvs; odd ⇒
  `[RE] "Namespaced map literal must contain an even number of forms"`.
- Rewrite keys: unqualified Keyword/Symbol ⇒ prefix with ns;
  `_`-ns ⇒ strip to unqualified; otherwise (qualified, non-sym/kw)
  unchanged (`#::{:a 1 :_/b 2 :c/d 3 'e 4}`, `#:a{:b 1 :_/c 2}`,
  `#:a{1 2 "s" 3}`, probe1/4). Build with the **map-literal
  construction incl. dup check**: `#:a{:b 1 :b 2}` ⇒
  `[IAE] "Duplicate key: :a/b"` (probe2).
- `#:a{}` ⇒ `{}`; `#:`/`#::` at EOF ⇒ `EOF while reading`.

## §19 Collections, delimiters, duplicates

- `(`: capture `(line, col_after_( - 1)`; read `)`-delimited; attach
  as `:line`/`:column` meta (unless present) on non-empty lists only
  (`()` has no meta; `(a)` meta `{:line 1, :column 1}`, probe1).
- `[`, `{`, `#{`: vector / map / set; no line/col metas.
- Delimited EOF ⇒ `[RE] "EOF while reading, starting at line {N}"`
  iff line tracking on (§1 always-on in S2; the S5 `read-string` path
  passes line-numbers-off to get bare `EOF while reading`).
- `)`/`]`/`}` at top ⇒ `[RE] "Unmatched delimiter: {c}"` (col after
  the char, probe1).
- Map odd ⇒ `[RE] "Map literal must contain an even number of forms"`
  (checked after the full read — col after `}`, probe1/4 `{:a}`,
  `@{1}`).
- Duplicates ⇒ `[IAE] "Duplicate key: {dbg(key)}"` with java-debug
  rendering (§22; strings raw: `{"a" 1 "a" 2}` ⇒ `Duplicate key: a`,
  probe2). One match function everywhere; only the scan and the
  reported side differ (40+ probes, probe1–16):
  - Small maps (`canBePAM`: ≤16 forms, or ≤128 forms with every key
    at even index ≥16 a Keyword — replicate `RT.map`/`canBePAM`
    exactly): pairwise match over earlier/later pairs, **report the
    FIRST**. Else big-map rule.
  - Big maps + all sets: incremental (each new element vs all
    previous), **report the NEW** (`{0.0 … -0.0}` small ⇒ `0.0`,
    big ⇒ `-0.0`, probe10).
  - `dupMatch(a, b)`: both Keyword ⇒ value-eq (stands in for JVM
    identity — equivalent under interning); both sequential
    (List/Vector in either order) ⇒ same length + elementwise
    `dupMatch`; same variant otherwise ⇒ deep rule below; else
    false. Sequential comparison crosses List/Vector **in both
    orders at every nesting level** (`{[1] :a (1) :b}` DUP,
    probe13; `{(0.0) :a [-0.0] :b}` DUP, probe14). Apparent
    counter-examples are length mismatches (`{[1] :a (quote (1))
    :b}` OK, `{(quote (0.0)) :a [-0.0] :b}` OK — 1 vs 2 elements).
    (Mechanism: `.equiv`, not `.equals`, crosses `Sequential` both
    ways with equiv recursion.)
  - Deep rule: Nil true; Bool Int Str Char Symbol Keyword by value;
    **Double by `Double.compare == 0`** (`0.0`≡`-0.0`, NaN≡NaN);
    Map kv `dupMatch`; Set memberwise `dupMatch`; Regex false
    (identity: `#{#"a" #"a"}` keeps both, probe4); TaggedLiteral /
    ReaderConditional by `(tag, form)` with the **form compared via
    `plainEquals`** (`{#inst [0.0] 1 #inst [-0.0] 2}` preserved ⇒ OK,
    probe9; same in sets, probe15).
  - `plainEquals` (JVM `.equals`, used only for TL/RC forms and
    scalar fallbacks): scalars by value with **Double by bits**;
    Vector strict (other must be Vector); List loose (List/Vector);
    Map same-size + kv `plainEquals`; Set same-size + memberwise
    `plainEquals` (linear scan — outcome-equivalent to
    hash-contains since `.equals`⇒same `hashCode` holds); Regex
    false; TaggedLiteral/ReaderConditional by tag + `plainEquals`
    form.
  - Consequences (all probed): `{1 :a 1.0 :b}` OK (both); `#{1 1.0}`
    OK; `#{0.0 -0.0}` DUP; `#{##NaN ##NaN}` DUP (`Duplicate key:
    NaN`); seq-cross DUPs both orders in small maps, sets and big
    maps; nested-double collections DUP (`{[0.0] :a [-0.0] :b}`,
    `{{:x 0.0} :a {:x -0.0} :b}`, `#{[0.0] [-0.0]}`,
    `#{{:x 0.0} {:x -0.0}}`, `{(0.0) :a [-0.0] :b}`,
    `#{[(0.0)] ((-0.0))}`); `{[1] :a [1.0] :b}` OK; `#{[1] [1.0]}`
    OK; `{(1) :a [1.0] :b}` OK; non-adjacent dups detected;
    scale-independent (verified to 20+ elements).
  - Message divergence: `TaggedLiteral`/`ReaderConditional` keys
    render on the JVM as `clojure.lang.TaggedLiteral@7fc19ced`
    (identity hash — nondeterministic, probe7/8); ours renders
    readably (`#inst "…"`) — expected-divergent, §24.

## §20 Symbolic values (`##`)

`##`: read one form recursively (`##` at EOF ⇒ `EOF while reading`;
`## Inf` with a space works, probe1). Non-symbol ⇒
`[RE] "Invalid token: ##{dbg}"` (`##123` ⇒ `##123`, `##"x"` ⇒ `##x`
raw, `##nil` ⇒ `##null`, `##(a)` ⇒ `##(a)` readable, probe1).
Symbol not in {`Inf`, `-Inf`, `NaN`} ⇒
`[RE] "Unknown symbolic value: ##{o}"` (`##foo`).

## §21 `clj-value` changes (S2 owns these)

- **Meta storage**: add `meta: Option<Arc<Map>>` to `Symbol` (keep
  `new()` signature; add `with_meta`/`meta` accessors) and to
  List/Vector/Map/Set via newtype wrappers exposing the **same method
  names** as today (`push_front/back`, `insert`, `get`, `size`,
  `len`, `iter`, `contains`, `new_sync`, …) so S1 tests and the
  printer keep compiling untouched. `Keyword` gets **no** meta
  (JVM rejects `^:k :kw`, probe4).
- **Equality/hash ignore meta** (JVM `Symbol.equals`/`hasheq` ignore
  it): replace derived `PartialEq/Eq/Hash/Ord` on `Symbol`/`Keyword`
  with manual impls over `(ns, name)` only. Collection `=`/`Hash`
  likewise ignore meta (already the case — keep it).
- **`Symbol::intern(&str)`**: first-`/` split per §4 (no `/`
  rejection); `/` alone ⇒ no-ns.
- **New variants**: `Value::TaggedLiteral { tag: Symbol, form: Box<Value> }`
  (or `Arc`), `Value::ReaderConditional { form: Box<Value>, splicing: bool }`.
  Printer: `#` + tag + ` ` + form; `#?`/`#?@` + form (§14/§15 shapes,
  `core_print.clj`). `=`/hash: value-based (tag/form, splicing/form).
- **`PrintOpts.meta: bool`** (default false; additive field — existing
  literals use `..pr()`/`..print()` and keep compiling). Print `^meta `
  prefix when `meta && readably` (JVM requires `*print-meta*` **and**
  `*print-readably*`, `core_print.clj` `print-meta`). Short rule: meta
  map of count 1 with **truthy** `:tag` ⇒ `^tag`, else full map
  (`^String [x]` vs `^{:k true} [1]`, `^{:tag nil …}` full, probe4/5).
  No `:type` dissoc (probe5 `^:type ^String foo` shows `:type`).
- **`namespace_maps` default flips to `true`** (JVM
  `*print-namespace-maps*` defaults true: `#:a{:b 1}` prints lifted,
  probe1). S1's `lift_ns` rules are already correct (keywords and
  symbols, mixed types, same-ns-only — probe3 NSMAP-PRINT3); update
  the S1 test that asserts unlifted-by-default.
- **Publish for the reader/errors**: `pub fn java_double_string(d)`
  (finite ⇒ same as `print_double`; infinite ⇒ `Infinity`/`-Infinity`;
  NaN ⇒ `NaN`) and `pub fn java_debug(v)` (§22). S1's `print_regex`
  (incl. `\Q..\E` handling) is probe-verified exact (probe4 QCHARCODES,
  probe5 QBRANCH) — do not touch it.
- Reader construction helpers as needed (`List`/`Vector` from
  `Vec<Value>` in order, `Map` from kvs with the §19 check, etc.).

## §22 Error interpolation and taxonomy

`java_debug(v)`: Nil⇒`null`; Bool⇒`true`/`false`; Int⇒decimal;
Double⇒`java_double_string` (**not** `##`-forms: `Infinity`,
`-Infinity`, `NaN`, probe3); Str⇒raw contents; Char⇒raw char;
Symbol/Keyword⇒printed form; List/Vector/Map/Set⇒readable pr-str
(`[1]`, `{:a 1}`, `#{1}`, `(1)`, probe1/2); Regex⇒raw source;
TaggedLiteral/ReaderConditional⇒readable pr (§19 divergence note).

`ReaderError` (each with span + `message()` + `jvm_class()`):

| Variant | Message | Class |
|---|---|---|
| Eof | `EOF while reading` | RE |
| EofDelimited { line } | `EOF while reading, starting at line {line}` | RE |
| EofChar | `EOF while reading character` | RE |
| EofString | `EOF while reading string` | RE |
| EofRegex | `EOF while reading regex` | RE |
| InvalidToken { token } | `Invalid token: {token}` | RE |
| InvalidNumber { token } | `Invalid number: {token}` | NFE |
| RadixRange | `Radix out of range` | NFE |
| RadixDigit { digits, radix } | `For input string: "{digits}" under radix {radix}` | NFE |
| InvalidChar { token } | `Unsupported character: \{token}` | RE |
| InvalidCharConst { hex } | `Invalid character constant: \u{hex-lower}` | RE |
| BadUnicodeChar { token } | `Invalid unicode character: \{token}` | IAE |
| BadDigit { c } | `Invalid digit: {c}` | IAE |
| BadOctalLen { len } | `Invalid octal escape sequence length: {len}` | RE |
| OctalRange | `Octal escape sequence must be in range [0, 377].` | RE |
| BadEscape { c } | `Unsupported escape character: \{c}` | RE |
| BadUnicodeEscape { c } | `Invalid unicode escape: \u{c}` (−1 renders U+FFFF) | RE |
| BadUnicodeLen { got, want } | `Invalid character length: {got}, should be: {want}` | IAE |
| UnmatchedDelim { c } | `Unmatched delimiter: {c}` | RE |
| MapOdd | `Map literal must contain an even number of forms` | RE |
| DupKey { key: String } | `Duplicate key: {java-debug}` (pre-rendered) | IAE |
| BadMeta | `Metadata must be Symbol,Keyword,String,Vector or Map` | IAE |
| MetaTarget | `Metadata can only be applied to IMetas` | IAE |
| NestedFn | `Nested #()s are not allowed` | ISE |
| BadArg | `arg literal must be %, %& or %integer` | ISE |
| SpliceTop (sq) | `splice not in list` | ISE |
| UnknownSymbolic { sym } | `Unknown symbolic value: ##{sym}` | RE |
| BadSymbolic { dbg } | `Invalid token: ##{dbg}` | RE |
| TagNotSymbol | `Reader tag must be a symbol` | RE |
| UnknownTag { tag } | `No reader function for tag {tag}` | RE |
| Unreadable | `Unreadable form` | RE |
| CondNotAllowed | `Conditional read not allowed` | RE |
| CondBody | `read-cond body must be a list` | RE |
| ReservedFeature { kw } | `Feature name {kw} is reserved.` | RE |
| BadFeature { dbg } | `Feature should be a keyword: {dbg}` | RE |
| CondEven { line: Option } | line⇒`read-cond starting on line {l} requires an even number of forms`, none⇒`read-cond requires an even number of forms.` | RE |
| SpliceType | `Spliced form list in read-cond-splicing must implement java.util.List` | RE |
| SpliceTopLevel | `Reader conditional splicing not allowed at the top level.` | RE |
| NsMapNs | `Namespaced map must specify a namespace` | RE |
| NsMapMap | `Namespaced map must specify a map` | RE |
| NsMapValidNs { dbg } | `Namespaced map must specify a valid namespace: {dbg}` | RE |
| NsMapAlias { sym } | `Unknown auto-resolved namespace alias: {sym}` | RE |
| NsMapOdd | `Namespaced map literal must contain an even number of forms` | RE |

Deferrals (ours; harness expects divergence — same span discipline,
`jvm_class()` = `ReaderError`, i.e. no JVM counterpart):

- `Ratio { token }` ⇒
  `Ratio literals are not supported in this build (big-number follow-up N3): {token}`
- `BigInt { token }` ⇒
  `BigInt literals are not supported in this build (big-number follow-up N3): {token}`
- `BigDecimal { token }` ⇒
  `BigDecimal literals are not supported in this build (big-number follow-up N3): {token}`
- `ReadEval` ⇒
  `Eval reader #= is not supported in this build (data-reader follow-up N5)`
- `DataReader { tag }` (`inst`/`uuid`) ⇒
  `Reader for tag {tag} is not supported in this build (data-reader follow-up N5)`
- `Record { tag }` ⇒
  `Record construction is not supported in this build (S5): #{tag}`
- `LoneSurrogate { hex4-lower }` ⇒
  `Lone surrogate \u{hex} in string escape is not representable in str`
  (e.g. `\ud800`; JVM accepts — §24).

Unreachable (do not implement as user-facing errors, `debug_assert` /
`unreachable!` with a comment if the shape needs a branch):
`No dispatch macro for:`, `arg literal not in #()`,
`Gensym literal not in syntax-quote`, `Unknown Collection type`.

## §23 Resolver, options, and the differential harness

`ReaderOpts`: `read_cond` (`Allow` default / `Preserve` / `Deny`),
`features` (default `#{:clj}`, always includes `:clj` even if the
caller passes others — mirror `installPlatformFeature`),
`line_numbers` (default true; false gives `read-string` behavior:
bare `EOF while reading`, no list `:line`/`:column`), resolver maps:
`current_ns` (default `user`), `aliases: Map<name, ns>`,
`refers: Map<name, (ns, name)>`, `class_map: Map<name, dotted>`.
(S7's loader will pass `:cljr` in features; the reader stays neutral.)

Harness (JVM oracle side, per case): fresh JVM? No — one JVM, but
**fresh reader per case**; `*ns*` context queried once
(`(ns-name *ns*)`, `(ns-aliases *ns*)`, `(ns-refers *ns*)`,
class-valued `(ns-map *ns*)`) and serialized (EDN/JSON) for our
`ReaderOpts`. Read with `LineNumberingPushbackReader` +
`{:read-cond :allow}` (+ `:eof` sentinel for multi-form), `*read-eval*`
false. Report per form: pr-str (two print configs: default and
`print-meta`+`readably` true), or on error: root message + root class
simple name + ex-data line/col. Our side mirrors with the same
resolver maps.

Comparison: canonicalize **map/set order** on both sides first (walk
the value; sort map entries and set members by pr-str; print
canonically) — JVM insertion order vs our Fx order otherwise diverge
textually on identical values. Then: pr-strings must match exactly
after normalizing `__\d+#`→`__N#`, `__\d+__auto__`→`__N__auto__`
(gensym ids), and error triples (message, class, `(line, col)`) must
match exactly — except §24 cases, which assert divergence-shape
(expected error on one/both sides as catalogued).

## §24 Catalogued divergences (harness asserts these as divergences;
item 2 conformant since #125, kept numbered for stable references)

1. Big tower (§22 N3 ×3): JVM reads values; we error. Both directions pinned.
2. `#=` (N5 — conformant since #125): with `*read-eval*` false both
   sides report `[RE] "EvalReader not allowed when *read-eval* is
   false."` at (line, col after `#=`); the harness compares the
   triple exactly (corpus case `"#=(+ 1 2)"`, gate unit pin).
3. `#inst`/`#uuid` (N5): JVM invokes readers (valid ⇒ value; invalid ⇒
   its error); we always defer after consuming tag+form.
4. Record tags (S5): JVM checks `*read-eval*` first; we defer.
5. Invalid regex: JVM throws `PatternSyntaxException`; we accept (§8).
6. Lone surrogates in strings: JVM accepts; we error (§7).
7. `TaggedLiteral`/`ReaderConditional` in `Duplicate key:` messages:
   JVM `@hex` (nondeterministic); ours readable (§19).
8. Multi-entry collections inside error interpolation
   (`Feature should be a keyword: {...}`, …): same values, different
   key order. **Avoid in corpus** (behavioral parity is pinned by
   single-entry cases).
9. Non-ASCII digit/letter-number chars (`isDigit` Nd vs our
   `is_ascii_digit || is_numeric`): JVM takes the number path;
   exotic Nl/No inputs may differ. Avoid in corpus.
10. `*read-eval* :unknown` toplevel check: no S2 equivalent (no vars).
11. `print-dup`, `EdnReader`/`clojure.edn`, `data_readers.clj`,
    `load-reader` extras: out of S2 scope entirely (S5/S9).

## §25 Corpus requirements

- Every probe input in `docs/t1174/probes/probe*.clj` becomes a corpus
  case (mechanical translation; keep the probe label as the case id).
- Full-file reads of every `src/clj/**/*.clj` (all forms; default +
  print-meta configs). Pre-verified: `#=`/`#inst`/`#uuid`/`1M` hits in
  those files are all inside strings/comments/docstrings, and no
  `N/M` or `a/b` numeric literals occur (grep 2026-10-18); re-run
  that grep in CI if the corpus script globs new files. Also include
  `test/` `.clj`/`.cljc` files that read cleanly, plus negative cases
  for every `ReaderError` variant (§22 table — at least one case
  each, asserting message + class + span).
- Corpus files live next to the harness with expected outcomes
  generated by the oracle (checked in), so `cargo test` needs no JVM;
  a separate `xtask`/script regenerates expectations with the oracle.

## §26 `clj-reader` API (shape)

```rust
pub struct ReaderOpts { /* §23 */ }
pub struct Reader<'a> { /* §1 tracker + pending queue + opts + envs */ }
pub enum ReaderError { /* §22, each with Span { line, col } */ }
pub fn read_str(s: &str, opts: &ReaderOpts) -> Result<Value, ReaderError>;
pub fn read_all(s: &str, opts: &ReaderOpts) -> Result<Vec<Value>, ReaderError>;
```

`read_all` loops `read_str`-on-a-shared-`Reader` until the EOF
sentinel (fresh `(line 1, col 1)` per call; gensym ids from the
process-global counter). Errors abort the read (no recovery).

## §27 Guesses the corpus must confirm (implement as stated; on JVM
mismatch, report back instead of guessing further)

- Integer-vs-Long same-value dup pairs (unreachable until S3's
  `Integer` variant; S3 re-probes: likely dup via `Numbers.equal`).
- `print-dup` parity is not attempted (only `pr`/`print` configs).

Resolved during S2 review (probe12–15, now normative above):
radix-10 digit errors omit `under radix`; greedy radix digits swallow
`N` (`36r1N`⇒59); seq-cross dups both orders with patched recursion
everywhere (no small-map asymmetry, no plain elementwise); TL/RC
forms compare plain in maps, sets and big maps alike.


