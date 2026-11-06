// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! `clojure-rs`: native Clojure runtime, REPL, LLVM AOT compiler, and JIT.
//!
//! Slices S1+S7A: CLI shell plus `eval`/`run` wiring over the booted
//! `clj-eval` core. `repl` landed in S7B/C, `build` in S8 (spec
//! `docs/spec/t1174/`), `jit` in gap-121 slice 1 (spec
//! `docs/spec/gap-121/slice-1.md`).

/// mimalloc (#226) off Linux: the interpreter allocates many small,
/// short-lived blocks (reader lists, analyzed forms, seq steps), where
/// it is markedly cheaper than the platform allocators. Linux keeps
/// glibc's allocator for peak RSS: under the default `madvise`
/// transparent-huge-page mode mimalloc's arenas take huge pages (a
/// trivial run peaked at 19 MB instead of 9), and even without them a
/// trivial run peaks 0.7 MB higher and data-heavy runs 20-30% higher.
#[cfg(not(target_os = "linux"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::{Parser, Subcommand};
use clj_eval::{Evaluator, SourceError, boot, compile_diagnostic, compile_head};
use clj_value::Value;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "clojure-rs",
    version,
    about = "Native Clojure runtime, REPL, and compiler"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Evaluate a single form and print the result.
    Eval {
        /// The form to evaluate.
        form: String,
        /// Restrict the reader to EDN.
        #[arg(long)]
        edn: bool,
    },
    /// Load and run a script file.
    Run {
        /// The script file to run.
        file: PathBuf,
        /// Bound to `*command-line-args*`.
        args: Vec<String>,
    },
    /// Start the interactive REPL.
    Repl {
        /// History file (default `~/.clojure-rs/history`).
        #[arg(long)]
        history_file: Option<PathBuf>,
        /// Disable history read/write.
        #[arg(long)]
        no_history: bool,
    },
    /// AOT-compile the pure subset to a native binary.
    Build {
        /// The source file to compile.
        file: PathBuf,
        /// The native binary to write.
        #[arg(short, long)]
        output: PathBuf,
        /// Also write the LLVM IR to this file (skips linking).
        #[arg(long)]
        emit_ir: Option<PathBuf>,
        /// Also write the native object file to this path (skips linking).
        #[arg(long)]
        emit_obj: Option<PathBuf>,
        /// Link against this runtime archive instead of the autodetected one.
        #[arg(long)]
        runtime_archive: Option<PathBuf>,
    },
    /// JIT-run the pure subset in-process (no binary emitted).
    Jit {
        /// The source file to run.
        file: PathBuf,
        /// Passed to the JITted `-main` as its argv list.
        args: Vec<String>,
    },
}

/// Exit codes per spec C8: 0 ok, 1 evaluation/compile error,
/// 2 usage error (clap's default), 3 I/O error.
fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Eval { form, edn } => cmd_eval(&form, edn),
        Command::Run { file, args } => cmd_run(&file, &args),
        Command::Repl {
            history_file,
            no_history,
        } => cmd_repl(history_file, no_history),
        Command::Build {
            file,
            output,
            emit_ir,
            emit_obj,
            runtime_archive,
        } => cmd_build(
            &file,
            &output,
            emit_ir.as_deref(),
            emit_obj.as_deref(),
            runtime_archive.as_deref(),
        ),
        Command::Jit { file, args } => cmd_jit(&file, &args),
    }
}

/// Install the linked AOT backend as `(compile ...)`'s driver
/// (S18). Unlinked builds keep the hook unset, so every compile
/// reports the exact unlinked line (T11).
fn install_compile_driver() {
    #[cfg(feature = "aot")]
    clj_aot::driver::install_driver();
}

/// The `<eval>` file token (pins P5).
const EVAL_FILE: &str = "<eval>";

/// `eval` reads every form and evaluates + prints EACH value on
/// its own line — exactly like JVM `clojure.main -e` (pins P5).
/// Reader spans blame the input directly (no textual wrap, so no
/// shift to undo); `*1` stays untouched (the JVM binds none here).
fn cmd_eval(form: &str, edn: bool) -> ExitCode {
    install_compile_driver();
    let scan = EvalScanner::new(form).run();
    // (Match guard, not a let-chain: MSRV 1.85.)
    match &scan.edn_violation {
        Some(v) if edn => {
            eprintln!(
                "{EVAL_FILE}:{}:{}: non-EDN reader syntax: '{}'",
                v.line, v.col, v.lexeme
            );
            return ExitCode::from(1);
        }
        _ => {}
    }
    if !scan.has_code {
        // Blank input (whitespace/commas/comments only): silent
        // exit 0, like JVM `-e ''` (pins P5).
        return ExitCode::SUCCESS;
    }
    let mut eval = boot();
    let code = cmd_eval_forms(&mut eval, form);
    // JVM parity: non-daemon pool/future threads keep the JVM alive
    // past main; join them here (T1192 FX1).
    eval.join_drainers();
    code
}

/// Evaluate and print each form (split from `cmd_eval` so the exit
/// join runs on every path, success and failure alike).
fn cmd_eval_forms(eval: &mut Evaluator, form: &str) -> ExitCode {
    // Live hooks (S7): `#=` evaluates during the read, like
    // `eval_source`; each form reads after the previous evaluates
    // (clojure.main's `-e` loop), so `(ns foo) ::kw` is `:foo/kw`.
    let mut forms = eval.live_forms(form);
    loop {
        let form = match eval.next_live_form(&mut forms) {
            Ok(Some(form)) => form,
            Ok(None) => break,
            Err(e) => {
                eprintln!("{}", eval_diagnostic(&e));
                return ExitCode::from(1);
            }
        };
        let value = match eval.eval_top_form(&form) {
            Ok(value) => value,
            Err(e) => {
                eprintln!("{}", eval_diagnostic(&e));
                return ExitCode::from(1);
            }
        };
        if matches!(value, Value::Nil) {
            // JVM `-e` suppresses nil results (pins P5); the REPL
            // still prints them (pins P1).
            continue;
        }
        match print_value(eval, &value) {
            Ok(text) => println!("{text}"),
            Err(diagnostic) => {
                eprintln!("{EVAL_FILE}:1:1: {diagnostic}");
                return ExitCode::from(1);
            }
        }
    }
    ExitCode::SUCCESS
}

/// Render one eval value exactly like `prn`: bound `*print-*`
/// opts, deep-realized (lazy seqs print walked, like the JVM).
/// Failures come back as their one-line diagnostic.
fn print_value(eval: &mut Evaluator, value: &Value) -> Result<String, String> {
    let opts = match clj_eval::core::print_opts(eval, None) {
        Ok(opts) => opts,
        Err(err) => return Err(class_diagnostic(err.class, &err.message)),
    };
    let realized = match eval.realize_deep(value) {
        Ok(realized) => realized,
        Err(signal) => return Err(print_failure(signal)),
    };
    Ok(clj_value::pr_to_string(&realized, &opts))
}

/// A print-time forcing failure as a one-line diagnostic (a `recur`
/// cannot escape forcing — every loop/fn frame catches its own — so
/// the `Recur` arm is defense in depth).
fn print_failure(signal: clj_eval::Signal) -> String {
    match signal {
        clj_eval::Signal::Throw(t) => {
            let message = t.message.as_deref().unwrap_or("");
            class_diagnostic(t.class, message)
        }
        clj_eval::Signal::Recur(_) => "internal error: recur escaped its loop or fn".to_owned(),
    }
}

/// The one-line `SimpleClass: message` diagnostic (bare class when
/// the message is empty).
fn class_diagnostic(class: clj_value::ThrowClass, message: &str) -> String {
    let simple = class.jvm_name().rsplit('.').next().unwrap_or("Throwable");
    if message.is_empty() {
        simple.to_owned()
    } else {
        format!("{simple}: {message}")
    }
}

/// A `SourceError` as `file:line:col: message` (pins P5: col always
/// present). Spans blame the input directly. `Throwable` carries no
/// span, so uncaught runtime errors blame the input start (1:1).
fn eval_diagnostic(e: &SourceError) -> String {
    match e {
        SourceError::Reader(e) => {
            let span = e.span();
            let line = span.line;
            let col = span.col;
            let msg = e.message();
            format!("{EVAL_FILE}:{line}:{col}: {msg}")
        }
        SourceError::Compile(e) => compile_diagnostic(EVAL_FILE, e),
        SourceError::Uncaught(t) => {
            let class = t.class.jvm_name();
            let msg = t.message.as_deref().unwrap_or("");
            format!("{EVAL_FILE}:1:1: {class}: {msg}")
        }
        SourceError::Internal(msg) => format!("{EVAL_FILE}:1:1: internal error: {msg}"),
    }
}

/// The wrapper line minus its ` at (site).` tail, e.g. `Syntax error
fn cmd_run(file: &Path, args: &[String]) -> ExitCode {
    install_compile_driver();
    if !file.exists() {
        eprintln!("run: no such file: {}", file.display());
        return ExitCode::from(3);
    }
    if file.is_dir() {
        eprintln!("run: is a directory: {}", file.display());
        return ExitCode::from(3);
    }
    // Read-time IO failures (permissions, races) are C8 exit 3 —
    // before `load-file` can reframe them as eval errors.
    if let Err(e) = std::fs::File::open(file) {
        eprintln!("run: cannot open {}: {e}", file.display());
        return ExitCode::from(3);
    }
    let path = file.to_string_lossy();
    let mut eval = boot();
    // Upstream `clojure.main/with-bindings` ("vars that often need
    // to be set!"): every script runs inside thread bindings for
    // the set below (issue #201 S3 — upstream `string.clj:46`
    // `set!`s `*warn-on-reflection*` at top level, which needs
    // this frame, not a loader binding). `*compile-path*` keeps
    // its root (no `System/getProperty` yet); spec's
    // `*explain-out*` stays out (spec isn't loaded here).
    //
    // `clojure.string` is required up front, inside the frame:
    // JVM `clojure.main <script>` loads `clojure.main`, which
    // requires `clojure.spec.alpha`, which requires
    // `clojure.string` — so fully-qualified `clojure.string/...`
    // uses resolve on the JVM even when the script never
    // requires it (27 recipe scripts rely on this). `require` is
    // idempotent, so scripts that require it themselves are
    // unaffected.
    let inits = args_expr(args);
    let target = clj_string(&path);
    let script = format!(
        "(binding [*ns* *ns* \
         *warn-on-reflection* *warn-on-reflection* \
         *math-context* *math-context* \
         *print-meta* *print-meta* \
         *print-length* *print-length* \
         *print-level* *print-level* \
         *print-namespace-maps* true \
         *data-readers* *data-readers* \
         *default-data-reader-fn* *default-data-reader-fn* \
         *compile-path* *compile-path* \
         *command-line-args* {inits} \
         *unchecked-math* *unchecked-math* \
         *assert* *assert* \
         *1 nil *2 nil *3 nil *e nil] \
         (require 'clojure.string) \
         (load-file {target}))"
    );
    let code = match eval.eval_source(&script) {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{}", run_diagnostic(&path, &e));
            ExitCode::from(1)
        }
    };
    // Exit join (see `cmd_eval`): drained agents and futures print
    // before the process goes away, like the JVM's non-daemon wait.
    eval.join_drainers();
    // The process ends here: tearing the runtime down var by var
    // (every namespace, fn and value still reachable) is pure exit
    // cost (#226), and the JVM runs no finalizers at exit either.
    // Nothing buffers behind it: stdout is std's (flushed at exit)
    // and file writers write through.
    std::mem::forget(eval);
    code
}

/// The `*command-line-args*` init (pins P4): nil without args, else a
/// quoted list of strings (a list value on read).
fn args_expr(args: &[String]) -> String {
    if args.is_empty() {
        return "nil".to_owned();
    }
    let mut out = "(quote (".to_owned();
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(&clj_string(arg));
    }
    out.push_str("))");
    out
}

/// A Rust string as a Clojure string literal.
fn clj_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The load wrapper's class: only `load-file` compile/reader failures
/// carry a parseable ` at (site).` tail.
const COMPILER_EXCEPTION: &str = "clojure.lang.Compiler$CompilerException";

/// A `run` failure as `file:line:col: message` naming the path as
/// passed. `load-file` wraps file errors as `... at (site).` with the
/// real message on the cause chain, so the site is recovered from the
/// wrapper text; anything else (notably span-less runtime throws)
/// blames 1:1.
fn run_diagnostic(path: &str, e: &SourceError) -> String {
    match e {
        SourceError::Reader(e) => {
            let span = e.span();
            let msg = e.message();
            format!("{path}:{}:{}: {msg}", span.line, span.col)
        }
        SourceError::Compile(e) => {
            // `compile_err` converts every bare failure to `Uncaught`
            // before `eval_source` returns, so `e.bare` is always
            // false here (the single caller passes `eval_source`
            // errors only).
            let wrapper = e.wrapper();
            let head = compile_head(&wrapper);
            let msg = &e.message;
            format!("{path}:1:1: {head}: {msg}")
        }
        SourceError::Uncaught(t) => {
            let outer = t.message.as_deref().unwrap_or("");
            let compiler = t.class.jvm_name() == COMPILER_EXCEPTION;
            match load_site(outer) {
                Some((line, col)) if compiler => {
                    let mut msg = outer;
                    let mut cur = t;
                    while let Some(next) = cur.cause.as_deref() {
                        if let Some(m) = next.message.as_deref() {
                            msg = m;
                        }
                        cur = next;
                    }
                    let head = compile_head(outer);
                    format!("{path}:{line}:{col}: {head}: {msg}")
                }
                _ => {
                    let class = t.class.jvm_name();
                    format!("{path}:1:1: {class}: {outer}")
                }
            }
        }
        SourceError::Internal(msg) => format!("{path}:1:1: internal error: {msg}"),
    }
}

/// Recover `(line, col)` from a load wrapper's trailing ` at (site).`
/// (`site` is `file:line:col`, so the last two `:` segments parse;
/// the file part itself may hold colons).
fn load_site(message: &str) -> Option<(u32, u32)> {
    let rest = message.strip_suffix(").")?;
    let i = rest.rfind(" at (")?;
    let site = &rest[i + 5..];
    let mut parts = site.rsplit(':');
    let col: u32 = parts.next()?.parse().ok()?;
    let line: u32 = parts.next()?.parse().ok()?;
    parts.next()?;
    Some((line, col))
}

/// The eval pre-scan: whether the input holds code outside
/// comments/whitespace, and the first non-EDN reader lexeme (if
/// any). Balance is the reader's job (it reports real spans).
struct EvalScan {
    has_code: bool,
    edn_violation: Option<EdnViolation>,
}

/// A non-EDN reader lexeme and its 1-based position.
struct EdnViolation {
    line: u32,
    col: u32,
    lexeme: String,
}

/// One pass over eval input, tracking line/col while skipping string
/// bodies, `;`/`#!` comments, and `\` char literals. Commas and
/// `char::is_whitespace` are blank; entering a string counts as code,
/// so an unterminated string still reaches the reader's diagnostic.
struct EvalScanner<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
    line: u32,
    col: u32,
    scan: EvalScan,
}

impl<'a> EvalScanner<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            chars: src.chars().peekable(),
            line: 1,
            col: 1,
            scan: EvalScan {
                has_code: false,
                edn_violation: None,
            },
        }
    }

    fn run(mut self) -> EvalScan {
        while let Some(c) = self.next() {
            let (at_line, at_col) = (self.line, self.col);
            self.advance(c);
            match c {
                '\n' => {}
                ';' => self.skip_comment(),
                '"' => {
                    self.scan.has_code = true;
                    self.skip_string();
                }
                '\\' => {
                    self.scan.has_code = true;
                    self.skip_char_tail();
                }
                '#' => self.hash_dispatch(at_line, at_col),
                '\'' | '`' | '~' | '@' | '^' => {
                    self.scan.has_code = true;
                    self.flag_edn(at_line, at_col, c.to_string());
                }
                ':' => {
                    self.scan.has_code = true;
                    self.colon_dispatch(at_line, at_col);
                }
                _ if c.is_whitespace() || c == ',' => {}
                _ => {
                    self.scan.has_code = true;
                }
            }
        }
        self.scan
    }

    /// One `#` dispatch: flag the `#x` lexeme as non-EDN. `#_` is
    /// valid EDN (no flag); `#!` is a line comment (skipped, holds
    /// no code by itself); a regex body skins string rules (escapes
    /// honored) so its contents flag nothing.
    fn hash_dispatch(&mut self, at_line: u32, at_col: u32) {
        match self.chars.peek().copied() {
            Some('!') => {
                self.next();
                self.advance('!');
                self.skip_comment();
            }
            Some(d) if matches!(d, '(' | '"' | '\'' | '?' | '=' | ':' | '#') => {
                self.scan.has_code = true;
                self.flag_edn(at_line, at_col, format!("#{d}"));
                self.next();
                self.advance(d);
                if d == '"' {
                    self.skip_string();
                }
            }
            _ => {
                self.scan.has_code = true;
            }
        }
    }

    /// A `::` opener is non-EDN (auto-resolve needs ns context);
    /// single-`:` keywords stay valid.
    fn colon_dispatch(&mut self, at_line: u32, at_col: u32) {
        if self.chars.peek() == Some(&':') {
            self.flag_edn(at_line, at_col, "::".to_owned());
            self.next();
            self.advance(':');
        }
    }

    /// Next normalized char (`\r\n`/`\r` arrive as `\n`, like the
    /// reader's stream).
    fn next(&mut self) -> Option<char> {
        let c = self.chars.next()?;
        if c == '\r' {
            if self.chars.peek() == Some(&'\n') {
                self.chars.next();
            }
            return Some('\n');
        }
        Some(c)
    }

    /// Consume one already-read char's worth of position.
    fn advance(&mut self, c: char) {
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
    }

    /// Record the first non-EDN lexeme only.
    fn flag_edn(&mut self, line: u32, col: u32, lexeme: String) {
        if self.scan.edn_violation.is_none() {
            self.scan.edn_violation = Some(EdnViolation { line, col, lexeme });
        }
    }

    /// Skip a `;`/`#!` comment body (the newline re-syncs position).
    fn skip_comment(&mut self) {
        while let Some(d) = self.next() {
            self.advance(d);
            if d == '\n' {
                break;
            }
        }
    }

    /// Skip a string body after the opening quote (escapes honored).
    fn skip_string(&mut self) {
        while let Some(d) = self.next() {
            self.advance(d);
            if d == '\\' {
                if let Some(e) = self.next() {
                    self.advance(e);
                }
                continue;
            }
            if d == '"' {
                return;
            }
        }
    }

    /// Skip a char literal's tail after the backslash (one char plus
    /// the alnum run for `\newline`/`\u0041`-style spellings).
    fn skip_char_tail(&mut self) {
        let Some(first) = self.next() else {
            return;
        };
        self.advance(first);
        if !first.is_ascii_alphabetic() {
            return;
        }
        while let Some(e) = self.chars.peek().copied() {
            if !e.is_ascii_alphanumeric() {
                break;
            }
            self.next();
            self.advance(e);
        }
    }
}

/// `build`: AOT-compile the pure subset to a native binary (S8,
/// spec §2.4). The IO gate runs on every configuration (missing
/// input is exit 3 even unlinked); only compilation itself splits
/// on the `aot` feature.
fn cmd_build(
    file: &Path,
    output: &Path,
    emit_ir: Option<&Path>,
    emit_obj: Option<&Path>,
    runtime_archive: Option<&Path>,
) -> ExitCode {
    if !file.exists() {
        eprintln!("build: no such file: {}", file.display());
        return ExitCode::from(3);
    }
    if file.is_dir() {
        eprintln!("build: is a directory: {}", file.display());
        return ExitCode::from(3);
    }
    let source = match std::fs::read_to_string(file) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("build: cannot open {}: {e}", file.display());
            return ExitCode::from(3);
        }
    };
    cmd_build_compile(file, &source, output, emit_ir, emit_obj, runtime_archive)
}

/// Linked `build`: plan → LLVM → link. Plan failures print the
/// diagnostic verbatim (exit 1); LLVM failures print `build: ...`
/// (exit 1); IO/link failures print `build: ...` (exit 3) — C8.
#[cfg(feature = "aot")]
fn cmd_build_compile(
    file: &Path,
    source: &str,
    output: &Path,
    emit_ir: Option<&Path>,
    emit_obj: Option<&Path>,
    runtime_archive: Option<&Path>,
) -> ExitCode {
    use clj_aot::{CompileError, EmitOptions, LinkOptions};
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf))
        .unwrap_or_default();
    let path = file.to_string_lossy();
    let emit = EmitOptions {
        emit_ir: emit_ir.map(std::path::Path::to_path_buf),
        emit_obj: emit_obj.map(std::path::Path::to_path_buf),
    };
    let link = LinkOptions {
        runtime_archive: runtime_archive.map(std::path::Path::to_path_buf),
        exe_dir,
        linker: None,
    };
    match clj_aot::compile_str(source, &path, output, &emit, &link) {
        Ok(()) => ExitCode::SUCCESS,
        Err(CompileError::Plan(msg)) => {
            eprintln!("{msg}");
            ExitCode::from(1)
        }
        Err(CompileError::Target(msg) | CompileError::Emit(msg)) => {
            eprintln!("build: {msg}");
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("build: {e}");
            ExitCode::from(3)
        }
    }
}

/// Unlinked `build`: the IO gate already ran, so any input reaching
/// here is readable — report the missing backend (exit 1).
#[cfg(not(feature = "aot"))]
fn cmd_build_compile(
    _file: &Path,
    _source: &str,
    _output: &Path,
    _emit_ir: Option<&Path>,
    _emit_obj: Option<&Path>,
    _runtime_archive: Option<&Path>,
) -> ExitCode {
    eprintln!("{}", clj_eval::compiler::UNLINKED_BUILD_LINE);
    ExitCode::from(1)
}

/// `jit`: JIT-run the pure subset in-process (gap-121 slice 1).
/// The IO gate runs on every configuration (missing input is
/// exit 3 even unlinked); only running itself splits on the
/// `aot` feature.
fn cmd_jit(file: &Path, args: &[String]) -> ExitCode {
    if !file.exists() {
        eprintln!("jit: no such file: {}", file.display());
        return ExitCode::from(3);
    }
    if file.is_dir() {
        eprintln!("jit: is a directory: {}", file.display());
        return ExitCode::from(3);
    }
    let source = match std::fs::read_to_string(file) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("jit: cannot open {}: {e}", file.display());
            return ExitCode::from(3);
        }
    };
    cmd_jit_compile(file, &source, args)
}

/// Linked `jit`: plan → LLVM → run in-process. Both failures
/// print through one arm (exit 1) — the diagnostic verbatim,
/// backend failures `jit: `-prefixed (the prefix is baked into
/// the error at construction, since the LLVM arm is
/// environmentally untriggerable and could never cover its own
/// lines — spec §6).
#[cfg(feature = "aot")]
fn cmd_jit_compile(file: &Path, source: &str, args: &[String]) -> ExitCode {
    let path = file.to_string_lossy();
    match clj_aot::jit::run_str(source, &path, args) {
        Ok(code) => ExitCode::from(code as u8),
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}

/// Unlinked `jit`: the IO gate already ran, so any input reaching
/// here is readable — report the missing backend (exit 1).
#[cfg(not(feature = "aot"))]
fn cmd_jit_compile(_file: &Path, _source: &str, _args: &[String]) -> ExitCode {
    eprintln!("{}", clj_eval::compiler::UNLINKED_JIT_LINE);
    ExitCode::from(1)
}

/// History persistence from the CLI flags (`--no-history` wins over
/// `--history-file`).
fn history_options(no_history: bool, history_file: Option<PathBuf>) -> clj_repl::HistoryOptions {
    match (no_history, history_file) {
        (true, _) => clj_repl::HistoryOptions::disabled(),
        (false, Some(path)) => clj_repl::HistoryOptions::with_path(path),
        (false, None) => clj_repl::HistoryOptions::default(),
    }
}

/// A prompt closure cloning the shared prompt text on each read (so
/// `in-ns` re-prompts at once).
fn prompt_fn(text: std::rc::Rc<std::cell::RefCell<String>>) -> impl Fn() -> String {
    move || text.borrow().clone()
}

/// `repl`: tty stdin gets the banner + rustyline driver (history,
/// completion, multiline); anything else degrades to streaming eval
/// with no banner and no prompts (pins P6 purity).
fn cmd_repl(history_file: Option<PathBuf>, no_history: bool) -> ExitCode {
    use std::io::IsTerminal;
    cmd_repl_with(
        history_file,
        no_history,
        std::io::stdin().is_terminal(),
        make_rustyline_source,
    )
}

/// The production line backend (a named fn, not a closure, so the
/// tty-driving test shares — and covers — this exact line).
fn make_rustyline_source(
    names: Vec<String>,
    is_complete: fn(&str) -> bool,
) -> Result<clj_repl::RustyLineSource, String> {
    clj_repl::RustyLineSource::new(names, is_complete)
}

/// [`cmd_repl`] with its environment explicit (the tty probe and the
/// line-backend constructor), so tests drive every path headless.
fn cmd_repl_with(
    history_file: Option<PathBuf>,
    no_history: bool,
    tty: bool,
    make_source: impl FnOnce(Vec<String>, fn(&str) -> bool) -> Result<clj_repl::RustyLineSource, String>,
) -> ExitCode {
    install_compile_driver();
    use std::io::BufReader;
    if !tty {
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        let mut session =
            clj_repl::Session::new(boot(), BufReader::new(stdin.lock()), stdout.lock());
        let code = session.run();
        session.evaluator_mut().join_drainers();
        return ExitCode::from(code as u8);
    }
    println!("clojure-rs {}", env!("CARGO_PKG_VERSION"));
    let stdout = std::io::stdout();
    let mut session = clj_repl::Session::new(boot(), std::io::empty(), stdout.lock());
    session.set_prompt(Some("{ns}=> ".to_owned()));
    let names = clj_repl::core_names(session.evaluator_mut());
    let source = match make_source(names, clj_repl::read_is_complete) {
        Ok(source) => source,
        Err(msg) => {
            // Terminal backend unavailable: degrade to streaming
            // rather than dying (D3 — a REPL that cannot edit
            // still evaluates).
            eprintln!("clojure-rs: warning: line editing unavailable ({msg}); streaming input");
            let stdin = std::io::stdin();
            let (eval, _, _) = session.into_parts();
            let mut fallback = clj_repl::Session::new(eval, BufReader::new(stdin.lock()), stdout);
            let code = fallback.run();
            fallback.evaluator_mut().join_drainers();
            return ExitCode::from(code as u8);
        }
    };
    let history = history_options(no_history, history_file);
    cmd_repl_tty(session, source, history)
}

/// The tty editor loop over an explicit line source (generic so tests
/// drive it with a stub; production passes the rustyline backend).
fn cmd_repl_tty<W: std::io::Write, S: clj_repl::LineSource>(
    session: clj_repl::Session<std::io::Empty, W>,
    source: S,
    history: clj_repl::HistoryOptions,
) -> ExitCode {
    let mut editor = clj_repl::TtyEditor::with_history(source, history);
    let prompt_text = std::rc::Rc::new(std::cell::RefCell::new(
        session.render_prompt().unwrap_or_default(),
    ));
    let prompt = prompt_fn(std::rc::Rc::clone(&prompt_text));
    let mut handler = SessionHandler::new(session, prompt_text);
    let code = editor.run(&prompt, &mut handler);
    handler.session.evaluator_mut().join_drainers();
    let status = handler.status.get();
    ExitCode::from(if code == 0 { status as u8 } else { code as u8 })
}

/// [`clj_repl::LineHandler`] over a [`clj_repl::Session`]: submit
/// each completed unit, re-render the prompt (so `in-ns`
/// re-prompts at once), and remember the worst IO status (the
/// driver always reports 0, so broken-stdout surfaces here).
struct SessionHandler<W: std::io::Write> {
    session: clj_repl::Session<std::io::Empty, W>,
    prompt_text: std::rc::Rc<std::cell::RefCell<String>>,
    status: std::rc::Rc<std::cell::Cell<i32>>,
}

impl<W: std::io::Write> SessionHandler<W> {
    /// A handler over `session`, sharing `prompt_text` with the tty
    /// prompt closure; the IO status starts clean.
    fn new(
        session: clj_repl::Session<std::io::Empty, W>,
        prompt_text: std::rc::Rc<std::cell::RefCell<String>>,
    ) -> Self {
        Self {
            session,
            prompt_text,
            status: std::rc::Rc::new(std::cell::Cell::new(0)),
        }
    }
}

impl<W: std::io::Write> clj_repl::LineHandler for SessionHandler<W> {
    fn handle_line(&mut self, line: &str) -> clj_repl::LoopAction {
        let status = self.session.submit_complete(line);
        if status != 0 {
            self.status.set(status);
        }
        if let Some(rendered) = self.session.render_prompt() {
            *self.prompt_text.borrow_mut() = rendered;
        }
        clj_repl::LoopAction::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes the stdin-reading tests: only one may own fd 0.
    static STDIN_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A swapped fd 0, restored on drop (panic-safe).
    #[cfg(unix)]
    struct StdinSwap {
        saved: Option<rustix::fd::OwnedFd>,
    }

    #[cfg(unix)]
    impl StdinSwap {
        /// Feed `feed` (write end already closed) as fd 0.
        fn install(feed: rustix::fd::OwnedFd) -> Self {
            let saved = rustix::io::dup(rustix::stdio::stdin()).expect("save stdin");
            rustix::stdio::dup2_stdin(&feed).expect("swap stdin");
            Self { saved: Some(saved) }
        }
    }

    #[cfg(unix)]
    impl Drop for StdinSwap {
        fn drop(&mut self) {
            if let Some(saved) = self.saved.take() {
                let _ = rustix::stdio::dup2_stdin(&saved);
            }
        }
    }

    #[test]
    fn class_diagnostic_spells_both_shapes() {
        assert_eq!(
            class_diagnostic(clj_value::ThrowClass::ArithmeticException, ""),
            "ArithmeticException"
        );
        assert_eq!(
            class_diagnostic(clj_value::ThrowClass::ArithmeticException, "Divide by zero"),
            "ArithmeticException: Divide by zero"
        );
    }

    #[test]
    fn eval_diagnostic_internal_blames_input_start() {
        // `Internal` is unreachable via eval (only an escaping
        // `recur` builds one, and the analyzer guarantees every
        // `recur` a frame), so the error is synthesized.
        let err = SourceError::Internal("boom".to_owned());
        assert_eq!(eval_diagnostic(&err), "<eval>:1:1: internal error: boom");
    }

    #[test]
    fn run_diagnostic_internal_blames_script_start() {
        let err = SourceError::Internal("boom".to_owned());
        assert_eq!(
            run_diagnostic("script.clj", &err),
            "script.clj:1:1: internal error: boom"
        );
    }

    #[test]
    fn run_diagnostic_reader_blames_script_span() {
        // The `Reader` arm is defensive (`cmd_run`'s generated
        // wrapper always reads clean), so a REAL reader error from
        // the same `read_forms` exercises the formatting.
        let err = boot()
            .read_forms_live(")")
            .expect_err("stray closer must not read");
        assert!(matches!(&err, SourceError::Reader(_)));
        assert_eq!(
            run_diagnostic("script.clj", &err),
            "script.clj:1:2: Unmatched delimiter: )"
        );
    }

    #[test]
    fn run_diagnostic_compile_blames_script_start() {
        // The `Compile` arm is defensive (the wrapper always
        // compiles), so a REAL analyzer error exercises it: `nope`
        // fails resolve — never bare (bare is macro-arity/`declare`
        // only).
        let mut eval = boot();
        let err = eval.eval_source("nope").expect_err("nope must not resolve");
        assert!(matches!(&err, SourceError::Compile(_)));
        assert_eq!(
            run_diagnostic("script.clj", &err),
            "script.clj:1:1: Syntax error compiling: Unable to resolve symbol: nope in this context"
        );
    }

    #[test]
    fn run_preloads_clojure_string_201() {
        // JVM `clojure.main <script>` loads `clojure.main` →
        // `clojure.spec.alpha` → `clojure.string`, so
        // fully-qualified `clojure.string/...` uses resolve even
        // when the script never requires it (27 recipe scripts).
        // Pre-fix this exits 1 (`No such namespace`).
        let path =
            std::env::temp_dir().join(format!("clj-run-preload-string-{}.clj", std::process::id()));
        std::fs::write(
            &path,
            "(ns preload-probe)\n(def x (clojure.string/trim \"  a  \"))\n(assert (= x \"a\"))\n",
        )
        .expect("write probe script");
        let code = cmd_run(&path, &[]);
        let _ = std::fs::remove_file(&path);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn clj_string_escapes_every_special() {
        assert_eq!(
            clj_string("a\"b\\c\nd\re\tf\x01g\x7fh"),
            "\"a\\\"b\\\\c\\nd\\re\\tf\\u0001g\\u007fh\""
        );
    }

    #[test]
    fn string_bodies_scan_as_code() {
        let scan = EvalScanner::new("\"a\\b\"").run();
        assert!(scan.has_code);
        assert!(scan.edn_violation.is_none());
    }

    #[test]
    fn unterminated_string_still_reaches_reader() {
        // Entering a string counts as code.
        let scan = EvalScanner::new("\"abc").run();
        assert!(scan.has_code);
        assert!(scan.edn_violation.is_none());
    }

    #[test]
    fn char_tails_scan_as_code() {
        for src in ["\\", "\\5", "\\a", "\\a ", "\\newline"] {
            let scan = EvalScanner::new(src).run();
            assert!(scan.has_code, "{src:?} holds code");
            assert!(scan.edn_violation.is_none(), "{src:?} is edn-clean");
        }
    }

    #[test]
    fn quote_chars_flag_edn() {
        let scan = EvalScanner::new("'a").run();
        assert!(scan.has_code);
        let v = scan.edn_violation.expect("quote flags edn");
        assert_eq!((v.line, v.col), (1, 1));
        assert_eq!(v.lexeme, "'");
    }

    #[test]
    fn regex_dispatch_skips_body() {
        let scan = EvalScanner::new("#\"a\"").run();
        assert!(scan.has_code);
        let v = scan.edn_violation.expect("regex flags edn");
        assert_eq!((v.line, v.col), (1, 1));
        assert_eq!(v.lexeme, "#\"");
    }

    #[test]
    fn carriage_returns_normalize_to_newline() {
        // The (2, 1) violation position PROVES one-newline
        // normalization observationally.
        for src in ["a\r\n'x", "a\r'x"] {
            let scan = EvalScanner::new(src).run();
            assert!(scan.has_code);
            let v = scan.edn_violation.expect("quote flags edn");
            assert_eq!((v.line, v.col), (2, 1));
            assert_eq!(v.lexeme, "'");
        }
    }

    #[test]
    fn history_options_from_cli_flags() {
        assert_eq!(
            format!("{:?}", history_options(true, None)),
            "HistoryOptions { path: None, enabled: false }"
        );
        assert_eq!(
            format!(
                "{:?}",
                history_options(false, Some(PathBuf::from("/x/hist")))
            ),
            "HistoryOptions { path: Some(\"/x/hist\"), enabled: true }"
        );
        assert_eq!(
            format!("{:?}", history_options(false, None)),
            "HistoryOptions { path: None, enabled: true }"
        );
    }

    #[test]
    fn prompt_fn_clones_shared_text() {
        let text = std::rc::Rc::new(std::cell::RefCell::new("user=> ".to_owned()));
        let prompt = prompt_fn(std::rc::Rc::clone(&text));
        assert_eq!(prompt(), "user=> ");
        *text.borrow_mut() = "foo=> ".to_owned();
        assert_eq!(prompt(), "foo=> ");
    }

    fn test_handler(prompt: Option<&str>) -> SessionHandler<Vec<u8>> {
        let mut session = clj_repl::Session::new(boot(), std::io::empty(), Vec::new());
        session.set_prompt(prompt.map(str::to_owned));
        SessionHandler::new(
            session,
            std::rc::Rc::new(std::cell::RefCell::new(String::new())),
        )
    }

    #[test]
    fn handler_submits_and_reprompts() {
        let mut handler = test_handler(Some("{ns}=> "));
        let action = clj_repl::LineHandler::handle_line(&mut handler, "(+ 1 2)");
        assert!(matches!(action, clj_repl::LoopAction::Continue));
        assert_eq!(handler.prompt_text.borrow().as_str(), "user=> ");
        assert_eq!(handler.status.get(), 0);
        let SessionHandler { session, .. } = handler;
        let (_, _, out) = session.into_parts();
        assert_eq!(out, b"3\n".to_vec());
    }

    #[test]
    fn handler_without_prompt_keeps_text() {
        let mut handler = test_handler(None);
        let action = clj_repl::LineHandler::handle_line(&mut handler, "(+ 1 2)");
        assert!(matches!(action, clj_repl::LoopAction::Continue));
        assert!(handler.prompt_text.borrow().is_empty());
        let SessionHandler { session, .. } = handler;
        let (_, _, out) = session.into_parts();
        assert_eq!(out, b"3\n".to_vec());
    }

    struct FailWriter;

    impl std::io::Write for FailWriter {
        fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("boom"))
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::Error::other("boom"))
        }
    }

    #[test]
    fn print_failure_spells_both_signals() {
        let thrown = clj_eval::Signal::Throw(clj_value::Throwable {
            id: 1,
            class: clj_value::ThrowClass::ArithmeticException,
            message: Some(std::sync::Arc::from("Divide by zero")),
            data: None,
            cause: None,
        });
        assert_eq!(print_failure(thrown), "ArithmeticException: Divide by zero");
        // The forcing-impossible case: pins the helper's total
        // contract directly.
        assert_eq!(
            print_failure(clj_eval::Signal::Recur(vec![])),
            "internal error: recur escaped its loop or fn"
        );
    }

    /// A scripted line source (tty-loop tests without a terminal).
    struct ScriptSource {
        outcomes: std::collections::VecDeque<clj_repl::ReadOutcome>,
    }

    impl clj_repl::LineSource for ScriptSource {
        fn readline(&mut self, _prompt: &str) -> clj_repl::ReadOutcome {
            self.outcomes
                .pop_front()
                .unwrap_or(clj_repl::ReadOutcome::Eof)
        }

        fn is_complete(&self, _input: &str) -> bool {
            true
        }

        fn load_history(&mut self, _path: &std::path::Path) -> Result<(), String> {
            Ok(())
        }

        fn append_history(&mut self, _path: &std::path::Path) -> Result<(), String> {
            Ok(())
        }

        fn add_history_entry(&mut self, _line: &str) {}
    }

    #[test]
    fn tty_loop_runs_headless_with_stub_source() {
        let mut session = clj_repl::Session::new(boot(), std::io::empty(), Vec::new());
        session.set_prompt(Some("{ns}=> ".to_owned()));
        let source = ScriptSource {
            outcomes: [clj_repl::ReadOutcome::Line("(+ 1 2)".to_owned())]
                .into_iter()
                .collect(),
        };
        // Enabled history with an explicit (stub-ignored) path, so
        // the load/append seam executes without touching home.
        let hist = std::env::temp_dir().join("clj-tty-history-stub");
        let code = cmd_repl_tty(session, source, clj_repl::HistoryOptions::with_path(hist));
        assert_eq!(code, ExitCode::SUCCESS);
    }

    #[test]
    fn tty_degradation_streams_when_backend_fails() {
        use std::io::IsTerminal;
        // The fallback reads inherited stdin: run only piped (CI
        // always pipes); a live terminal skips silently. (`then`,
        // not `if`, so the skip leaves no dead branch line.)
        (!std::io::stdin().is_terminal()).then(|| {
            let _serial = STDIN_SERIAL.lock().expect("stdin serial");
            let code = cmd_repl_with(None, true, true, |_, _| Err("boom".to_owned()));
            assert_eq!(code, ExitCode::SUCCESS);
        });
    }

    #[test]
    fn tty_loop_with_real_source_exits_on_eof_stdin() {
        use std::io::IsTerminal;
        // rustyline would block on a live terminal; CI pipes EOF.
        (!std::io::stdin().is_terminal()).then(|| {
            let _serial = STDIN_SERIAL.lock().expect("stdin serial");
            let code = cmd_repl_with(None, true, true, make_rustyline_source);
            assert_eq!(code, ExitCode::SUCCESS);
        });
    }

    /// The validator runs over a piped line (covers
    /// `ReplHelper::validate`, unreachable through any other
    /// headless path — `ValidationContext::new` is `pub(crate)`):
    /// fd 0 is swapped onto a one-line pipe for the call and
    /// restored after (panic-safe via the guard). A live
    /// terminal skips like its siblings.
    #[cfg(unix)]
    #[test]
    fn tty_loop_with_real_source_validates_piped_line() {
        use std::io::IsTerminal;
        (!std::io::stdin().is_terminal()).then(|| {
            let _serial = STDIN_SERIAL.lock().expect("stdin serial");
            let (feed_read, feed_write) = rustix::pipe::pipe().expect("stdin pipe");
            {
                let mut feed = std::fs::File::from(feed_write);
                use std::io::Write;
                feed.write_all(b"(+ 1 2)\n").expect("feed stdin line");
            }
            let swap = StdinSwap::install(feed_read);
            let code = cmd_repl_with(None, true, true, make_rustyline_source);
            drop(swap);
            assert_eq!(code, ExitCode::SUCCESS);
        });
    }

    #[test]
    fn handler_remembers_io_status() {
        // `submit_complete` returns 1 when the write fails, so the
        // handler records it.
        let mut session = clj_repl::Session::new(boot(), std::io::empty(), FailWriter);
        session.set_prompt(Some("{ns}=> ".to_owned()));
        let mut handler = SessionHandler::new(
            session,
            std::rc::Rc::new(std::cell::RefCell::new(String::new())),
        );
        let action = clj_repl::LineHandler::handle_line(&mut handler, "(+ 1 2)");
        assert!(matches!(action, clj_repl::LoopAction::Continue));
        assert_eq!(handler.status.get(), 1);
        assert_eq!(handler.prompt_text.borrow().as_str(), "user=> ");
        let SessionHandler { session, .. } = handler;
        let (_, _, mut out) = session.into_parts();
        assert!(std::io::Write::flush(&mut out).is_err());
    }

    #[test]
    fn eval_installs_driver_and_evaluates() {
        // Covers the S18 install call at the top of `cmd_eval`.
        assert_eq!(cmd_eval("(+ 1 2)", false), ExitCode::SUCCESS);
    }

    #[cfg(not(feature = "aot"))]
    #[test]
    fn build_unlinked_reports_shared_line() {
        // Covers the S18 shared unlinked line (T11's CLI side —
        // the text itself is pinned by `driver_unset` on the same
        // const, so the two can never drift).
        assert_eq!(
            cmd_build_compile(
                std::path::Path::new("in.clj"),
                "(+ 1 2)",
                std::path::Path::new("o"),
                None,
                None,
                None
            ),
            ExitCode::from(1)
        );
    }

    #[cfg(not(feature = "aot"))]
    #[test]
    fn jit_unlinked_reports_shared_line() {
        // The `jit` side of the shared unlinked line (gap-121
        // slice 1 — the text lives on `UNLINKED_JIT_LINE`, so
        // the CLI and any future hook can never drift).
        assert_eq!(
            cmd_jit_compile(std::path::Path::new("in.clj"), "(+ 1 2)", &[]),
            ExitCode::from(1)
        );
    }

    #[test]
    fn run_installs_driver_and_runs_script() {
        // Covers the S18 install call at the top of `cmd_run`.
        let dir = std::env::temp_dir().join(format!("clj-s18-run-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("s.clj");
        std::fs::write(&script, "(+ 1 2)\n").unwrap();
        assert_eq!(cmd_run(&script, &[]), ExitCode::SUCCESS);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
