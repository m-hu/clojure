// Copyright (c) Rich Hickey. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.
// You must not remove this notice, or any other, from this software.

//! Differential corpus: every case in `tests/corpus/cases.edn`
//! against the checked-in JVM oracle `expectations.edn` (no JVM
//! needed), plus the full-file hash corpus (`files.edn`).
//!
//! Exact cases compare canonical renderings (default + print-meta
//! configs, gensym-normalized) or error triples
//! (message + class + line + col). `:diverge` cases compare against
//! the reviewed `ours.edn` snapshots instead (refresh with
//! `UPDATE_OURS=1 cargo test -p clj-reader --test snapshot`, then
//! review the diff).

#[path = "harness.rs"]
mod harness;

use std::path::PathBuf;

use harness::{
    canon, case_opts, compare_outcomes, fnv1a_hex, our_err, our_outcome, parse_cases,
    parse_expectations, parse_files, parse_ours, parse_resolver, read_data_file, run_ours,
};

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/corpus")
}

#[test]
fn corpus_cases() {
    let dir = corpus_dir();
    let cases = parse_cases(&read_data_file(&dir.join("cases.edn")));
    let exp_value = read_data_file(&dir.join("expectations.edn"));
    let resolver = parse_resolver(&exp_value);
    let expected = parse_expectations(&exp_value);
    let snapshots = parse_ours(&dir.join("ours.edn"));

    // Case/expectation skew fails loudly (stale regen or edited cases).
    for case in &cases {
        assert!(
            expected.contains_key(&case.id),
            "case {} has no checked-in oracle expectation (run ./regen.sh)",
            case.id
        );
    }
    for id in expected.keys() {
        assert!(
            cases.iter().any(|c| &c.id == id),
            "stale oracle expectation for removed case {id} (run ./regen.sh)"
        );
    }

    let mut failures = Vec::new();
    for case in &cases {
        let opts = case_opts(case, &resolver);
        let got = match run_ours(&case.input, &opts) {
            Ok(values) => our_outcome(values),
            Err(e) => our_err(&e),
        };
        let result = match &case.diverge {
            None => compare_outcomes(&got, &expected[&case.id]),
            Some(kind) => match snapshots.get(&case.id) {
                Some(want) => compare_outcomes(&got, want).map_err(|d| format!("[{kind}] {d}")),
                None => Err(format!(
                    "[{kind}] missing :ours snapshot (UPDATE_OURS=1 cargo test -p clj-reader --test snapshot)"
                )),
            },
        };
        if let Err(diff) = result {
            failures.push(format!(
                "--- {}\ninput: {:?}\ndiverge: {:?}\n{diff}",
                case.id, case.input, case.diverge
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} corpus failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(!cases.is_empty(), "no cases loaded");
}

#[test]
fn corpus_files() {
    let dir = corpus_dir();
    // Full-file inputs are vendored under tests/corpus/inputs/
    // (mirroring the removed src/ + test/ layout); files.edn rels
    // resolve there.
    let inputs = dir.join("inputs");
    let entries = parse_files(&read_data_file(&dir.join("files.edn")));
    let resolver = parse_resolver(&read_data_file(&dir.join("expectations.edn")));

    let mut failures = Vec::new();
    let mut checked = 0;
    for entry in &entries {
        if let Some(reason) = &entry.exclude {
            eprintln!("skipping {} ({reason})", entry.path);
            continue;
        }
        let text = std::fs::read_to_string(inputs.join(&entry.path)).unwrap();
        let opts = clj_reader::ReaderOpts {
            current_ns: resolver.current_ns.clone(),
            aliases: resolver.aliases.clone(),
            refers: resolver.refers.clone(),
            class_map: resolver.class_map.clone(),
            // Like the driver: full files read with `*read-eval*`
            // false (#125).
            read_eval: false,
            ..Default::default()
        };
        let forms = match run_ours(&text, &opts) {
            Ok(forms) => forms,
            Err(e) => {
                failures.push(format!("--- {}\nours failed: {e}", entry.path));
                continue;
            }
        };
        if forms.len() != entry.count {
            failures.push(format!(
                "--- {}\nform count: ours {} vs want {}",
                entry.path,
                forms.len(),
                entry.count
            ));
            continue;
        }
        let canon_default = forms
            .iter()
            .map(|f| canon(f, false))
            .collect::<Vec<_>>()
            .join("\n");
        let canon_meta = forms
            .iter()
            .map(|f| canon(f, true))
            .collect::<Vec<_>>()
            .join("\n");
        // Gensym ids are run-order dependent on both sides: normalize first.
        let ours_default = fnv1a_hex(&harness::normalize_gensyms(&canon_default));
        let ours_meta = fnv1a_hex(&harness::normalize_gensyms(&canon_meta));
        if ours_default != entry.fnv_default || ours_meta != entry.fnv_printmeta {
            failures.push(format!(
                "--- {}\nhash: ours {ours_default}/{ours_meta} vs want {}/{}",
                entry.path, entry.fnv_default, entry.fnv_printmeta
            ));
        }
        checked += 1;
    }
    assert!(
        failures.is_empty(),
        "{} file failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(checked > 0, "no files checked");
    eprintln!("checked {checked} files");
}
