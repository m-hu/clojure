// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! S8 coverage: force every lazy producer through strict consumers.
//! `run_last` realizes via `realize_deep`, which bypasses the
//! `step_*` forcing paths in `lazy.rs`; only strict natives (`first`,
//! `rest`, `doall`, `nth`, `count`, ...) stepping a pending seq walk
//! them, so each producer below is forced through several consumers
//! plus empty/singleton/exhausted edges.

use clj_eval::{Evaluator, SourceError, boot};
use clj_value::{PrintOpts, Value, pr_to_string};

fn pr(v: &Value) -> String {
    pr_to_string(v, &PrintOpts::pr())
}

fn run_last(eval: &mut Evaluator, src: &str) -> String {
    let vs = eval.eval_source(src).unwrap();
    let v = vs.last().unwrap();
    let r = eval.realize_deep(v).unwrap();
    pr(&r)
}

fn uncaught(eval: &mut Evaluator, src: &str) -> String {
    match eval.eval_source(src) {
        Err(SourceError::Uncaught(t)) => format!(
            "{} | {}",
            t.class.jvm_name(),
            t.message.as_deref().unwrap_or("")
        ),
        Err(other) => panic!("expected an uncaught throw, got {other}"),
        Ok(vs) => panic!("expected an uncaught throw for {src}, got {vs:?}"),
    }
}

#[test]
fn lazy_core_producer_rows() {
    let mut eval = boot();
    // map / filter / remove through first/rest/doall.
    assert_eq!(run_last(&mut eval, "(first (map inc [1 2 3]))"), "2");
    assert_eq!(run_last(&mut eval, "(rest (map inc [1 2 3]))"), "(3 4)");
    assert_eq!(run_last(&mut eval, "(doall (map inc [1 2 3]))"), "(2 3 4)");
    assert_eq!(
        run_last(&mut eval, "(first (map + [1 2] [10 20 30]))"),
        "11"
    );
    assert_eq!(
        run_last(&mut eval, "(doall (map + [1 2] [10 20] [100 200]))"),
        "(111 222)"
    );
    assert_eq!(
        run_last(&mut eval, "(doall (filter odd? [1 2 3 4]))"),
        "(1 3)"
    );
    assert_eq!(run_last(&mut eval, "(first (filter odd? [2 4 5]))"), "5");
    assert_eq!(
        run_last(&mut eval, "(doall (remove odd? [1 2 3 4]))"),
        "(2 4)"
    );
    assert_eq!(run_last(&mut eval, "(first (remove odd? [1 3 4]))"), "4");
    // take / drop incl. zero, overrun, and negative edges.
    assert_eq!(run_last(&mut eval, "(doall (take 2 [1 2 3]))"), "(1 2)");
    assert_eq!(run_last(&mut eval, "(doall (take 0 [1 2]))"), "()");
    assert_eq!(run_last(&mut eval, "(doall (take 5 [1 2]))"), "(1 2)");
    assert_eq!(run_last(&mut eval, "(first (take 2 [7 8 9]))"), "7");
    assert_eq!(run_last(&mut eval, "(doall (drop 2 [1 2 3 4]))"), "(3 4)");
    assert_eq!(run_last(&mut eval, "(doall (drop 0 [1 2]))"), "(1 2)");
    assert_eq!(run_last(&mut eval, "(doall (drop 9 [1 2]))"), "()");
    assert_eq!(run_last(&mut eval, "(doall (take -1 [1 2]))"), "()");
    assert_eq!(run_last(&mut eval, "(doall (drop -1 [1 2]))"), "(1 2)");
    // take/drop-while incl. empty-match edges.
    assert_eq!(
        run_last(&mut eval, "(doall (take-while odd? [1 3 2 4]))"),
        "(1 3)"
    );
    assert_eq!(run_last(&mut eval, "(doall (take-while odd? [2 4]))"), "()");
    assert_eq!(
        run_last(&mut eval, "(doall (drop-while odd? [1 3 2 4]))"),
        "(2 4)"
    );
    assert_eq!(run_last(&mut eval, "(doall (drop-while odd? [1 3]))"), "()");
    // take-nth incl. step 1 and over-infinite.
    assert_eq!(
        run_last(&mut eval, "(doall (take-nth 2 [1 2 3 4 5]))"),
        "(1 3 5)"
    );
    assert_eq!(run_last(&mut eval, "(doall (take-nth 1 [1 2]))"), "(1 2)");
    assert_eq!(run_last(&mut eval, "(first (take-nth 3 [1 2 3 4]))"), "1");
    assert_eq!(
        run_last(&mut eval, "(doall (take 3 (take-nth 2 (iterate inc 0))))"),
        "(0 2 4)"
    );
    // concat / mapcat incl. nil items and empty outers.
    assert_eq!(
        run_last(&mut eval, "(doall (concat [1 2] [3] nil [4]))"),
        "(1 2 3 4)"
    );
    assert_eq!(run_last(&mut eval, "(doall (concat))"), "()");
    assert_eq!(run_last(&mut eval, "(doall (concat [1]))"), "(1)");
    assert_eq!(run_last(&mut eval, "(first (concat nil [5]))"), "5");
    assert_eq!(run_last(&mut eval, "(doall (mapcat list [1 2]))"), "(1 2)");
    assert_eq!(run_last(&mut eval, "(doall (mapcat list []))"), "()");
}

#[test]
fn lazy_combinator_rows() {
    let mut eval = boot();
    // interleave incl. uneven, empty, and 3-way queues.
    assert_eq!(
        run_last(&mut eval, "(doall (interleave [1 2] [3 4]))"),
        "(1 3 2 4)"
    );
    assert_eq!(
        run_last(&mut eval, "(doall (interleave [1 2 3] [4]))"),
        "(1 4)"
    );
    assert_eq!(run_last(&mut eval, "(doall (interleave))"), "()");
    assert_eq!(run_last(&mut eval, "(doall (interleave [1] []))"), "()");
    assert_eq!(
        run_last(&mut eval, "(doall (interleave [1 2] [3 4] [5 6]))"),
        "(1 3 5 2 4 6)"
    );
    assert_eq!(run_last(&mut eval, "(first (interleave [9] [8]))"), "9");
    // interpose incl. singleton/empty and rest-walking.
    assert_eq!(
        run_last(&mut eval, "(doall (interpose :x [1 2 3]))"),
        "(1 :x 2 :x 3)"
    );
    assert_eq!(run_last(&mut eval, "(doall (interpose :x [1]))"), "(1)");
    assert_eq!(run_last(&mut eval, "(doall (interpose :x []))"), "()");
    assert_eq!(run_last(&mut eval, "(rest (interpose :x [1 2]))"), "(:x 2)");
    // distinct / dedupe incl. empty edges.
    assert_eq!(
        run_last(&mut eval, "(doall (distinct [1 2 1 3 2]))"),
        "(1 2 3)"
    );
    assert_eq!(run_last(&mut eval, "(first (distinct [5 5 6]))"), "5");
    assert_eq!(run_last(&mut eval, "(doall (distinct nil))"), "()");
    assert_eq!(
        run_last(&mut eval, "(doall (dedupe [1 1 2 2 1]))"),
        "(1 2 1)"
    );
    assert_eq!(run_last(&mut eval, "(doall (dedupe []))"), "()");
    // keep / keep-indexed / map-indexed incl. all-nil keeps.
    assert_eq!(
        run_last(
            &mut eval,
            "(doall (keep (fn [x] (if (odd? x) (* x 10))) [1 2 3]))"
        ),
        "(10 30)"
    );
    assert_eq!(run_last(&mut eval, "(doall (keep (fn [x]) [1 2]))"), "()");
    assert_eq!(
        run_last(
            &mut eval,
            "(doall (keep-indexed (fn [i x] (if (odd? i) x)) [10 20 30 40]))"
        ),
        "(20 40)"
    );
    assert_eq!(
        run_last(&mut eval, "(doall (map-indexed (fn [i x] [i x]) [7 8]))"),
        "([0 7] [1 8])"
    );
    assert_eq!(
        run_last(&mut eval, "(first (map-indexed vector [5 6]))"),
        "[0 5]"
    );
    // reductions incl. init, singleton, and empty.
    assert_eq!(
        run_last(&mut eval, "(doall (reductions + [1 2 3]))"),
        "(1 3 6)"
    );
    assert_eq!(
        run_last(&mut eval, "(doall (reductions + 10 [1 2]))"),
        "(10 11 13)"
    );
    assert_eq!(run_last(&mut eval, "(first (reductions + [4 5]))"), "4");
    assert_eq!(run_last(&mut eval, "(doall (reductions + [5]))"), "(5)");
    // No-init over empty seeds from `(f)` (oracle-verbatim `(0)`).
    assert_eq!(run_last(&mut eval, "(doall (reductions + []))"), "(0)");
    assert_eq!(run_last(&mut eval, "(doall (reductions + 10 []))"), "(10)");
    // replace preserves vectors, seqs stay seqs.
    assert_eq!(
        run_last(&mut eval, "(doall (replace {1 :a} [1 2 1]))"),
        "[:a 2 :a]"
    );
    assert_eq!(
        run_last(&mut eval, "(doall (replace {1 :a} (seq [1 2 1])))"),
        "(:a 2 :a)"
    );
}

#[test]
fn lazy_partition_rows() {
    let mut eval = boot();
    // partition incl. dropped tails, steps, and pads.
    assert_eq!(
        run_last(&mut eval, "(doall (partition 2 [1 2 3 4]))"),
        "((1 2) (3 4))"
    );
    assert_eq!(
        run_last(&mut eval, "(doall (partition 2 [1 2 3]))"),
        "((1 2))"
    );
    assert_eq!(
        run_last(&mut eval, "(doall (partition 2 1 [1 2 3]))"),
        "((1 2) (2 3))"
    );
    assert_eq!(
        run_last(&mut eval, "(doall (partition 2 3 [1 2 3 4 5]))"),
        "((1 2) (4 5))"
    );
    // Pad arity is [n step pad coll] (oracle-verbatim).
    assert_eq!(
        run_last(&mut eval, "(doall (partition 2 2 [:p] [1 2 3]))"),
        "((1 2) (3 :p))"
    );
    assert_eq!(
        run_last(&mut eval, "(doall (partition 2 2 (repeat :p) [1 2 3]))"),
        "((1 2) (3 :p))"
    );
    assert_eq!(run_last(&mut eval, "(doall (partition 2 []))"), "()");
    assert_eq!(
        run_last(&mut eval, "(first (partition 2 [9 8 7 6]))"),
        "(9 8)"
    );
    // partition-all keeps trailing partials.
    assert_eq!(
        run_last(&mut eval, "(doall (partition-all 2 [1 2 3]))"),
        "((1 2) (3))"
    );
    assert_eq!(
        run_last(&mut eval, "(doall (partition-all 2 1 [1 2 3]))"),
        "((1 2) (2 3) (3))"
    );
    // partition-by incl. counting and empty input.
    assert_eq!(
        run_last(&mut eval, "(doall (partition-by odd? [1 3 2 4 5]))"),
        "((1 3) (2 4) (5))"
    );
    assert_eq!(
        run_last(&mut eval, "(first (partition-by odd? [2 4 5]))"),
        "(2 4)"
    );
    assert_eq!(
        run_last(&mut eval, "(count (partition-by odd? [1 2 2]))"),
        "2"
    );
    assert_eq!(run_last(&mut eval, "(doall (partition-by odd? []))"), "()");
    // split-with / split-at return vectors of seqs.
    assert_eq!(
        run_last(&mut eval, "(doall (split-with odd? [1 3 2 4]))"),
        "[(1 3) (2 4)]"
    );
    assert_eq!(
        run_last(&mut eval, "(doall (split-with odd? [1 3]))"),
        "[(1 3) ()]"
    );
    assert_eq!(
        run_last(&mut eval, "(doall (split-at 2 [1 2 3 4]))"),
        "[(1 2) (3 4)]"
    );
    assert_eq!(
        run_last(&mut eval, "(doall (split-at 0 [1 2]))"),
        "[() (1 2)]"
    );
}

#[test]
fn lazy_generative_rows() {
    let mut eval = boot();
    // cycle incl. empty (every cycle below is take-bounded).
    assert_eq!(
        run_last(&mut eval, "(doall (take 5 (cycle [1 2])))"),
        "(1 2 1 2 1)"
    );
    assert_eq!(run_last(&mut eval, "(first (cycle [7 8]))"), "7");
    assert_eq!(run_last(&mut eval, "(doall (take 1 (cycle [])))"), "()");
    assert_eq!(run_last(&mut eval, "(first (cycle []))"), "nil");
    // iterate incl. nth-walking and drop-over-infinite.
    assert_eq!(
        run_last(&mut eval, "(doall (take 3 (iterate inc 0)))"),
        "(0 1 2)"
    );
    assert_eq!(run_last(&mut eval, "(first (iterate inc 5))"), "5");
    assert_eq!(run_last(&mut eval, "(nth (iterate inc 0) 3)"), "3");
    assert_eq!(
        run_last(&mut eval, "(doall (take 2 (drop 3 (iterate inc 0))))"),
        "(3 4)"
    );
    // repeat / repeatedly incl. zero count.
    assert_eq!(
        run_last(&mut eval, "(doall (take 3 (repeat 7)))"),
        "(7 7 7)"
    );
    assert_eq!(run_last(&mut eval, "(doall (repeat 2 :a))"), "(:a :a)");
    assert_eq!(run_last(&mut eval, "(doall (repeat 0 :x))"), "()");
    assert_eq!(run_last(&mut eval, "(doall (repeat -1 :x))"), "()");
    assert_eq!(run_last(&mut eval, "(first (repeat 9))"), "9");
    assert_eq!(
        run_last(&mut eval, "(doall (take 2 (repeatedly (fn [] 9))))"),
        "(9 9)"
    );
    assert_eq!(run_last(&mut eval, "(first (repeatedly (fn [] 8)))"), "8");
    // range incl. open-ended, stepped, and degenerate bounds.
    assert_eq!(run_last(&mut eval, "(doall (range 3))"), "(0 1 2)");
    assert_eq!(run_last(&mut eval, "(doall (take 4 (range)))"), "(0 1 2 3)");
    assert_eq!(run_last(&mut eval, "(doall (range 1 6 2))"), "(1 3 5)");
    assert_eq!(run_last(&mut eval, "(doall (range 5 1 -1))"), "(5 4 3 2)");
    assert_eq!(run_last(&mut eval, "(doall (range 3 1))"), "()");
    assert_eq!(run_last(&mut eval, "(doall (range 1 1))"), "()");
    assert_eq!(run_last(&mut eval, "(first (range 10))"), "0");
    assert_eq!(run_last(&mut eval, "(nth (range 10) 4)"), "4");
    // Long walks over chunked realizations.
    assert_eq!(run_last(&mut eval, "(count (map inc (range 40)))"), "40");
    assert_eq!(run_last(&mut eval, "(nth (map inc (range 40)) 33)"), "34");
    assert_eq!(run_last(&mut eval, "(last (map inc (range 40)))"), "40");
}

#[test]
fn lazy_collection_seq_rows() {
    let mut eval = boot();
    // keys / vals (multi-entry order pinned via sort).
    assert_eq!(run_last(&mut eval, "(doall (keys {:a 1}))"), "(:a)");
    assert_eq!(run_last(&mut eval, "(doall (vals {:a 1}))"), "(1)");
    assert_eq!(run_last(&mut eval, "(sort (keys {:b 1 :a 2}))"), "(:a :b)");
    assert_eq!(run_last(&mut eval, "(sort (vals {:b 2 :a 1}))"), "(1 2)");
    assert_eq!(run_last(&mut eval, "(first (keys {:k 9}))"), ":k");
    assert_eq!(run_last(&mut eval, "(doall (keys {}))"), "nil");
    assert_eq!(run_last(&mut eval, "(seq {})"), "nil");
    // seq / drop over maps (drop walks the map-shift path).
    assert_eq!(run_last(&mut eval, "(doall (seq {:a 1}))"), "([:a 1])");
    assert_eq!(run_last(&mut eval, "(doall (drop 0 {:a 1}))"), "([:a 1])");
    assert_eq!(run_last(&mut eval, "(doall (drop 1 {:a 1}))"), "()");
    assert_eq!(run_last(&mut eval, "(first {:a 1})"), "[:a 1]");
    assert_eq!(run_last(&mut eval, "(key (first {:a 1}))"), ":a");
    assert_eq!(run_last(&mut eval, "(val (first {:a 1}))"), "1");
    // cons cells over pending tails.
    assert_eq!(
        run_last(&mut eval, "(doall (cons 0 (map inc [1 2])))"),
        "(0 2 3)"
    );
    assert_eq!(run_last(&mut eval, "(first (cons 0 (map inc [1])))"), "0");
    assert_eq!(
        run_last(&mut eval, "(rest (cons 0 (map inc [1 2])))"),
        "(2 3)"
    );
    // String seqs step per character.
    assert_eq!(
        run_last(&mut eval, "(doall (seq \"abc\"))"),
        "(\\a \\b \\c)"
    );
    assert_eq!(run_last(&mut eval, "(first \"abc\")"), "\\a");
    assert_eq!(run_last(&mut eval, "(nth \"abc\" 1)"), "\\b");
    assert_eq!(run_last(&mut eval, "(nth \"abc\" 0)"), "\\a");
    assert_eq!(run_last(&mut eval, "(count \"abc\")"), "3");
    assert_eq!(run_last(&mut eval, "(doall (take 2 \"abc\"))"), "(\\a \\b)");
    assert_eq!(run_last(&mut eval, "(doall (drop 1 \"abc\"))"), "(\\b \\c)");
    assert_eq!(
        run_last(
            &mut eval,
            "(doall (filter (fn [c] (= (str c) \"b\")) \"abc\"))"
        ),
        "(\\b)"
    );
    // Set seqs (order pinned via sort / count).
    assert_eq!(run_last(&mut eval, "(first #{:a})"), ":a");
    assert_eq!(run_last(&mut eval, "(count #{:a :b})"), "2");
    assert_eq!(run_last(&mut eval, "(sort (seq #{:b :a}))"), "(:a :b)");
    // Subvector seqs step with offset bounds.
    // doall returns its arg: the subvector prints as a vector.
    assert_eq!(run_last(&mut eval, "(doall (subvec [1 2 3] 1))"), "[2 3]");
    assert_eq!(run_last(&mut eval, "(first (subvec [1 2] 1))"), "2");
    assert_eq!(run_last(&mut eval, "(count (subvec [1 2 3] 1 2))"), "1");
    // List seqs and entry seqs.
    assert_eq!(run_last(&mut eval, "(first (list 1 2))"), "1");
    assert_eq!(run_last(&mut eval, "(rest (quote (1 2)))"), "(2)");
}

#[test]
fn lazy_forcing_consumer_rows() {
    let mut eval = boot();
    // next / nnext / second over pending seqs.
    assert_eq!(run_last(&mut eval, "(next (map inc [1]))"), "nil");
    assert_eq!(run_last(&mut eval, "(next (map inc [1 2]))"), "(3)");
    assert_eq!(run_last(&mut eval, "(next nil)"), "nil");
    assert_eq!(run_last(&mut eval, "(nnext (map inc [1 2 3]))"), "(4)");
    assert_eq!(run_last(&mut eval, "(second (map inc [1 2 3]))"), "3");
    // nth / nthrest over pending seqs.
    assert_eq!(run_last(&mut eval, "(nth (map inc [1 2 3]) 2)"), "4");
    assert_eq!(run_last(&mut eval, "(nth (filter odd? [1 2 3]) 1)"), "3");
    assert_eq!(run_last(&mut eval, "(nth (map inc [1 2 3]) 0)"), "2");
    assert_eq!(run_last(&mut eval, "(nthrest (map inc [1 2 3]) 2)"), "(4)");
    // n=0 returns coll un-seq'd (oracle + control.rs nthrest_rows pin).
    assert_eq!(run_last(&mut eval, "(nthrest [1 2 3] 0)"), "[1 2 3]");
    // last / butlast / take-last / drop-last over pending seqs.
    assert_eq!(run_last(&mut eval, "(last (map inc [1 2 3]))"), "4");
    assert_eq!(run_last(&mut eval, "(last nil)"), "nil");
    assert_eq!(run_last(&mut eval, "(last [])"), "nil");
    assert_eq!(run_last(&mut eval, "(butlast [1 2 3])"), "(1 2)");
    assert_eq!(
        run_last(&mut eval, "(take-last 2 (map inc [1 2 3]))"),
        "(3 4)"
    );
    assert_eq!(
        run_last(&mut eval, "(drop-last (map inc [1 2 3]))"),
        "(2 3)"
    );
    // seq / empty? / count edges.
    assert_eq!(run_last(&mut eval, "(seq (map inc [1]))"), "(2)");
    assert_eq!(run_last(&mut eval, "(seq nil)"), "nil");
    assert_eq!(run_last(&mut eval, "(seq [])"), "nil");
    assert_eq!(run_last(&mut eval, "(empty? (map inc [1]))"), "false");
    assert_eq!(run_last(&mut eval, "(empty? nil)"), "true");
    assert_eq!(run_last(&mut eval, "(count nil)"), "0");
    assert_eq!(run_last(&mut eval, "(count [])"), "0");
    assert_eq!(run_last(&mut eval, "(count (map inc [1 2 3]))"), "3");
    // first / rest over nil and empties.
    assert_eq!(run_last(&mut eval, "(first nil)"), "nil");
    assert_eq!(run_last(&mut eval, "(rest nil)"), "()");
    assert_eq!(run_last(&mut eval, "(first [])"), "nil");
    assert_eq!(run_last(&mut eval, "(rest [])"), "()");
    // Predicates and folds force through seq.
    assert_eq!(run_last(&mut eval, "(every? odd? [1 3])"), "true");
    assert_eq!(run_last(&mut eval, "(every? odd? [1 2])"), "false");
    assert_eq!(run_last(&mut eval, "(some odd? [2 4 5])"), "true");
    assert_eq!(run_last(&mut eval, "(into [] (map inc [1 2]))"), "[2 3]");
    assert_eq!(run_last(&mut eval, "(reduce + (map inc [1 2 3]))"), "9");
    // String results print quoted under pr.
    assert_eq!(
        run_last(&mut eval, "(apply str (interpose \",\" [1 2 3]))"),
        "\"1,2,3\""
    );
    // Equality and conj walk pending seqs.
    assert_eq!(run_last(&mut eval, "(= (map inc [1 2]) [2 3])"), "true");
    assert_eq!(
        run_last(&mut eval, "(= (map inc [1 2]) (quote (2 3)))"),
        "true"
    );
    assert_eq!(run_last(&mut eval, "(conj (map inc [1 2]) 0)"), "(0 2 3)");
    // doseq forces through macroexpansion; dorun/doall return shapes.
    assert_eq!(run_last(&mut eval, "(doseq [x (map inc [1 2])] x)"), "nil");
    assert_eq!(run_last(&mut eval, "(dorun (map inc [1 2 3]))"), "nil");
    assert_eq!(run_last(&mut eval, "(dorun 2 (map inc [1 2 3]))"), "nil");
    // Explicit lazy-seq forms.
    assert_eq!(
        run_last(&mut eval, "(doall (lazy-seq (list 1 2)))"),
        "(1 2)"
    );
    assert_eq!(run_last(&mut eval, "(first (lazy-seq (list 9 8)))"), "9");
    // str / pr-str over pending seqs.
    assert_eq!(run_last(&mut eval, "(str [1 2])"), "\"[1 2]\"");
    assert_eq!(run_last(&mut eval, "(pr-str (map inc [1 2]))"), "\"(2 3)\"");
    assert_eq!(run_last(&mut eval, "(get (frequencies [1 2 1]) 1)"), "2");
    assert_eq!(
        run_last(&mut eval, "(get (group-by odd? [1 2 3]) true)"),
        "[1 3]"
    );
    assert_eq!(run_last(&mut eval, "(reverse (map inc [1 2]))"), "(3 2)");
    assert_eq!(run_last(&mut eval, "(sort [3 1 2])"), "(1 2 3)");
    assert_eq!(
        run_last(&mut eval, "(sort-by first [[2 :b] [1 :a]])"),
        "([1 :a] [2 :b])"
    );
}

#[test]
fn lazy_error_rows() {
    let mut eval = boot();
    // Non-seqables blame ISeq (classes mirror coll.rs pins).
    assert_eq!(
        uncaught(&mut eval, "(first 5)"),
        "java.lang.IllegalArgumentException | Don't know how to create ISeq from: java.lang.Long"
    );
    assert_eq!(
        uncaught(&mut eval, "(rest :k)"),
        "java.lang.IllegalArgumentException | Don't know how to create ISeq from: clojure.lang.Keyword"
    );
    assert_eq!(
        uncaught(&mut eval, "(doall 5)"),
        "java.lang.IllegalArgumentException | Don't know how to create ISeq from: java.lang.Long"
    );
    assert_eq!(
        uncaught(&mut eval, "(seq :k)"),
        "java.lang.IllegalArgumentException | Don't know how to create ISeq from: clojure.lang.Keyword"
    );
    // nth overruns blame bounds even through pending seqs.
    assert_eq!(
        uncaught(&mut eval, "(nth (map inc [1 2]) 5)"),
        "java.lang.IndexOutOfBoundsException | "
    );
    assert_eq!(
        uncaught(&mut eval, "(nth (map inc [1 2]) -1)"),
        "java.lang.IndexOutOfBoundsException | "
    );
}

#[test]
fn lazy_chunked_cons_and_exhaustion_rows() {
    let mut eval = boot();
    // A cast names the ChunkedCons class (rest of a forced chunk).
    assert_eq!(
        uncaught(&mut eval, "(+ 1 (rest (map inc [1 2 3])))"),
        "java.lang.ClassCastException | class clojure.lang.ChunkedCons cannot be cast to class java.lang.Number (clojure.lang.ChunkedCons is in unnamed module of loader 'app'; java.lang.Number is in module java.base of loader 'bootstrap')"
    );
    // rest after next-to-end falls through the emptied chunk.
    assert_eq!(
        run_last(&mut eval, "(let [s (map inc [1])] (next s) (rest s))"),
        "()"
    );
}

#[test]
fn lazy_cons_runs_rest_as_lazy_seqs() {
    let mut eval = boot();
    // `take` / `interpose` (and `partition`'s padded window) force a
    // realized run of items at once, but cons one item per step on the
    // JVM: their rests stay unchunked LazySeqs, whether `rest` forces
    // the run or walks an already-forced one.
    assert_eq!(
        run_last(
            &mut eval,
            "(let [t (take 5 (range 10)) r (rest t)] \
               [(class r) (class (rest t)) (class (rest r)) (chunked-seq? r) r])"
        ),
        "[clojure.lang.LazySeq clojure.lang.LazySeq clojure.lang.LazySeq false (1 2 3 4)]"
    );
    assert_eq!(
        run_last(
            &mut eval,
            "(let [i (interpose 0 [1 2 3])] \
               [(class (rest i)) (class (next (seq i))) (chunked-seq? (next i)) i])"
        ),
        "[clojure.lang.LazySeq clojure.lang.LazySeq false (1 0 2 0 3)]"
    );
    assert_eq!(
        run_last(
            &mut eval,
            "(let [w (last (partition 3 3 [:p :q] [1 2 3 4]))] [(class (rest w)) w])"
        ),
        "[clojure.lang.LazySeq (4 :p :q)]"
    );
    // One rest cell per item (the JVM's conses), forced or not; a
    // chunked seq's rests stay fresh `ChunkedCons` cells.
    assert_eq!(
        run_last(
            &mut eval,
            "(let [t (take 5 (range 10)) u (take 5 (range 10)) i (interpose 0 [1 2]) m (map inc [1 2 3])] \
               (seq u) \
               [(identical? (rest t) (rest t)) (identical? (next u) (next u)) \
                (identical? (rest (rest t)) (next (next t))) (identical? (rest i) (rest i)) \
                (identical? (rest m) (rest m)) (vec (rest (rest t)))])"
        ),
        "[true true true true false [2 3 4]]"
    );
    assert_eq!(
        uncaught(&mut eval, "(+ 1 (rest (take 3 (range 5))))"),
        "java.lang.ClassCastException | class clojure.lang.LazySeq cannot be cast to class java.lang.Number (clojure.lang.LazySeq is in unnamed module of loader 'app'; java.lang.Number is in module java.base of loader 'bootstrap')"
    );
}
