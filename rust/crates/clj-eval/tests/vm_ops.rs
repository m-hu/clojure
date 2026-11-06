// Copyright (c) Hao Hu. All rights reserved.
// The use and distribution terms for this software are covered by the
// Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
// which can be found in the file epl-v10.html at the root of this distribution.
// By using this software in any fashion, you are agreeing to be bound by
// the terms of this license.

//! #226 shape-specialized VM ops (`vm_ops`): every op across operand
//! shapes (registers, captured values, literals, computed forms),
//! number kinds, and native fallbacks; n-ary folds and comparison
//! chains; update ops; the `for` keep stage; prepared callees and the
//! direct seq walks. Every row runs twice — compiled by the VM (a
//! top-level form) and on the tree evaluator (inside `(binding [] …)`,
//! which the VM hands to it whole) — and both must give the JVM's
//! value (Clojure 1.12) for [`JVM_ROWS`], the port's own semantics for
//! [`PORT_ROWS`] (direct calls see a redefined core fn; the JVM
//! compiler inlines those fns instead).

use clj_eval::{Evaluator, boot};
use clj_value::{PrintOpts, pr_to_string};

/// `pr-str` of the form's value, or `Class: message` when it throws.
fn show(eval: &mut Evaluator, form: &str) -> String {
    let src = format!(
        "(try (pr-str {form}) (catch Throwable t (str (.getName (class t)) \": \" (.getMessage t))))"
    );
    let vs = eval
        .eval_source(&src)
        .unwrap_or_else(|e| panic!("{form}: {e}"));
    pr_to_string(vs.last().unwrap(), &PrintOpts::print())
}

const JVM_ROWS: &[(&str, &str)] = &[
    (
        r#"((fn [a] [(try (+ (identity a) 1) (catch Exception e :xs)) (try (+ 1 (identity a)) (catch Exception e :sx)) (try (+ (identity a) (identity a)) (catch Exception e :xx)) (if (+) :t :f)]) "s")"#,
        r#"[:xs :sx :xx :t]"#,
    ),
    (
        r#"((fn [a b] [(+ a b) (- a b) (* a b) (quot a b) (rem a b) (mod a b) (mod (- a) b) (min a b) (max a b)]) 17 5)"#,
        r#"[22 12 85 3 2 2 3 5 17]"#,
    ),
    (
        r#"((fn [a b] [(< a b) (<= a b) (> a b) (>= a b) (== a b) (= a b) (not= a b)]) 3 3)"#,
        r#"[false true false true true true false]"#,
    ),
    (
        r#"((fn [a b] [(bit-and a b) (bit-or a b) (bit-xor a b) (bit-shift-left a b) (bit-shift-right a b) (unsigned-bit-shift-right (- a) b)]) 12 2)"#,
        r#"[0 14 14 48 3 4611686018427387901]"#,
    ),
    (
        r#"((fn [a] [(+ a 1) (+ 1 a) (- a 1) (- 10 a) (* a 2) (< a 10) (> 10 a) (inc a) (dec a) (- a)]) 7)"#,
        r#"[8 8 6 3 14 true true 8 6 -7]"#,
    ),
    (
        r#"(let [k 5] ((fn [a] [(+ a k) (+ k a) (* k k) (< k a) (inc k) (zero? k)]) 7))"#,
        r#"[12 12 25 true 6 false]"#,
    ),
    (
        r#"((fn [a] [(+ (inc a) 1) (+ 1 (inc a)) (+ (inc a) (dec a)) (* a (inc a)) (- (inc a)) (inc (inc a))]) 7)"#,
        r#"[9 9 14 56 -8 9]"#,
    ),
    (
        r#"((fn [a b] [(+ a b) (- a b) (* a b) (< a b) (> b a) (== a 2.0) (+ b a) (inc a) (dec a) (- a)]) 1.5 2)"#,
        r#"[3.5 -0.5 3.0 true true false 3.5 2.5 0.5 -1.5]"#,
    ),
    (
        r#"((fn [a b] [(+ a b) (< a b) (<= b a)]) 2 0.5)"#,
        r#"[2.5 false true]"#,
    ),
    (
        r#"((fn [a] [(try (+ a 1) (catch Exception e :err)) (try (< a 1) (catch Exception e :cerr)) (try (inc a) (catch Exception e :ierr))]) "x")"#,
        r#"[:err :cerr :ierr]"#,
    ),
    (
        r#"((fn [a] [(try (+ a 1) (catch ArithmeticException e :ovf)) (try (inc a) (catch ArithmeticException e :iovf)) (try (* a 2) (catch ArithmeticException e :movf))]) Long/MAX_VALUE)"#,
        r#"[:ovf :iovf :movf]"#,
    ),
    (
        r#"((fn [a] [(try (- a) (catch ArithmeticException e :novf)) (try (dec a) (catch ArithmeticException e :dovf))]) Long/MIN_VALUE)"#,
        r#"[:novf :dovf]"#,
    ),
    (
        r#"((fn [a b] [(try (quot a b) (catch ArithmeticException e :q0)) (try (rem a b) (catch ArithmeticException e :r0)) (try (mod a b) (catch ArithmeticException e :m0))]) 5 0)"#,
        r#"[:q0 :r0 :m0]"#,
    ),
    (
        r#"((fn [a b] [(quot a b) (rem a b) (mod a b) (bit-and 7 a)]) -7 2)"#,
        r#"[-3 -1 1 1]"#,
    ),
    (
        r#"((fn [a b] [(if (< a b) :lt :ge) (if (zero? a) :z :nz) (if (pos? b) :p :np) (if (neg? a) :n :nn) (if (even? b) :e :o) (if (odd? b) :o :e) (if (= a b) :eq :ne) (if (not= a b) :ne :eq) (if (== a b) :eq :ne) (if (nil? a) :nil :some) (if (some? a) :some :nil) (if (not a) :f :t)]) 1 2)"#,
        r#"[:lt :nz :p :nn :e :e :ne :ne :ne :some :some :t]"#,
    ),
    (
        r#"((fn [a b] [(if (< (inc a) b) :lt :ge) (if (> b (inc a)) :y :n) (if (< (inc a) (dec b)) :y :n) (if (zero? (dec a)) :z :nz) (if (contains? #{1 2} (inc a)) :in :out)]) 1 3)"#,
        r#"[:lt :y :n :z :in]"#,
    ),
    (
        r#"((fn [x] [(zero? x) (pos? x) (neg? x) (zero? 0.0) (pos? -1.5) (neg? -1.5)]) 0)"#,
        r#"[true false false true false true]"#,
    ),
    (
        r#"((fn [x] [(try (even? x) (catch Exception e :ee)) (try (zero? x) (catch Exception e :ze)) (not x) (nil? x) (some? x)]) 1.5)"#,
        r#"[:ee false false false true]"#,
    ),
    (
        r#"((fn [v m s] [(nth v 1) (nth "abc" 2) (get m :a) (get v 0) (get nil :a) (get m :zz) (contains? m :a) (contains? s 3) (contains? s 9) (count v) (count m) (count s) (count "héllo") (count nil)]) [10 20 30] {:a 1} #{3 4})"#,
        r#"[20 \c 1 10 nil nil true true false 3 1 2 5 0]"#,
    ),
    (
        r#"((fn [v] [(try (nth v 9) (catch IndexOutOfBoundsException e :oob)) (nth v -1 :nf) (nth v 9 :nf) (try (nth "ab" 5) (catch Exception e :soob)) (get v 9) (get (sorted-map :a 1) :a) (contains? (sorted-set 1) 1) (count '(1 2)) (nth '(5 6) 1)]) [1 2])"#,
        r#"[:oob :nf :nf :soob nil 1 true 2 6]"#,
    ),
    (
        r#"((fn [t] (let [t (conj! t 1) t (conj! t 2)] (persistent! t))) (transient []))"#,
        r#"[1 2]"#,
    ),
    (
        r#"((fn [t] (persistent! (conj! t [:a 1]))) (transient {}))"#,
        r#"{:a 1}"#,
    ),
    (
        r#"((fn [x] [(int x) (long x) (double x) (int \a) (long \b) (double 2.5)]) 7)"#,
        r#"[7 7 7.0 97 98 2.5]"#,
    ),
    (
        r#"((fn [x] [(try (int x) (catch Exception e :ibig)) (long 2.9) (int 2.9) (try (double nil) (catch Exception e :dnil))]) 3000000000)"#,
        r#"[:ibig 2 2 :dnil]"#,
    ),
    (
        r#"((fn [a b c d] [(+ a b c) (+ a b c d) (- a b c) (* a b c) (min a b c) (max a b c d) (bit-and a b c) (bit-or a b c) (bit-xor a b c)]) 12 10 6 3)"#,
        r#"[28 31 -4 720 6 12 0 14 0]"#,
    ),
    (
        r#"((fn [a b c] [(< a b c) (< a c b) (<= a a b) (> c b a) (>= c c a) (== a a a) (< 1 0 :x)]) 1 2 3)"#,
        r#"[true false true true true true false]"#,
    ),
    (
        r#"((fn [a b c] [(if (< a b c) :in :out) (if (<= 0 a 10) :in :out) (+ a 1.5 c) (try (+ a b :x) (catch Exception e :cce)) (try (< a :x 0) (catch Exception e :ccc))]) 1 2 3)"#,
        r#"[:in :in 5.5 :cce :ccc]"#,
    ),
    (
        r#"((fn [a b] [(+ a b (inc a)) (+ (inc a) b 1) (* 2 a b)]) 4 5)"#,
        r#"[14 11 40]"#,
    ),
    (
        r#"((fn [a] (try (+ a 1 1) (catch ArithmeticException e :novf))) Long/MAX_VALUE)"#,
        r#":novf"#,
    ),
    (
        r#"((fn [v] (let [v (assoc v 0 :x) v (assoc v 3 :y)] [v (try (assoc v 9 :z) (catch IndexOutOfBoundsException e :oob))])) [1 2 3])"#,
        r#"[[:x 2 3 :y] :oob]"#,
    ),
    (
        r#"((fn [m] [(assoc m :b 2) (assoc m [1] 3) (assoc (sorted-map :b 1) :a 2) (assoc nil :k 1)]) {:a 1})"#,
        r#"[{:a 1, :b 2} {:a 1, [1] 3} {:a 2, :b 1} {:k 1}]"#,
    ),
    (
        r#"((fn [v] (let [w (assoc v 0 :changed)] [v w])) [1 2])"#,
        r#"[[1 2] [:changed 2]]"#,
    ),
    (
        r#"((fn [] (let [v [1 2 3]] [(assoc v 1 (inc (nth v 1))) v])))"#,
        r#"[[1 3 3] [1 2 3]]"#,
    ),
    (
        r#"((fn [x] [(conj [1] x) (conj '(1) x) (conj nil x) (conj #{1} x) (contains? (conj #{[1]} [x]) [2]) (conj {:a 1} [:b x]) (conj (sorted-set 3) x)]) 2)"#,
        r#"[[1 2] (2 1) (2) #{1 2} true {:a 1, :b 2} #{2 3}]"#,
    ),
    (
        r#"((fn [m] (reduce (fn [acc k] (assoc acc k (inc (get acc k 0)))) m [:a :b :a])) {})"#,
        r#"{:a 2, :b 1}"#,
    ),
    (
        r#"(let [k 3] ((fn [v] [(assoc v 0 k) (conj v k)]) [9]))"#,
        r#"[[3] [9 3]]"#,
    ),
    (
        r#"((fn [] [(conj (vec (range 3)) 9) (assoc (vec (range 40)) 39 :end)]))"#,
        r#"[[0 1 2 9] [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 :end]]"#,
    ),
    (r#"(letfn [(f [x] (if (nil? f) 0 x))] (f 5))"#, r#"5"#),
    (
        r#"((fn [m] (if (get m :a nil) :hit :miss)) {:a 1})"#,
        r#":hit"#,
    ),
    (
        r#"((fn [v] [(nth v 1 :nf) (get v 5 :dflt) (get {:a 1} :b :nf)]) [1 2])"#,
        r#"[2 :dflt :nf]"#,
    ),
    (
        r#"((fn [t] (persistent! (assoc! t 0 :z))) (transient [1 2]))"#,
        r#"[:z 2]"#,
    ),
    (r#"(if (let [x 1] (if x 2 3)) :a :b)"#, r#":a"#),
    (
        r#"(let [and1 5] (if (let [and1 and1] (if and1 (= and1 5) and1)) :y :n))"#,
        r#":y"#,
    ),
    (
        r#"[(if (and (pos? 1) (odd? 3) (< 1 2)) :all :not) (if (or (neg? 1) (zero? 0)) :any :none) (if (and (pos? 1) (neg? 1)) :all :not) (if (or (neg? 1) (neg? 2)) :any :none)]"#,
        r#"[:all :any :not :none]"#,
    ),
    (
        r#"(do (defn io-rec [n] (if (pos? n) (io! (recur (dec n))) :done)) (io-rec 3))"#,
        r#":done"#,
    ),
    (
        r#"[(mapv inc [1 2]) (reduce + [1.5 2]) (reduce conj [] [1 2]) (reduce (fn [& xs] (apply + xs)) [1 2 3]) (reduce + (repeat 3 2)) (reduce + (range 4)) (reduce + (range 0.5 3)) (reduce (fn [a x] (if (> a 5) (reduced a) (+ a x))) 0 (repeat 1))]"#,
        r#"[[2 3] 3.5 [1 2] 6 6 6 4.5 6]"#,
    ),
    (
        r#"[(vec (repeat 3 0)) (vec (repeat 2 :a)) (vec (range 5)) (vec (range 0.5 2)) (vec (range 2 11 3)) (keep :a [{:a 1} {}]) (into [] (map inc) (range 3))]"#,
        r#"[[0 0 0] [:a :a] [0 1 2 3 4] [0.5 1.5] [2 5 8] (1) [1 2 3]]"#,
    ),
    (
        r#"(let [s (apply str (repeat 300 "é"))] [(count s) (count s) (nth s 5) (nth s 299)])"#,
        r#"[300 300 \é \é]"#,
    ),
    (
        r#"(let [s (apply str (repeat 100 "ab")) t (apply str (repeat 100 "cd"))] (loop [i 0 acc []] (if (< i 4) (recur (inc i) (conj acc (nth s i) (nth t i))) acc)))"#,
        r#"[\a \c \b \d \a \c \b \d]"#,
    ),
    (
        r#"[(meta ^{:m 1} [1 2]) (for [x (range 10) :when (odd? x)] (when (> x 4) x)) (for [x [1 2 3] :let [y (* x x)] :when (odd? y)] [x y]) (take 3 (for [x (range) :when (zero? (mod x 7))] x)) (for [x [1 2] y [3 4] :when (not= x 4)] [x y]) (for [[a b] [[1 2] [3 4]] :when (odd? a)] (+ a b))]"#,
        r#"[{:m 1} (nil nil 5 7 9) ([1 1] [3 9]) (0 7 14) ([1 3] [1 4] [2 3] [2 4]) (3 7)]"#,
    ),
    (
        r#"((fn [a b c d e] [(loop [i 0 a a b b c c] (if (< i 3) (recur (inc i) (+ a 1) (+ b 2) (+ c 3)) [a b c])) (loop [i 0 a a b b c c d d] (if (< i 2) (recur (inc i) (inc a) (inc b) (inc c) (inc d)) [a b c d])) (loop [i 0 a a b b c c d d e e] (if (< i 2) (recur (inc i) a b c d (inc e)) e))]) 1 2 3 4 5)"#,
        r#"[[4 8 12] [3 4 5 6] 7]"#,
    ),
    // Seq ops (#226): `first`/`empty?`/`seq` over each shape (and in
    // test position), `next`/`rest` over owned operands (a last-use
    // seq steps in place, a shared one gets a fresh tail; metadata
    // stays on string seqs only), O(1) vector/string seqs and their
    // 32-item chunks.
    (
        r#"((fn [l v s e] [(first nil) (first l) (first ()) (first v) (first []) (first s) (first "") (first (seq v)) (first (rest v)) (first e) (if (first v) :y :n) (if (first [false]) :y :n)]) '(1 2) [3 4] "ab" (map inc [1]))"#,
        r#"[nil 1 nil 3 nil \a nil 3 4 2 :y :n]"#,
    ),
    (
        r#"((fn [e] [(empty? nil) (empty? '(1)) (empty? ()) (empty? [1]) (empty? []) (empty? {:a 1}) (empty? {}) (empty? #{1}) (empty? #{}) (empty? "a") (empty? "") (empty? (seq [1])) (empty? (rest [1])) (empty? e) (if (empty? []) :y :n) (if (empty? "x") :y :n)]) (map inc [1]))"#,
        r#"[true false true false true false true false true false true false true false :y :n]"#,
    ),
    (
        r#"((fn [e] [(seq nil) (seq ()) (seq '(1)) (seq []) (seq [1 2]) (seq "") (seq "ab") (seq (seq [1])) (seq (rest [1 2])) (seq {}) (seq e) (if (seq []) :y :n) (if (seq "a") :y :n) (if (seq (rest [1])) :y :n) (if (seq {:a 1}) :y :n)]) (map inc [1]))"#,
        r#"[nil nil (1) nil (1 2) nil (\a \b) (1) (2) nil (2) :n :y :n :y]"#,
    ),
    (
        r#"((fn [e m] [(next nil) (next ()) (next '(1)) (next '(1 2)) (next []) (next [1]) (next [1 2 3]) (next "") (next "a") (next "ab") (next "aé") (next e) (next m) (rest nil) (rest ()) (rest '(1)) (rest [1]) (rest "a") (rest "abc") (rest e)]) (map inc [1 2]) {:a 1})"#,
        r#"[nil nil nil (2) nil nil (2 3) nil nil (\b) (\é) (3) nil () () () () () (\b \c) (3)]"#,
    ),
    (
        r#"(loop [s (seq [1 2 3]) n 0] (if s (recur (next s) (inc n)) n))"#,
        r#"3"#,
    ),
    (
        r#"(loop [s (seq "aéz") acc []] (if s (let [c (first s)] (recur (next s) (conj acc c))) acc))"#,
        r#"[\a \é \z]"#,
    ),
    (
        r#"(loop [s (seq [1 2 3]) acc []] (if (seq s) (let [x (first s)] (recur (rest s) (conj acc x))) acc))"#,
        r#"[1 2 3]"#,
    ),
    (
        r#"(let [s (seq [1 2 3]) t (next s)] [s t (next t) (next (next t)) (rest (rest t)) (rest (rest (rest t)))])"#,
        r#"[(1 2 3) (2 3) (3) nil () ()]"#,
    ),
    (
        r#"(let [s (seq "xyz") t (rest s)] [s t (rest t) (next (rest t)) (rest (rest (rest t)))])"#,
        r#"[(\x \y \z) (\y \z) (\z) nil ()]"#,
    ),
    (
        r#"[(let [s (with-meta (seq "abc") {:a 1})] (meta (next s))) (let [s (with-meta (seq [1 2 3]) {:a 1})] (meta (next s))) (let [s (with-meta (seq "abc") {:a 1})] [(meta (rest s)) (meta s)])]"#,
        r#"[{:a 1} nil [{:a 1} {:a 1}]]"#,
    ),
    (
        r#"(let [s (with-meta (seq [1 2 3]) {:a 1}) t (rest s)] [(meta s) (meta t)])"#,
        r#"[{:a 1} nil]"#,
    ),
    (
        r#"[(count (chunk-first (seq (vec (range 40))))) (count (chunk-first (rest (seq (vec (range 40)))))) (count (chunk-first (seq (vec (map str (range 40)))))) (first (chunk-rest (seq (vec (range 40))))) (count (chunk-first (rest (seq (vec (map str (range 40)))))))]"#,
        r#"[32 31 32 32 31]"#,
    ),
    (
        r#"[(count (map inc (vec (range 70)))) (last (map str (vec (map str (range 70))))) (apply str (map identity "the quick brown fox jumps over the lazy dog!")) (count (seq "aéz")) (count (rest (seq "abc"))) (count (seq [1 2 3])) (count (rest (seq [1 2 3])))]"#,
        r#"[70 "69" "the quick brown fox jumps over the lazy dog!" 3 2 3 2]"#,
    ),
    (
        r#"(let [v (vec (range 3))] (loop [s v acc 0] (if (empty? s) acc (recur (rest s) (+ acc (first s))))))"#,
        r#"3"#,
    ),
    (
        r#"(let [w "hello"] (loop [cs (seq w) n 0] (if (nil? cs) n (recur (next cs) (inc n)))))"#,
        r#"5"#,
    ),
    (
        r#"[(chunked-seq? (seq [1 2])) (chunked-seq? (seq "ab")) (chunked-seq? (rest (seq [1 2]))) (class (seq "ab")) (class (next (seq "ab"))) (class (seq [1])) (class (rest (seq [1 2])))]"#,
        r#"[true false true clojure.lang.StringSeq clojure.lang.StringSeq clojure.lang.PersistentVector$ChunkedSeq clojure.lang.PersistentVector$ChunkedSeq]"#,
    ),
    // `(apply str coll)` streams a realized coll into one string (lazy
    // colls spread first); packed char vectors (#226 memory).
    (
        r#"[(apply str nil) (apply str [\a \b]) (apply str '(1 nil "x")) (apply str "abc") (apply str (map inc [1 2])) (apply str 1 2 [3]) (apply str 1 [\a nil]) (apply + 1 2 [3 4]) (apply max [3 1 2]) (apply str (seq [1 2])) (apply str (repeat 3 \q)) (apply str [])]"#,
        r#"["" "ab" "1x" "abc" "23" "123" "1a" 10 3 "12" "qqq" ""]"#,
    ),
    (
        r#"[(count (vec (repeat 40 \z))) (vec (repeat 3 \z)) (str \a "b" nil 1 :k) (apply str [[1 2] {:a 1} 'sym \c]) (apply vector 1 [2 3]) (apply list [])]"#,
        r#"[40 [\z \z \z] "ab1:k" "[1 2]{:a 1}symc" [1 2 3] ()]"#,
    ),
    (
        r#"(let [cur (transient [])] (dotimes [i 5] (conj! cur (char (+ 97 i)))) (apply str (persistent! cur)))"#,
        r#""abcde""#,
    ),
    (
        r#"[(conj (vec "abc") \d) (conj (vec "ab") 1) (assoc (vec "ab") 0 \z) (assoc (vec "ab") 1 :x) (pop (vec "ab")) (conj (pop [\a]) 1) (= (vec "ab") [\a \b]) (hash (vec "ab"))]"#,
        r#"[[\a \b \c \d] [\a \b 1] [\z \b] [\a :x] [\a] [1] true -338459411]"#,
    ),
    // Maps and keywords called as fns answer inline as the `get` they
    // are; the promoting ops share `+`/`-`/`*` until a long overflows;
    // `str` of a long writes its digits.
    (
        r#"((fn [m k s] [(m :a) (m :z) (m :z 9) (m :a 9) (:a m) (:z m 7) (:a nil) (:a nil 5) (:a [1 2]) (m k) (k {:b 2}) (s 2) (s 9) (try (m) (catch Exception e :arity))]) {:a 1} :b #{2})"#,
        r#"[1 nil 9 1 1 7 nil 5 nil nil 2 2 nil :arity]"#,
    ),
    (
        r#"[(+' 1 2) (+' Long/MAX_VALUE 1) (-' Long/MIN_VALUE 1) (*' 3037000500 3037000500) (inc' Long/MAX_VALUE) (dec' Long/MIN_VALUE) (-' Long/MIN_VALUE) (-' 5) (+' 1 2 3) (+' Long/MAX_VALUE 1 1) (*' 2 3 4) (-' 10 1 2) (+' 1.5 2) (*' 1/2 4) (-' 2.5)]"#,
        r#"[3 9223372036854775808N -9223372036854775809N 9223372037000250000N 9223372036854775808N -9223372036854775809N 9223372036854775808N -5 6 9223372036854775809N 24 7 3.5 2N -2.5]"#,
    ),
    (
        r#"[(str 1) (str -42 " " 0) (str Long/MIN_VALUE) (apply str (range 5)) (str 1 \a nil "b" 2.5 :k)]"#,
        r#"["1" "-42 0" "-9223372036854775808" "01234" "1ab2.5:k"]"#,
    ),
    // Destructuring: plain map patterns on unsorted maps and positional
    // patterns on vectors/map entries bind inline; every other shape
    // (nil, sorted maps, seqs, strings, `:or`, `&`, nested) generally.
    (
        r#"((fn [m] (let [{a :a b "b" :keys [c d] :strs [e] :syms [g] :as all} m] [a b c d e g (count all)])) {:a 1 "b" 2 :c 3 "e" 5 'g 7})"#,
        r#"[1 2 3 nil 5 7 5]"#,
    ),
    (
        r#"((fn [x] (let [{a :a :keys [b]} x] [a b])) nil)"#,
        r#"[nil nil]"#,
    ),
    (
        r#"((fn [x] (let [{a :a :as all} x] [a all])) (sorted-map :a 1 :b 2))"#,
        r#"[1 {:a 1, :b 2}]"#,
    ),
    (r#"((fn [x] (let [{a :a} x] a)) '(:a 9))"#, r#"9"#),
    (r#"((fn [x] (let [{a :a :or {a 5}} x] a)) {})"#, r#"5"#),
    (
        r#"((fn [x] (let [{[p q] :pt} x] [p q])) {:pt [1 2]})"#,
        r#"[1 2]"#,
    ),
    (
        r#"((fn [x] (let [{a [1 2]} x] a)) {[1 2] :vec-key})"#,
        r#":vec-key"#,
    ),
    (
        r#"((fn [v] (let [[a b c :as all] v] [a b c all])) [1 2])"#,
        r#"[1 2 nil [1 2]]"#,
    ),
    (r#"((fn [v] (let [[a b] v] [a b])) "xy")"#, r#"[\x \y]"#),
    (r#"((fn [v] (let [[a b] v] [a b])) nil)"#, r#"[nil nil]"#),
    (r#"((fn [v] (let [[a b] v] [a b])) '(1 2 3))"#, r#"[1 2]"#),
    (
        r#"((fn [v] (let [[a & more] v] [a more])) [1 2 3])"#,
        r#"[1 (2 3)]"#,
    ),
    (r#"((fn [v] (let [[[a] b] v] [a b])) [[1] 2])"#, r#"[1 2]"#),
    (
        r#"(vec (for [[k v] {:a 1 :b 2}] [v k]))"#,
        r#"[[1 :a] [2 :b]]"#,
    ),
    (
        r#"((fn [e] (let [[k v x :as all] e] [k v x all])) (first {:a 1}))"#,
        r#"[:a 1 nil [:a 1]]"#,
    ),
    (
        r#"(let [f (fn [{:keys [a]} [b c]] [a b c])] (f {:a 1} [2 3]))"#,
        r#"[1 2 3]"#,
    ),
    (
        r#"(reduce (fn [{:keys [n s]} x] {:n (inc n) :s (+ s x)}) {:n 0 :s 0} [1 2 3])"#,
        r#"{:n 3, :s 6}"#,
    ),
];

const PORT_ROWS: &[(&str, &str)] = &[
    (
        r#"(do (defn r-xs [a] [(+ (identity a) 1) (+ 1 (identity a)) (+ (identity a) (identity a))]) (with-redefs [+ *] (r-xs 3)))"#,
        r#"[3 3 9]"#,
    ),
    (
        r#"(do (defn r-add [a b] (+ a b)) (with-redefs [+ *] (r-add 3 4)))"#,
        r#"12"#,
    ),
    (
        r#"(do (defn r-inc [a] (inc (inc a))) (with-redefs [inc dec] (r-inc 3)))"#,
        r#"1"#,
    ),
    (
        r#"(do (defn r-add3 [a b c] (+ a b c)) (with-redefs [+ (fn [& xs] (count xs))] (r-add3 1 2 3)))"#,
        r#"3"#,
    ),
    (
        r#"(do (defn r-asc [m] (assoc m :a 1)) (with-redefs [assoc (fn [m k v] :redef)] (r-asc {})))"#,
        r#":redef"#,
    ),
    (
        r#"(do (defn r-lt [a b] (if (< a b) :lt :ge)) (with-redefs [< >] (r-lt 1 2)))"#,
        r#":ge"#,
    ),
    (
        r#"(do (defn r-next [s] (next s)) (with-redefs [next (fn [s] [:redef s])] (r-next [1 2])))"#,
        r#"[:redef [1 2]]"#,
    ),
    (
        r#"(do (defn r-rest [s] [(rest s) (first s)]) (with-redefs [rest (fn [s] :redef)] (r-rest [1 2])))"#,
        r#"[:redef 1]"#,
    ),
];

fn check(rows: &[(&str, &str)]) {
    for (form, want) in rows {
        let mut eval = boot();
        assert_eq!(show(&mut eval, form), *want, "VM: {form}");
        let mut eval = boot();
        let tree = format!("(binding [] {form})");
        assert_eq!(show(&mut eval, &tree), *want, "tree: {form}");
    }
}

#[test]
fn vm_ops_match_the_jvm() {
    check(JVM_ROWS);
}

#[test]
fn vm_ops_share_port_semantics() {
    check(PORT_ROWS);
}
