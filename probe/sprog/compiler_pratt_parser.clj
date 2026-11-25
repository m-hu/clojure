(ns compiler.pratt-parser
  "Pratt (top-down operator precedence) parser and evaluator for integer
   arithmetic with plus, minus, times, divide, caret (right associative)
   and parentheses.

   Time: O(n) tokenizing plus O(n) parsing (each token consumed once);
   evaluation is O(n) except exponentiation which uses binary
   exponentiation in O(log exponent) multiplications (the original used
   repeated multiplication in O(exponent)). Space: O(n) tokens plus
   O(depth) recursion and tree. Linear parsing is asymptotically optimal."
  (:refer-clojure :exclude [parse-long])
  (:require [clojure.string :as str]))

(def digit-values
  "Map digit char -> numeric value."
  (zipmap "0123456789" (range 10)))

(def max-long
  "Largest signed 64-bit value; tokens beyond long range are invalid."
  9223372036854775807)

(def min-long
  "Smallest signed 64-bit value."
  -9223372036854775808)

(defn fit-long
  "Returns n when it fits in a signed 64-bit long, else nil."
  [n]
  (when (<= min-long n max-long) n))


(def whitespace-chars
  "ASCII whitespace accepted between tokens."
  #{\space \tab \newline \return \formfeed \u000B})

(defn parse-long
  "Parse optional sign + digits into an integer (auto-promoting). Returns nil
   when `s` is not a valid integer literal."
  [s]
  (when (and (string? s) (seq s))
    (let [cs (seq s)
          [neg? rest-cs] (cond
                           (= (first cs) \-) [true (next cs)]
                           (= (first cs) \+) [false (next cs)]
                           :else [false cs])]
      (when (seq rest-cs)
        (loop [cs rest-cs
               acc 0]
          (if (nil? cs)
            (fit-long (if neg? (-' acc) acc))
            (let [d (get digit-values (first cs))]
              (when-not (nil? d)
                (recur (next cs) (+' (*' acc 10) d))))))))))

(defn tokenize
  "Pure scan of expression string `s` into [type value] tokens. Unknown
   characters are skipped, matching the original."
  [s]
  (let [chars (vec s)
        n (count chars)]
    (loop [i 0
           tokens []]
      (if (>= i n)
        tokens
        (let [c (nth chars i)]
          (cond
            (contains? whitespace-chars c)
            (recur (inc i) tokens)

            (contains? digit-values c)
            (let [[j val] (loop [k i
                                 acc 0]
                            (if (and (< k n) (contains? digit-values (nth chars k)))
                              (recur (inc k) (+' (*' acc 10) (get digit-values (nth chars k))))
                              [k acc]))]
              (recur j (conj tokens [:num val])))

            (= c \+) (recur (inc i) (conj tokens [:+ nil]))
            (= c \-) (recur (inc i) (conj tokens [:- nil]))
            (= c \*) (recur (inc i) (conj tokens [:* nil]))
            (= c \/) (recur (inc i) (conj tokens [:/ nil]))
            (= c \^) (recur (inc i) (conj tokens [:pow nil]))
            (= c \() (recur (inc i) (conj tokens [:lparen nil]))
            (= c \)) (recur (inc i) (conj tokens [:rparen nil]))
            :else (recur (inc i) tokens)))))))

(defn infix-binding-power
  "Return [left-bp right-bp] for an infix operator type."
  [token-type]
  (case token-type
    (:+ :-) [10 11]
    (:* :/) [20 21]
    :pow [31 30]
    [-1 -1]))

(defn to-sexpr
  "Render AST node as an s-expression string."
  [node]
  (case (first node)
    :num (str (second node))
    :unary (let [op (nth node 1)]
             (str "(" (if (= op :pow) "^" (name op)) " " (to-sexpr (nth node 2)) ")"))
    :binary (let [op (nth node 1)]
              (str "(" (if (= op :pow) "^" (name op)) " "
                   (to-sexpr (nth node 2)) " " (to-sexpr (nth node 3)) ")"))))

(defn fast-pow
  "Binary exponentiation matching HEAD semantics: non-positive exponents
   yield 1 (HEAD's (repeat r l) is empty then). O(log r) multiplications."
  [base exp]
  (if (<= exp 0)
    1
    (loop [b base
           e exp
           acc 1]
      (if (zero? e)
        acc
        (recur (*' b b) (quot e 2) (if (odd? e) (*' acc b) acc))))))

(defn eval-ast
  "Evaluate AST node to an integer. Division by zero yields 0 (HEAD)."
  [node]
  (case (first node)
    :num (second node)
    :unary (let [op (nth node 1)
                 child (eval-ast (nth node 2))]
             (if (= op :-) (-' child) child))
    :binary (let [op (nth node 1)
                  l (eval-ast (nth node 2))
                  r (eval-ast (nth node 3))]
              (case op
                :+ (+' l r)
                :- (-' l r)
                :* (*' l r)
                :/ (if (zero? r) 0 (quot l r))
                :pow (fast-pow l r)))))

(defn parse-expr
  "Parse tokens at `pos` with minimum binding power `min-bp`.
   Returns [next-pos ast] or nil on failure."
  [tokens pos min-bp]
  (let [token (get tokens pos)]
    (when token
      (let [[tt val] token
            prefix-res
            (case tt
              :num [(inc pos) [:num val]]
              :- (let [rhs-res (parse-expr tokens (inc pos) 40)]
                   (when rhs-res
                     (let [[new-pos rhs] rhs-res]
                       [new-pos [:unary :- rhs]])))
              :lparen (let [inner-res (parse-expr tokens (inc pos) 0)]
                        (when inner-res
                          (let [[new-pos inner] inner-res]
                            (if (= (first (get tokens new-pos)) :rparen)
                              [(inc new-pos) inner]
                              [new-pos inner]))))
              nil)]
        (when prefix-res
          (loop [cur-pos (first prefix-res)
                 lhs (second prefix-res)]
            (let [next-token (get tokens cur-pos)]
              (if-not next-token
                [cur-pos lhs]
                (let [op-type (first next-token)
                      [l-bp r-bp] (infix-binding-power op-type)]
                  (if (< l-bp min-bp)
                    [cur-pos lhs]
                    (let [rhs-res (parse-expr tokens (inc cur-pos) r-bp)]
                      (if-not rhs-res
                        nil
                        (let [[new-pos rhs] rhs-res]
                          (recur new-pos [:binary op-type lhs rhs]))))))))))))))

(defn error
  "Print message to stderr and return nil."
  [msg]
  (binding [*out* *err*]
    (println msg))
  nil)

(defn -main
  "Entry point. Blank stdin produces no output. Expects a count line then
   that many expression lines."
  [& _args]
  (let [s (slurp *in*)]
    (when-not (str/blank? s)
      (let [lines (vec (remove str/blank? (str/split-lines s)))]
        (when (seq lines)
          (let [num-exprs (parse-long (str/trim (first lines)))]
            (if (nil? num-exprs)
              (error "invalid input: expected expression count on first line")
              (let [expr-lines (take num-exprs (rest lines))]
                (doseq [line expr-lines]
                  (let [tokens (tokenize line)
                        parsed (parse-expr tokens 0 0)]
                    (if parsed
                      (let [ast (second parsed)]
                        (println (str (to-sexpr ast) " = " (eval-ast ast))))
                      (println "PARSE_ERROR"))))))))))))

(-main)
