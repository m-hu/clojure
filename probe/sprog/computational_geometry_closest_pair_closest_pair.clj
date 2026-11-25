;; Closest pair of points by divide and conquer.
;;
;; Time: O(n log n) worst case -- one sort by x and one by y, then a
;;   linear split/merge per level; the strip scan checks at most 15
;;   successors per point (packing argument), so every level is O(n).
;;   The old code re-sorted the strip on every level (O(n log n) per
;;   level, O(n log squared n) total).
;; Space: O(n).
;; Purity: core Clojure only, no host interop. Distances are compared
;;   as exact squared distances (one square root at the very end);
;;   parsing and six-decimal rendering are hand-written.
;; Robustness: blank stdin prints nothing; malformed input prints a
;;   message to stderr and returns normally (exit 0 -- a nonzero exit
;;   needs host interop, which this port forbids).
(ns computational-geometry.closest-pair.closest-pair
  (:require [clojure.string :as str]))

(def digit-values
  "Digit character to numeric value."
  (zipmap "0123456789" (range)))

(defn whitespace?
  "True for the ASCII whitespace that separates input tokens."
  [ch]
  (or (= ch \space) (= ch \tab) (= ch \newline) (= ch \return)
      (= ch \formfeed) (= ch (char 11))))

(defn tokenize
  "Split s on ASCII whitespace."
  [s]
  (if (nil? s)
    []
    (filterv seq (clojure.string/split s #"[ \t\n\r\f\x0B]+"))))

(defn parse-long-token
  "Parse an integer token; nil when malformed or outside int64 range."
  [tok]
  (let [cs (vec tok)
        n (count cs)]
    (when (pos? n)
      (let [c0 (nth cs 0)
            neg (or (= c0 \-) (= c0 \+))
            want-neg (= c0 \-)
            digits (subvec cs (if neg 1 0))]
        (when (and (seq digits) (every? digit-values digits))
          (let [stripped (vec (drop-while #(= % \0) digits))
                len (count stripped)
                lim (if want-neg "9223372036854775808" "9223372036854775807")]
            (when (or (< len 19)
                      (and (= len 19) (<= (compare (apply str stripped) lim) 0)))
              (let [v (reduce (fn [acc ch] (- (* acc 10) (digit-values ch)))
                              0 stripped)]
                (if want-neg v (- v))))))))))

(defn parse-double-token
  "Parse a decimal token; nil when malformed."
  [tok]
  (cond
    (= tok "NaN") (/ 0.0 0.0)
    (or (= tok "Infinity") (= tok "+Infinity")) (/ 1.0 0.0)
    (= tok "-Infinity") (/ -1.0 0.0)
    :else
    (let [cs (vec tok)
          n (count cs)]
      (when (pos? n)
        (let [c0 (nth cs 0)
              signed (or (= c0 \-) (= c0 \+))
              want-neg (= c0 \-)
              i0 (if signed 1 0)]
          (when (< i0 n)
            (let [epos (loop [i i0]
                         (cond (>= i n) -1
                               (let [c (nth cs i)]
                                 (or (= c \e) (= c \E))) i
                               :else (recur (inc i))))
                  mend (if (= epos -1) n epos)
                  mant (subvec cs i0 mend)
                  dots (count (filter #(= % \.) mant))
                  md (vec (remove #(= % \.) mant))]
              (when (and (<= dots 1)
                         (seq md)
                         (every? digit-values md))
                (let [dot-at (loop [i i0]
                               (cond (>= i mend) -1
                                     (= (nth cs i) \.) i
                                     :else (recur (inc i))))
                      intc (if (= dot-at -1)
                             (subvec cs i0 mend)
                             (subvec cs i0 dot-at))
                      fracc (if (= dot-at -1) [] (subvec cs (inc dot-at) mend))
                      expv (when (not (= epos -1))
                             (let [r (subvec cs (inc epos) n)
                                   rc (count r)]
                               (when (pos? rc)
                                 (let [s0 (nth r 0)
                                       es (or (= s0 \-) (= s0 \+))
                                       ed (subvec r (if es 1 0))]
                                   (when (and (seq ed) (every? digit-values ed))
                                     (if (> (count ed) 6)
                                       (if (= s0 \-) :neg-huge :pos-huge)
                                       (let [v (reduce (fn [a c]
                                                         (+ (* a 10) (digit-values c)))
                                                       0 ed)]
                                         (if (= s0 \-) (- v) v))))))))]
                  (when (or (= epos -1) (some? expv))
                    (let [alld (vec (concat intc fracc))
                          sig (vec (drop-while #(= % \0) alld))]
                      (if (empty? sig)
                        (if want-neg (* -1.0 0.0) 0.0)
                        (let [keep (take 15 sig)
                              lead (reduce (fn [a c]
                                             (+ (* a 10.0)
                                                (double (digit-values c))))
                                           0.0 keep)
                              ex (+ (if (number? expv) expv 0)
                                    (- (count fracc))
                                    (- (count sig) (count keep)))
                              ex (cond (= expv :pos-huge) 999
                                       (= expv :neg-huge) -999
                                       :else ex)
                              mag (cond (> ex 308) (/ 1.0 0.0)
                                        (< ex -324) 0.0
                                        :else (loop [i 0 v lead]
                                                (cond (= i ex) v
                                                      (< i ex) (recur (inc i) (* v 10.0))
                                                      :else (recur (dec i) (/ v 10.0)))))]
                          (if want-neg (- mag) mag))))))))))))))

(defn sqrt
  "Newton square root for a nonnegative finite double."
  [x]
  (cond
    (zero? x) x
    (not (== x x)) x
    (== x (* x 2.0)) x
    (neg? x) (/ 0.0 0.0)
    :else
    (let [[m adj] (loop [m x adj 1.0]
                    (cond (> m 4.0) (recur (/ m 4.0) (* adj 2.0))
                          (< m 0.25) (recur (* m 4.0) (/ adj 2.0))
                          :else [m adj]))
          g (loop [i 0 g (if (> m 1.0) m 1.0)]
              (if (= i 30) g (recur (inc i) (/ (+ g (/ m g)) 2.0))))]
      (* g adj))))

(defn double-to-bigint
  "Exact integer value of an integral double beyond long range."
  [v]
  (loop [q v k 0]
    (if (< q 9007199254740992.0)
      (loop [i 0 acc (bigint (long q))]
        (if (= i k) acc (recur (inc i) (*' acc 2))))
      (recur (/ q 2.0) (inc k)))))

(defn format-fixed
  "Render x with exactly k decimals, halves up."
  [x k]
  (cond
    (not (== x x)) "NaN"
    (and (not (zero? x)) (== x (* x 2.0)))
    (if (neg? x) "-Infinity" "Infinity")
    :else
    (let [neg (neg? (/ 1.0 x))
          a (if neg (- x) x)]
      (if (>= a 9.0e18)
        (str (when neg "-") (str (double-to-bigint a)) "."
             (apply str (repeat k \0)))
        (let [pow10k (loop [i 0 v 1.0]
                       (if (= i k) v (recur (inc i) (* v 10.0))))
              ip (long a)
              frac (- a (double ip))
              r (long (+ (* frac pow10k) 0.5))
              [ip r] (if (>= r (long pow10k)) [(inc ip) 0] [ip r])
              fs (str r)
              pad (subs "00000000000000000000" 0 (- k (count fs)))]
          (str (when neg "-") (str ip) "." pad fs))))))

(defn dsq
  "Squared distance between indexed points [x y i]."
  [[x1 y1 _] [x2 y2 _]]
  (let [dx (- (double x1) (double x2))
        dy (- (double y1) (double y2))]
    (+ (* dx dx) (* dy dy))))

(defn brute-dsq
  "Minimum squared distance among at most 3 points."
  [pts]
  (let [n (count pts)]
    (loop [i 0 best ##Inf]
      (if (>= i n)
        best
        (recur (inc i)
               (loop [j (inc i) best best]
                 (if (>= j n)
                   best
                   (recur (inc j) (min best (dsq (nth pts i) (nth pts j)))))))))))

(defn strip-scan
  "Sweep the y-ordered strip in linear time: past the current distance
  band every later point is farther away, and inside the band at most
  15 successors fit by the packing argument."
  [strip d]
  (let [n (count strip)]
    (loop [i 0 d d]
      (if (>= i n)
        d
        (let [pi (nth strip i)
              yi (double (nth pi 1))
              d (loop [j (inc i) d d]
                  (if (>= j n)
                    d
                    (let [pj (nth strip j)
                          dy (- (double (nth pj 1)) yi)]
                      (if (>= (* dy dy) d)
                        d
                        (recur (inc j) (min d (dsq pi pj)))))))]
          (recur (inc i) d))))))

(defn closest-dsq
  "Minimum squared distance. px is x-ordered, py holds the same points in
  y-order; both halves split in linear time, hence O(n log n)."
  [px py]
  (let [n (count px)]
    (if (<= n 3)
      (brute-dsq px)
      (let [mid (quot n 2)
            midx (double (nth (nth px mid) 0))
            in-left (into #{} (map peek (subvec px 0 mid)))
            ly (filterv #(contains? in-left (peek %)) py)
            ry (filterv #(not (contains? in-left (peek %))) py)
            d (min (closest-dsq (subvec px 0 mid) ly)
                   (closest-dsq (subvec px mid n) ry))
            strip (filterv (fn [p]
                             (let [dx (- (double (nth p 0)) midx)]
                               (< (* dx dx) d)))
                           py)]
        (strip-scan strip d)))))

(defn die
  "Report invalid input on stderr and return normally."
  [msg]
  (binding [*out* *err*]
    (println msg)))

(defn -main
  [& _args]
  (let [content (slurp *in*)]
    (when-not (str/blank? content)
      (let [tokens (tokenize content)
            n (parse-long-token (first tokens))]
        (cond
          (nil? n)
          (die "error: first token must be an integer point count")

          (or (> n 4611686018427387903) (< n -4611686018427387904))
          (die "error: point count out of range")

          :else
          (let [coords (take (* 2 n) (rest tokens))
                raw (vec (for [[xs ys] (partition 2 coords)]
                           [(parse-double-token xs) (parse-double-token ys)]))]
            (if (some (fn [[a b]] (or (nil? a) (nil? b))) raw)
              (die "error: expected numeric coordinate pairs after the count")
              (let [pts (vec (map-indexed (fn [i [x y]] [x y i]) raw))]
                (if (< (count pts) 2)
                  (println (format-fixed 1e18 6))
                  (let [px (vec (sort-by (fn [[x y i]] [x y i]) pts))
                        py (vec (sort-by (fn [[x y i]] [y x i]) pts))]
                    (println (format-fixed (sqrt (closest-dsq px py)) 6))))))))))))

(-main)
