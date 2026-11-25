;; Segment intersections by the Bentley-Ottmann sweep.
;;
;; Time: O((n + k) log n) for k reported crossings -- one event-queue
;;   and one sweep-status operation per event, each logarithmic; the old
;;   code tested every pair (O(n squared)). Verticals are crossed with
;;   the status through range queries, each visit paid for by a reported
;;   point, so the bound holds with them too.
;; Space: O(n + k).
;; Purity: core Clojure only, no host interop. The event queue is a
;;   sorted map and the sweep status a sorted set; the status comparator
;;   reads the current sweep position and exact event heights from two
;;   volatile cells (plain core values, moved only between event
;;   batches). Segments touching the next batch leave the status before
;;   the position moves, and re-enter ranked at their exact event
;;   height, so every comparison stays consistent. Parsing and
;;   two-decimal rendering are hand-written.
;; Robustness: blank stdin prints nothing; malformed input prints a
;;   message to stderr and returns normally (exit 0 -- a nonzero exit
;;   needs host interop, which this port forbids).
(ns computational-geometry.sweep-line.bentley-ottmann
  (:require [clojure.string :as str]))

(def EPS 1e-9)

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

(defn abs-num
  "Absolute value."
  [x]
  (if (neg? x) (- x) x))

(defn intersect
  "Crossing of two segments, or nil when parallel or disjoint."
  [[[x1 y1] [x2 y2]] [[x3 y3] [x4 y4]]]
  (let [x1 (double x1) y1 (double y1)
        x2 (double x2) y2 (double y2)
        x3 (double x3) y3 (double y3)
        x4 (double x4) y4 (double y4)
        denom (- (* (- x1 x2) (- y3 y4)) (* (- y1 y2) (- x3 x4)))]
    (when (> (abs-num denom) EPS)
      (let [t (/ (- (* (- x1 x3) (- y3 y4)) (* (- y1 y3) (- x3 x4))) denom)
            u (/ (- (* (- x1 x2) (- y1 y3)) (* (- y1 y2) (- x1 x3))) (- denom))]
        (when (and (>= t (- EPS)) (<= t (+ 1.0 EPS))
                   (>= u (- EPS)) (<= u (+ 1.0 EPS)))
          [(+ x1 (* t (- x2 x1))) (+ y1 (* t (- y2 y1)))])))))

(defn point-cmp
  "Order points by x then y, within tolerance."
  [[x1 y1] [x2 y2]]
  (if (> (abs-num (- x1 x2)) EPS)
    (compare x1 x2)
    (if (> (abs-num (- y1 y2)) EPS)
      (compare y1 y2)
      0)))

(defn transversal?
  "True when the two segments are not parallel."
  [[[x1 y1] [x2 y2]] [[x3 y3] [x4 y4]]]
  (let [x1 (double x1) y1 (double y1)
        x2 (double x2) y2 (double y2)
        x3 (double x3) y3 (double y3)
        x4 (double x4) y4 (double y4)
        denom (- (* (- x1 x2) (- y3 y4)) (* (- y1 y2) (- x3 x4)))]
    (> (abs-num denom) EPS)))

(defn seg-rank
  "Status order key of member m at the sweep position in sxa: probes
  rank below or above every segment sharing their height. Segments
  touching the current batch rank at their exact event height from
  snapa, so bundles tie exactly and the slope order below is the true
  just-right order instead of floating-point noise."
  [segs sxa snapa m]
  (if (vector? m)
    (let [[tag y] m]
      (if (= tag :lo) [y ##-Inf ##-Inf] [y ##Inf ##Inf]))
    (let [[[x1 y1] [x2 y2]] (nth segs m)
          sx @sxa
          snap (get @snapa m nil)
          y (if (some? snap)
              snap
              (if (== sx x1) y1 (if (== sx x2) y2 (+ y1 (* (- sx x1)
                                                           (/ (- y2 y1)
                                                              (- x2 x1)))))))]
      [y (/ (- y2 y1) (- x2 x1)) m])))

(defn record-hit
  "Fold a detected crossing into the queue, or report it directly when
  rounding placed it behind the sweep position. Pairs run in id order
  so every value matches the old pairwise loop bit for bit. Returns
  [events reported hitq] with hitq the queue key or nil."
  [events reported segs x s1 s2]
  (let [r (if (< s1 s2)
            (intersect (nth segs s1) (nth segs s2))
            (intersect (nth segs s2) (nth segs s1)))]
    (if (nil? r)
      [events reported nil]
      (if (< (first r) x)
        [events (conj reported r) nil]
        [(update events r (fnil conj #{}) s1 s2) reported r]))))

(defn lower-bound
  "First index in sorted vector v whose first entry reaches lo."
  [v lo]
  (loop [a 0 b (count v)]
    (if (>= a b)
      a
      (let [mid (quot (+ a b) 2)]
        (if (< (first (nth v mid)) lo)
          (recur (inc mid) b)
          (recur a mid))))))

(defn sweep
  "Bentley-Ottmann sweep over normalized segments; returns raw crossings."
  [segs]
  (let [n (count segs)
        starts (vec (map first segs))
        ends (vec (map second segs))
        vertical? (vec (map (fn [[[x1 _] [x2 _]]] (== (double x1) (double x2)))
                            segs))
        events (reduce (fn [ev i]
                         (-> ev
                             (update (nth starts i) (fnil conj #{}) i)
                             (update (nth ends i) (fnil conj #{}) i)))
                       (sorted-map) (range n))
        sxa (volatile! 0.0)
        snapa (volatile! {})
        cmp (fn [a b] (compare (seg-rank segs sxa snapa a)
                               (seg-rank segs sxa snapa b)))
        status (sorted-set-by cmp)]
    (loop [events events status status reported []]
      (if (empty? events)
        reported
        (let [x (first (first (first events)))
              kws (vec (map first
                             (take-while #(zero? (compare (first (first %)) x))
                                         (subseq events >= [x ##-Inf]))))
              involved (into #{} (mapcat #(get events %) kws))
              nonvert (remove #(nth vertical? %) involved)
              ending (into #{} (filter #(zero? (compare (first (nth ends %)) x))
                                        nonvert))
              starting (into #{} (filter #(zero? (compare (first (nth starts %)) x))
                                          nonvert))
              through (into #{} (remove #(or (contains? ending %)
                                             (contains? starting %))
                                        nonvert))
              verticals (vec (filter #(nth vertical? %) involved))
              status (reduce disj status (concat ending through))
              fresh (into #{} (concat starting through))
              snap (into {} (mapcat (fn [q]
                                      (let [qy (second q)]
                                        (map (fn [id] [id qy])
                                             (filter #(contains? fresh %)
                                                     (get events q)))))
                                    kws))]
          (vswap! sxa (fn [_] x))
          (vswap! snapa (fn [_] snap))
          (let [status (reduce conj status fresh)
                rends (vec (sort-by first
                                     (map (fn [i] [(second (nth ends i)) i]) ending)))
                [events reported queue]
                (reduce (fn [[events reported queue] v]
                          (let [[[vx y1] [_ y2]] (nth segs v)
                                lo-y (- (min y1 y2) EPS)
                                hi-y (+ (max y1 y2) EPS)
                                cands (subseq status >= [:lo lo-y] <= [:hi hi-y])
                                [events reported queue]
                                (reduce (fn [[events reported queue] s]
                                          (let [[events reported hitq]
                                                (record-hit events reported segs x v s)]
                                            [events reported
                                             (if (some? hitq) (conj queue hitq) queue)]))
                                        [events reported queue] cands)
                                k0 (lower-bound rends lo-y)
                                k1 (lower-bound rends (+ hi-y 1e-12))]
                            (loop [k k0 events events reported reported queue queue]
                              (if (>= k (min k1 (count rends)))
                                [events reported queue]
                                (let [[_ s] (nth rends k)
                                      [events reported hitq]
                                      (record-hit events reported segs x v s)]
                                  (recur (inc k) events reported
                                         (if (some? hitq) (conj queue hitq) queue)))))))
                        [events reported (vec kws)] verticals)
                [events status reported]
                (loop [events events status status reported reported
                       queue queue qpos 0 processed {} gapdone #{}]
                  (if (>= qpos (count queue))
                    [events status reported]
                    (let [q (nth queue qpos)
                          qpos (inc qpos)]
                      (if (or (nil? (get events q))
                              (not (zero? (compare (first q) x))))
                        (recur events status reported queue qpos processed gapdone)
                        (let [bundle (vec (filter #(and (not (nth vertical? %))
                                                         (contains? status %))
                                                   (get events q)))
                              newm (vec (remove #(contains? (get processed q #{}) %)
                                                bundle))]
                          (if (seq newm)
                            (let [[events status reported queue]
                                  (reduce
                                   (fn [[events status reported queue] m]
                                     (let [status (if (contains? @snapa m)
                                                    status
                                                    (let [s1 (disj status m)]
                                                      (vswap! snapa assoc m (second q))
                                                      (conj s1 m)))
                                           below (last (subseq status < m))
                                           above (first (subseq status > m))
                                           [events reported queue]
                                           (reduce
                                            (fn [[events reported queue] [a b]]
                                              (if (and (some? a) (some? b))
                                                (let [[events reported hitq]
                                                      (record-hit events reported segs x a b)]
                                                  [events reported
                                                   (if (some? hitq)
                                                     (conj queue hitq)
                                                     queue)])
                                                [events reported queue]))
                                            [events reported queue]
                                            [[below m] [m above]])]
                                       [events status reported queue]))
                                   [events status reported queue] newm)]
                              (recur events status reported queue qpos
                                     (update processed q (fnil into #{}) newm)
                                     gapdone))
                            (if (or (contains? gapdone q)
                                    (every? #(nth vertical? %) (get events q))
                                    (not (some #(and (not (nth vertical? %))
                                                     (not (contains? status %)))
                                               (get events q))))
                              (recur events status reported queue qpos
                                     processed gapdone)
                              (let [qy (second q)
                                    below (last (subseq status <= [:lo qy]))
                                    above (first (subseq status >= [:hi qy]))
                                    [events reported queue]
                                    (if (and (some? below) (some? above))
                                      (let [[events reported hitq]
                                            (record-hit events reported segs x
                                                        below above)]
                                        [events reported
                                         (if (some? hitq) (conj queue hitq) queue)])
                                      [events reported queue])]
                                (recur events status reported queue qpos
                                       processed (conj gapdone q))))))))))
                final-keys (vec (map first
                                   (take-while #(zero? (compare (first (first %)) x))
                                               (subseq events >= [x ##-Inf]))))
                reported (reduce (fn [rep q]
                                    (let [ss (get events q)]
                                      (if (>= (count ss) 2)
                                        (let [s0 (apply min ss)
                                              partner (first
                                                       (filter
                                                        #(transversal?
                                                          (nth segs s0) (nth segs %))
                                                        (sort (disj ss s0))))]
                                          (if (some? partner)
                                            (conj rep (or (intersect (nth segs s0)
                                                                     (nth segs partner))
                                                          q))
                                            rep))
                                        rep)))
                                  reported final-keys)
                events (reduce dissoc events final-keys)]
            (recur events status reported)))))))

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
          (die "error: first token must be an integer segment count")

          (or (> n 2305843009213693951) (< n -2305843009213693952))
          (die "error: segment count out of range")

          :else
          (let [coords (take (* 4 n) (rest tokens))
                raw (vec (for [[a b c d] (partition 4 coords)]
                           [(parse-double-token a) (parse-double-token b)
                            (parse-double-token c) (parse-double-token d)]))]
            (cond
              (some #(some nil? %) raw)
              (die "error: expected numeric segment coordinates after the count")

              (not (= (count raw) (max n 0)))
              (die "error: segment count does not match the supplied coordinates")

              :else
              (let [segs (vec (for [[x1 y1 x2 y2] raw]
                                (let [p1 [x1 y1] p2 [x2 y2]]
                                  (if (pos? (point-cmp p1 p2)) [p2 p1] [p1 p2]))))
                    pts (sweep segs)
                    sorted-pts (sort point-cmp pts)
                    unique-pts (reduce (fn [acc pt]
                                         (if (and (seq acc)
                                                  (zero? (point-cmp (peek acc) pt)))
                                           acc
                                           (conj acc pt)))
                                       [] sorted-pts)]
                (println (count unique-pts))
                (doseq [[x y] unique-pts]
                  (println (str (format-fixed (double x) 2) " "
                                (format-fixed (double y) 2))))))))))))

(-main)
