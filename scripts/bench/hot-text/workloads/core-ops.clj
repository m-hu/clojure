;; Core operations most programs reach that the other workloads skip:
;; reading all of stdin, the comparison and arithmetic ops over locals,
;; constants and captured values (tested and as values, inside loops long
;; enough for the loop tier), vector equality, hashing and ordering, the
;; filter / concat / drop-while walks, number parsing, comparator-sorted
;; collections.
(def input (slurp *in*))

(defn tally
  "Counts of each comparison over `n` iterations, against a param, a
  constant and a captured value, in test and value position."
  [n k]
  (let [u (- n 3)
        f (fn [i] [(< i u) (<= u i) (> i u) (>= u i) (= i u) (== u i) (compare i u) (+ i u) (- u i) (* i u) (min i u) (max u i)])]
    (loop [i 0 lt 0 le 0 gt 0 ge 0 eq 0 ne 0 acc 0 m 0 v []]
      (if (< i n)
        (recur (inc i)
               (if (< i k) (inc lt) lt)
               (if (<= i 7) (inc le) le)
               (if (> k i) (inc gt) gt)
               (if (>= 7 i) (inc ge) ge)
               (if (= i k) (inc eq) eq)
               (if (not= i 7) (inc ne) ne)
               (+ acc (* i 2) (- k i) (* 3 i) (min i k) (max 7 i) (compare i k) (compare 7 i))
               (if (== i 7) (inc m) (if (zero? (rem i 5)) (+ m 2) m))
               (if (and (pos? i) (not (neg? (- i k))) (some? v) (not (nil? k))) (conj v (- i)) v))
        [lt le gt ge eq ne acc m (count v) (f n) (f (- n)) (double acc) (long (/ acc 2))]))))

(defn vectors
  "Vector equality, hashing, ordering, and vectors as keys."
  [n]
  (let [pts (vec (for [i (range n)] [(rem i 7) (quot i 7)]))
        seen (set pts)
        by (group-by first pts)
        sorted (sort pts)
        cmp (sorted-set-by (fn [a b] (compare (second a) (second b))) [1 2] [3 1] [2 3])]
    [(count seen) (contains? seen [3 1]) (= (first sorted) [0 0]) (= [1 2] [1 2]) (not= [1 2] [2 1])
     (compare [1 2] [1 3]) (hash [1 2]) (count by) (first cmp) (get (zipmap pts (range)) [2 2])
     (= (seq [1 2]) [1 2]) (= '(1 2) [1 2])]))

(defn walks
  "The lazy filter, concat and while walks, and their eager forms."
  [n]
  (let [xs (vec (range (- n) n))]
    [(count (filter even? xs)) (count (filterv odd? xs)) (count (remove neg? xs))
     (count (concat xs (reverse xs))) (first (drop-while neg? xs)) (count (take-while neg? xs))
     (count (concat (filter pos? xs) (take 3 xs) [:x])) (vec (take 3 (drop-while #(< % 2) xs)))]))

(defn numbers
  "Number parsing and the double and ratio arithmetic it feeds."
  [n]
  (let [ds (mapv #(parse-double (str % ".25")) (range n))
        ls (mapv #(parse-long (str %)) (range n))
        rs (mapv #(/ % 3) (range 1 n))]
    [(reduce + ds) (reduce + ls) (reduce + rs) (apply max ds) (double (apply min rs))
     (quot (reduce * 1 (take 10 ls)) 7) (Math/sqrt (reduce + ds)) (Math/abs -2.5)]))

(defn sorted-colls
  "Comparator-sorted sets and maps, and range queries over them."
  [n]
  (let [ss (into (sorted-set-by >) (range n))
        sm (into (sorted-map-by (fn [a b] (compare b a))) (map (fn [i] [i (* i i)]) (range n)))]
    [(first ss) (count (subseq ss > 5)) (first sm) (count (rsubseq sm < 10)) (get sm 3) (contains? ss 4)]))

(println (count input) (tally 200 37) (vectors 120) (walks 100) (numbers 80) (sorted-colls 60))
