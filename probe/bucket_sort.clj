;; bucket-sort: pure Clojure bucket sort.
;;
;; Distribute the values into one ordered bucket per element, sort
;; each bucket, and concatenate the buckets in order. Buckets stay
;; tiny on uniform input, so the average time is O(n); oversized
;; buckets fall back to a linearithmic sort, which keeps the worst
;; case at O(n log n). Space is O(n). Pure Clojure with no host
;; interop: persistent vectors, loop and recur, reduce where it fits.
;; Input and output match the shared sort driver contract: stdin
;; holds n followed by values, stdout is [v1,v2,...].

(def digit-values
  "Digit character to numeric value."
  (zipmap "0123456789" (range)))

(def separators
  "Whitespace characters that separate input tokens."
  #{\space \tab \newline \return \formfeed \u000B})

(defn tokenize
  "Split s on whitespace into a vector of token strings."
  [s]
  (if (nil? s)
    []
    (filterv seq (clojure.string/split s #"[ \t\n\r\f\x0B]+"))))

(def min-long
  "Most negative long value: 1 shifted into the sign bit."
  (bit-shift-left 1 63))

(def max-long
  "Largest long value: every bit below the sign bit set."
  (bit-not min-long))

(defn parse-token
  "Parse tok as a decimal long integer. An optional leading minus or
   plus sign is accepted. Returns nil when tok is malformed or falls
   outside the long range: the sign and ASCII digits are matched
   first, then parse-long reads the value."
  [tok]
  (when (and (string? tok) (re-matches #"[+-]?[0-9]+" tok))
    (parse-long tok)))

(defn format-sorted
  "Render xs as [v1,v2,...] with no spaces or trailing newline."
  [xs]
  (str "[" (apply str (interpose "," (map str xs))) "]"))

(defn insertion-small
  "Insertion sort of one bucket. Stable; O(m squared) worst and O(m)
   best for a bucket of m values, which is constant work for the
   tiny buckets this sort produces on uniform input."
  [bucket]
  (reduce (fn [out v]
            (let [pos (count (take-while (fn [u] (<= u v)) out))]
              (into (conj (subvec out 0 pos) v) (subvec out pos))))
          []
          bucket))

(def big-bucket
  "Buckets above this size use the fallback sorter. Any constant
   keeps the average case linear while the fallback keeps the
   worst case at O(n log n)."
  32)

(defn sort-bucket
  "Sort one bucket: insertion for the usual tiny bucket, a
   linearithmic sort for a degenerate oversized one."
  [bucket]
  (if (<= (count bucket) big-bucket)
    (insertion-small bucket)
    (vec (sort bucket))))

(defn bucket-sort
  "Bucket sort of xs. O(n) average time on uniform input, O(n log n)
   worst case, O(n) space. Values spread over one bucket per element
   by exact proportional index math, so even a range spanning the
   whole long domain distributes correctly."
  [xs]
  (let [v (vec xs)
        n (count v)]
    (if (<= n 1)
      v
      (let [mn (reduce min v)
            mx (reduce max v)]
        (if (= mn mx)
          v
          (let [lo (bigint mn)
                span (inc (- (bigint mx) lo))
                buckets (reduce (fn [bs x]
                                  (let [off (- (bigint x) lo)
                                        idx (long (quot (* off n) span))]
                                    (assoc bs idx (conj (nth bs idx) x))))
                                (vec (repeat n []))
                                v)]
            (into [] (mapcat sort-bucket) buckets)))))))

(defn -main
  "Read n followed by values from input and print the sorted values
   as [v1,v2,...]. Only the first n values are used; extra tokens are
   ignored and missing values shorten the sort, matching the original
   driver. Blank input prints nothing. Malformed input prints a
   message to stderr and returns normally."
  [input]
  (let [toks (tokenize input)]
    (when (seq toks)
      (let [nums (mapv parse-token toks)]
        (if (some nil? nums)
          (binding [*out* *err*]
            (println "error: expected n followed by integers"))
          (let [avail (max 0 (dec (count nums)))
                k (min (max (first nums) 0) avail)]
            (->> (rest nums)
                 (take k)
                 (vec)
                 (bucket-sort)
                 (format-sorted)
                 (print))))))))

(-main (slurp *in*))
