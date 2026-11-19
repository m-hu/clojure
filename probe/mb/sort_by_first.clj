(def n (parse-long (clojure.string/trim (slurp *in*))))
(def ts (vec (for [i (range 50)] [(mod (* i 7) 13) i])))
(println (loop [i 0 s 0] (if (< i n) (recur (inc i) (+ s (count (sort-by first ts)))) s)))
