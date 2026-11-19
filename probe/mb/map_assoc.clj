(def n (parse-long (clojure.string/trim (slurp *in*))))
(println (count (loop [i 0 m {}] (if (< i n) (recur (inc i) (assoc m i i)) m))))
