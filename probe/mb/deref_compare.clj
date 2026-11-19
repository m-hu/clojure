(def n (parse-long (clojure.string/trim (slurp *in*))))
(def a (atom 0))
(println (loop [i 0 s 0] (if (< i n) (recur (inc i) (+ s @a (compare i 7))) s)))
