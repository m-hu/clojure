(def n (parse-long (clojure.string/trim (slurp *in*))))
(let [v (vec (range n))] (println (count (vec (for [x v] (* x x))))))
