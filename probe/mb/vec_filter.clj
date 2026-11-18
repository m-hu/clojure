(def n (parse-long (clojure.string/trim (slurp *in*))))
(let [v (vec (range n))] (println (count (vec (filter even? v)))))
