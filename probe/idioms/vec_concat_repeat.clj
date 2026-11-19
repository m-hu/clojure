(def n (parse-long (clojure.string/trim (slurp *in*))))
(let [v (vec (range n))] (println (count (vec (concat v (repeat n 0.0))))))
