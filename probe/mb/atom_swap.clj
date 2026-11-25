(def n (parse-long (clojure.string/trim (slurp *in*))))
(let [a (atom 0)] (dotimes [i n] (swap! a + i)) (println @a))
