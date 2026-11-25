(def n (parse-long (clojure.string/trim (slurp *in*))))
(println (reduce + (map (fn [i] (count (str "P" i ": entries=" (* 2 i)))) (range n))))
