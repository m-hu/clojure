(def n (parse-long (clojure.string/trim (slurp *in*))))
(println (count (vec (map inc (range n)))))
