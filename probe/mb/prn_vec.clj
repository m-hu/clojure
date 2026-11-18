(def n (parse-long (clojure.string/trim (slurp *in*))))
(dotimes [i n] (prn [i (* 2 i) "a"]))
