(def n (parse-long (clojure.string/trim (slurp *in*))))
(dotimes [i n] (println (format "%.4f" (* 1.5 i))))
