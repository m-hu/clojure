(def n (parse-long (clojure.string/trim (slurp *in*))))
(dotimes [i n] (println (str "P" i ": entries=" (* 2 i))))
