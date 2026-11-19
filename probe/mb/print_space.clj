(def n (parse-long (clojure.string/trim (slurp *in*))))
(dotimes [i n] (print i "") )
