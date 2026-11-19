(def n (parse-long (clojure.string/trim (slurp *in*))))
(let [s "12345"] (println (loop [i 0 t 0] (if (< i n) (recur (inc i) (+ t (parse-long s))) t))))
