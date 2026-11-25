(def n (parse-long (clojure.string/trim (slurp *in*))))
(let [v [1 2 3 4 5]] (println (loop [i 0 t 0] (if (< i n) (recur (inc i) (+ t (count (clojure.string/join " " v)))) t))))
