(def n (parse-long (clojure.string/trim (slurp *in*))))
(let [h (fn [x] (inc x))] (println (loop [i 0 s 0] (if (< i n) (recur (inc i) (+ s (h i))) s))))
