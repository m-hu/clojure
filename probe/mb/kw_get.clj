(def n (parse-long (clojure.string/trim (slurp *in*))))
(let [mm {:a 1 :b 2}] (println (loop [i 0 s 0] (if (< i n) (recur (inc i) (+ s (:a mm))) s))))
