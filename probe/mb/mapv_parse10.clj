(def n (parse-long (clojure.string/trim (slurp *in*))))
(let [v ["1" "2" "3" "4" "5" "6" "7" "8" "9" "10"]] (println (loop [i 0 t 0] (if (< i n) (recur (inc i) (+ t (reduce + (mapv parse-long v)))) t))))
