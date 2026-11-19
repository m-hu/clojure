(def n (parse-long (clojure.string/trim (slurp *in*))))
(let [s "10 20 30 40 50 60 70 80 90 100"] (println (loop [i 0 t 0] (if (< i n) (recur (inc i) (+ t (count (clojure.string/split s #" ")))) t))))
