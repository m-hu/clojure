(def n (parse-long (clojure.string/trim (slurp *in*))))
(let [s (apply str (interpose " " (range n)))] (println (reduce + (map parse-long (clojure.string/split s #" ")))))
