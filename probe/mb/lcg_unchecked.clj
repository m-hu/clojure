(def n (parse-long (clojure.string/trim (slurp *in*))))
(println (loop [i 0 s 42] (if (< i n) (recur (inc i) (unchecked-add (unchecked-multiply s 6364136223846793005) 1442695040888963407)) s)))
