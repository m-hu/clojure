(def n (parse-long (clojure.string/trim (slurp *in*))))
(let [v (vec (range n))] (println (count (mapv (fn [x] (inc x)) v))))
