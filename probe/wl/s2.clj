(println (count (reduce (fn [m i] (assoc m i (* 2 i))) {} (range 50000))))
