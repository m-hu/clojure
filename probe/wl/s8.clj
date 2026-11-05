(println (reduce + (reduce (fn [v i] (assoc v (mod i 1000) i)) (vec (repeat 1000 0)) (range 200000))))
