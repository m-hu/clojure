(println (loop [i 0 s 0] (if (< i 1000000) (recur (inc i) (+ s i)) s)))
