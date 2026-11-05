(def grid (vec (for [i (range 300)] (vec (for [j (range 300)] (mod (* i j) 7))))))
(println (reduce + (for [i (range 300) j (range 300) :when (odd? (get-in grid [i j]))] 1)))
