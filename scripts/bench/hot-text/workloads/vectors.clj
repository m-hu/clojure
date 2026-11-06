(let [v (vec (range 100))]
  (println (subvec v 10 15) (peek v) (count (pop v)) (assoc v 0 :x) (nth v 99) (get v 100 :none) (rseq [1 2 3]))
  (println (mapv inc [1 2 3]) (filterv even? (range 10)) (into [] (range 5)) (conj v 100) (vec (map str [1 2])))
  (println (reduce + v) (count (into v v)) (first v) (rest [1 2 3]) (next [1]) (vector 1 2 3) (vector-of :long 1 2) (empty v)))
