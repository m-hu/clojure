(let [a (atom 0) v (volatile! 0)]
  (dotimes [i 1000] (swap! a + i) (vswap! v inc))
  (reset! a (+ @a @v))
  (println @a @v)
  (println (count (persistent! (reduce conj! (transient []) (range 1000)))))
  (println (count (persistent! (reduce #(assoc! %1 %2 %2) (transient {}) (range 500)))))
  (println (count (persistent! (reduce conj! (transient #{}) (range 500)))))
  (doseq [x [1 2 3]] (swap! a + x))
  (println @a (loop [i 0] (if (< i 10) (recur (inc i)) i))))
