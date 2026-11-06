(let [l '(1 2 3) q (into clojure.lang.PersistentQueue/EMPTY [1 2 3])]
  (println (conj l 0) (cons 0 l) (list* 1 2 [3 4]) (first l) (rest l) (peek q) (seq (pop q)) (count (conj q 4)))
  (println (reverse l) (into '() [1 2 3]) (apply list [1 2]) (list? l) (seq? (seq l)) (count (repeat 5 :x))))
