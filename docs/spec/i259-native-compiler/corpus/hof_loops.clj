;; Inline traversal loops (#271 Slice 4c): doseq (dorun of map), mapv,
;; vec / into [] of map, and reduce over a literal fn run the fn's body in
;; place for vectors and long ranges; every other collection takes the
;; general call over the evaluated collection, exactly like run.
(def v [3 1 4 1 5])
(doseq [x v] (print x " "))
(println)
(doseq [i (range 2 9 3)] (print i " "))
(println)
(doseq [x '(7 8)] (print x " "))
(doseq [[k n] {:a 1}] (print k n))
(doseq [c "hi"] (print c))
(doseq [x nil] (print :never))
(doseq [x []] (print :never))
(println)
(println (mapv (fn [x] (* x x)) v) (mapv (fn [i] (- i)) (range 4)) (mapv (fn [x] x) #{9}))
(println (vec (map (fn [x] (inc x)) v)) (into [] (map (fn [x] (str x)) (range 3))))
(println (vec (for [x v] (* 2 x))) (into [] (for [i (range 3)] [i i])))
(println (reduce (fn [acc x] (+ acc x)) 0 v) (reduce (fn [acc x] (conj acc x)) [] (range 4)))
(println (reduce (fn [acc x] (if (> x 3) (reduced acc) (+ acc x))) 0 v))
(println (reduce (fn [acc x] (if (> acc 1) (reduced :stop) acc)) 5 (range 3)))
(println (reduce (fn [acc x] (assoc acc x (count acc))) {} '(:a :b)))
(let [outer [1 2]]
  (println (mapv (fn [x] (conj outer x)) [10 20]) outer))
(let [acc (atom [])]
  (doseq [x (range 3) y [:a :b]] (swap! acc conj [x y]))
  (println @acc))
(println (mapv (fn [x] (mapv (fn [y] (* x y)) (range 3))) [1 2]))
(println (try (doseq [x [1 0 2]] (print (quot 6 x) " ")) (catch ArithmeticException e (str "caught " (.getMessage e)))))
(println (try (mapv (fn [x] x) 5) (catch Exception e (str "caught " (.getMessage e)))))
(def lazy-log (atom []))
(def src (map (fn [x] (swap! lazy-log conj [:make x]) x) (list 1 2 3)))
(doseq [x src] (swap! lazy-log conj [:use x]))
(println @lazy-log)
(println (mapv (fn [x] (let [y (* x 10)] ((fn [] y)))) [1 2]))
(println (count (mapv (fn [x] (let [y (* x 10)] ((fn [] y)))) (range 100000))))
;; A local the body reads stays put even where the collection was its
;; last use (the closure would have captured it first).
(let [n 3 m (mapv (fn [_] (mapv (fn [_] n) (range n))) (range n))] (println m))
(let [w [1 2 3]] (println (reduce (fn [acc x] (+ acc (count w) x)) (count w) w)))
