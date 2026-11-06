(require '[clojure.set :as set])
(let [a (set (range 0 100 2)) b (set (range 0 100 3))
      m (zipmap (range 50) (map #(* % %) (range 50)))]
  (println (count (set/union a b)) (count (set/intersection a b)) (count (set/difference a b)))
  (println (count (conj a 1)) (count (disj b 0)) (contains? a 4) (get m 7) (get m 99 :none))
  (println (count (dissoc m 1 2 3)) (update m 1 inc) (select-keys m [1 2 3]) (count (merge m {100 1})))
  (println (group-by odd? (range 10)) (frequencies "mississippi") (into {} [[1 2] [3 4]]))
  (println (reduce-kv (fn [acc k v] (+ acc k v)) 0 m) (get-in {:a {:b 1}} [:a :b]) (assoc-in {} [:x :y] 1) (update-in {:a 1} [:a] + 5))
  (println (keys (sort-by key {3 :c 1 :a 2 :b})) (vals {:a 1}) (find m 3) (count (filter (fn [[k v]] (even? k)) m))))
