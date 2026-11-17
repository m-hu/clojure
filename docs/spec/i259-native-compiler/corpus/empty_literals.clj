;; #271: an empty collection literal without metadata is one value per
;; form, as under run (each evaluation returns it; distinct forms stay
;; distinct), and adding to it leaves it empty.
(defn v [] [])
(defn m [] {})
(defn s [] #{})
(defn vm [] ^:tag [])

(println (identical? (v) (v)) (identical? (m) (m)) (identical? (s) (s)))
(println (identical? [] []) (identical? {} {}) (identical? #{} #{}))
(println (conj (v) 1) (assoc (m) :a 1) (conj (s) 2) (v) (m) (s))
(println (meta (vm)) (vm) (count (v)) (empty? (m)))
(println (loop [i 0 acc []] (if (< i 3) (recur (inc i) (conj acc i)) acc))
         (reduce (fn [acc x] (assoc acc x (* x x))) {} [1 2 3])
         (into #{} [1 1 2]))
