;; Direct calls of stable defns (#271 Slice 4b): calls from the defn's
;; own unit (recursion) or a later one skip the root check; forward calls
;; through declare keep it. Params only read as call operands borrow the
;; caller's argument array; ones that reach an owned op, a let, or the
;; result still get their own value.
(declare countdown)
(defn first-of [w i] (nth w i))
(defn relay [w] (first-of w 0))
(defn grow [w x] (conj w x))
(defn kept [w] (let [v w] (conj v :k)))
(defn same [w] w)
(defn ahead [x] (if (pos? x) (countdown (dec x)) [:done]))
(defn countdown [x] (conj (ahead x) x))
(defn fact [n] (if (< n 2) 1 (* n (fact (dec n)))))
(def base [1 2 3])
(println (relay base) (grow base 4) base (kept base) (same base) (countdown 3) (fact 15))
(defn deep [n] (if (zero? n) :bottom (deep (dec n))))
(println (deep 20000))
(println (reduce (fn [acc x] (conj acc (first-of base (rem x 3)))) [] (range 5)))
