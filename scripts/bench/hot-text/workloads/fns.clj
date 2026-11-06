(def fibm (memoize (fn [n] (if (< n 2) n (+ (fibm (- n 1)) (fibm (- n 2)))))))
(letfn [(ev? [n] (if (zero? n) true (od? (dec n)))) (od? [n] (if (zero? n) false (ev? (dec n))))]
  (println (ev? 100) (fibm 60)))
(println ((comp inc #(* 2 %)) 5) ((partial + 10) 5) ((juxt min max) 3 1 2) ((complement odd?) 2) ((fnil inc 0) nil))
(println (apply map vector [[1 2] [3 4]]) (trampoline (fn f [n] (if (pos? n) #(f (dec n)) :done)) 100) (identity 1) (constantly 1))
(println (let [[a & more] [1 2 3] {:keys [x y] :or {y 9}} {:x 1}] [a more x y]))
(println (case 2 1 :one 2 :two :other) (cond (> 1 2) :a :else :b) (condp = 3 1 :a 3 :c :d) (when-let [x 1] x) (if-let [x nil] x :no))
