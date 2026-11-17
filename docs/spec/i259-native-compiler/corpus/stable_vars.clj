;; Stable var roots and borrowed constants (#271): a single-def var
;; reads from its root cache, checked where code can run before the
;; def; keyword invokes go through the bitcode get, falling back to the
;; keyword's own invoke for shapes the helper leaves to the runtime.
(declare tbl)
(defn early [x] (if x (nth tbl 0) :none))
(println (early false))
(def tbl [10 20 30])
(println (early true))
(def big (zipmap (map #(keyword (str "k" %)) (range 20)) (range 20)))
(def small {:a 1 :b 2 "s" 3})
(defn probe [i]
  [(nth tbl i) (:a small) (:z small :dflt) (:k7 big) (:nope big 0)
   (get small "s") (contains? #{:a :b} :b) (:a #{:a}) (:a nil) (:a [1 2])
   (count tbl) (get tbl 1)])
(println (probe 2))
(def ^:dynamic *depth* 0)
(println (binding [*depth* 5] (inc *depth*)) *depth*)
(def twice 1)
(def twice (+ twice 1))
(println twice)
(defn reset-inner [] (def inner (rand-int 1)))
(reset-inner)
(println inner)
(def taken 4)
(println @#'taken (var-get #'taken))
(defn add-all [a b c] (+ (count a) (count b) c))
(println (add-all tbl "abc" 1) (add-all [1 2] tbl 0))
