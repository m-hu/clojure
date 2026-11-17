;; A program holding a var object (#'f) with a value reader keeps every
;; def's metadata, attr maps and type hints included (#271).
(defn f "doc of f" {:added "1.0" :tag String} [x] x)
(defn g [a b] (+ a b))
(println (:added (meta #'f)) (:doc (meta #'f)) (:arglists (meta #'f)) (:tag (meta #'f)))
(println (:arglists (meta #'g)) (f 1) (g 1 2))
