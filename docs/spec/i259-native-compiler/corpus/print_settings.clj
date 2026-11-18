;; Printing reads its settings vars through a cache of clojure.core var
;; ids and roots (#271): thread bindings, root writes and nested
;; with-out-str still steer every print, exactly as under run.
(prn [1 2 3])
(binding [*print-length* 1] (prn [1 2 3]))
(prn [1 2 3])
(alter-var-root #'*print-length* (constantly 2))
(prn [1 2 3])
(alter-var-root #'*print-length* (constantly nil))
(prn [1 2 3])
(println (with-out-str (prn :inside)) :outside)
(def ^:dynamic *x* 1)
(binding [*print-readably* false] (prn "q"))
(prn "q")
