;; #271: metadata-free constant collections (quoted data, catch class lists,
;; :arglists) build at startup without the reader, so a macro-built list
;; carries no read position (meta of :arglists is nil, as under run);
;; source lists keep theirs.
(defn f [x] x)
(defn g ([] 0) ([a] a) ([a & more] (count more)))
(println (meta (:arglists (meta #'f))) (:arglists (meta #'g)) (meta '(1 2)))
(println '[a b c] '{:a 1 "b" [2 3] :c #{x}} '#{p} '() '[] '{} (quote (1 (2 (3)))))
(println (try (throw (ex-info "boom" {})) (catch clojure.lang.ExceptionInfo e (ex-message e)) (catch Exception e :other)))
(println (vec (range 3)) '[1.5 \a "s" nil true :k/w sym/x])
