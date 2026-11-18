;; with-out-str and with-in-str bind clojure.core's *out*/*in* in a
;; compiled program too: a Lite program refers nothing, so the stream
;; var resolves through clojure.core (#271).
(println (count (with-out-str (print 1) (println "x"))))
(println (pr-str (with-out-str (prn [1 "a"]) (print :k))))
(println (with-out-str (println (with-out-str (print "inner")))))
(println (with-in-str "a b\nc" (read-line)))
(println (with-in-str "1 2 3" (count (clojure.string/split (read-line) #" "))))
(println (try (with-out-str (print "lost") (throw (ex-info "boom" {})))
              (catch Exception e (ex-message e))))
(print "after the throw, *out* is stdout again")
(newline)
