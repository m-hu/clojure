;; Print settings vars a compiled program first references after printing
;; (installed then, on first use) still steer later prints: the
;; settings cache remembers a missing var only until a new var appears (#271).
(println 1)
(prn [1 2 3])
(binding [*print-length* 1] (prn [1 2 3]))
(prn [4 5 6])
(binding [*print-level* 1] (prn [[1] 2]))
(binding [*print-meta* true] (prn (with-meta [1] {:m 1})))
(prn (with-meta [1] {:m 1}))
(println (pr-str [7 8]) (with-out-str (pr :x)))
