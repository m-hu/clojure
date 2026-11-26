;; `subvec` reads inline; other colls and every error take the native (#271
;; Slice 6p). (`vec` keeps its native, which consumes lazy seqs it owns.)
(defn digits-of [tok] (let [cs (vec tok) neg (= (first cs) \-) d (subvec cs (if neg 1 0))] [neg d (count d) (apply str d)]))
(println (digits-of "-123") (digits-of "45") (digits-of "") (vec "") (vec [1 2]) (meta (vec (with-meta [1] {:m 1}))) (vec '(1 2)) (vec nil))
(println (subvec [1 2 3] 1) (subvec [1 2 3] 1 2) (subvec [1 2 3] 3) (subvec (subvec [1 2 3 4] 1) 1 2) (subvec [0 1] 0 0))
(println (try (subvec [1 2] 3) (catch Exception e (class e))) (try (subvec [1 2] -1) (catch Exception e (class e)))
         (try (subvec [1 2] 2 1) (catch Exception e (class e))) (try (subvec "ab" 1) (catch Exception e (class e)))
         (try (vec 5) (catch Exception e (.getMessage e))) (class (subvec [1 2 3] 1)) (= (subvec [1 2 3] 1) [2 3]))
