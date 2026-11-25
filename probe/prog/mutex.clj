;; shared_memory_mutex: shared-memory mutual exclusion protocols under a
;; round-robin step scheduler (one protocol line per scheduled step):
;; peterson2, filter (n-process), bakery (Lamport), dijkstra (K-state b/c/k),
;; burns (priority backoff). Regions R/T/C/E; reports entries + max
;; concurrency (must be 1). Batch 6 (T334). Input: algo n T.
(require '[clojure.string :as str])
(def separators #"\s+")
(defn tokenize [input] (if (str/blank? input) [] (str/split (str/trim input) separators)))
(defn one-step [st i algo n]
  (let [j (mod (inc i) n) r (nth (:region st) i)]
    (cond
      (= r 0) (-> st (assoc-in [:region i] 1) (assoc-in [:pc i] 0) (assoc-in [:aux i] 0))
      (= r 2) (-> st (assoc-in [:region i] 3) (assoc-in [:pc i] 0))
      (= r 3) (cond (= algo "peterson2") (-> st (assoc-in [:flag i] 0) (assoc-in [:region i] 0))
                    (= algo "filter") (-> st (assoc-in [:level i] 0) (assoc-in [:region i] 0))
                    (= algo "bakery") (-> st (assoc-in [:number i] 0) (assoc-in [:region i] 0))
                    (= algo "dijkstra") (if (zero? (nth (:pc st) i))
                                          (-> st (assoc-in [:c i] 1) (assoc-in [:pc i] 1))
                                          (-> st (assoc-in [:b i] 1) (assoc-in [:region i] 0)))
                    :else (-> st (assoc-in [:flag i] 0) (assoc-in [:region i] 0)))
      (= algo "peterson2")
      (let [p (nth (:pc st) i)]
        (cond (= p 0) (-> st (assoc-in [:flag i] 1) (assoc-in [:pc i] 1))
              (= p 1) (-> st (assoc :turn j) (assoc-in [:pc i] 2))
              (and (= (nth (:flag st) j) 1) (= (:turn st) j)) st
              :else (-> st (assoc-in [:region i] 2) (update-in [:entries i] inc))))
      (= algo "filter")
      (let [p (nth (:pc st) i) L (quot p 3) sub (mod p 3) lv (inc L)]
        (cond (= sub 0) (-> st (assoc-in [:level i] lv) (update-in [:pc i] inc))
              (= sub 1) (-> st (assoc-in [:victim lv] i) (update-in [:pc i] inc))
              :else (let [blocked (some (fn [t] (and (not= t i) (>= (nth (:level st) t) lv))) (range n))]
                      (cond (and blocked (= (nth (:victim st) lv) i)) st
                            (= lv (- n 1)) (-> st (assoc-in [:region i] 2) (update-in [:entries i] inc))
                            :else (update-in st [:pc i] inc)))))
      (= algo "bakery")
      (let [p (nth (:pc st) i)]
        (cond (= p 0) (-> st (assoc-in [:choosing i] 1) (assoc-in [:pc i] 1))
              (= p 1) (-> st (assoc-in [:number i] (inc (reduce max 0 (:number st)))) (assoc-in [:pc i] 2))
              (= p 2) (-> st (assoc-in [:choosing i] 0) (assoc-in [:pc i] 3) (assoc-in [:aux i] 0))
              :else (let [t (nth (:aux st) i) ni (nth (:number st) i)]
                      (cond (= t i) (update-in st [:aux i] inc)
                            (>= t n) (-> st (assoc-in [:region i] 2) (update-in [:entries i] inc))
                            (= (nth (:choosing st) t) 1) st
                            (and (not= (nth (:number st) t) 0)
                                 (or (< (nth (:number st) t) ni)
                                     (and (= (nth (:number st) t) ni) (< t i)))) st
                            :else (update-in st [:aux i] inc)))))
      (= algo "dijkstra")
      (let [p (nth (:pc st) i)]
        (cond (= p 0) (-> st (assoc-in [:b i] 0) (assoc-in [:pc i] 1))
              (= p 1) (assoc-in st [:pc i] (if (not= (:k st) i) 2 4))
              (= p 2) (-> st (assoc-in [:c i] 1) (assoc-in [:pc i] 3))
              (= p 3) (-> st (assoc :k (if (= (nth (:b st) (:k st)) 1) i (:k st))) (assoc-in [:pc i] 1))
              (= p 4) (-> st (assoc-in [:c i] 0) (assoc-in [:pc i] 5))
              (some (fn [t] (and (not= t i) (zero? (nth (:c st) t)))) (range n)) (assoc-in st [:pc i] 1)
              :else (-> st (assoc-in [:region i] 2) (update-in [:entries i] inc))))
      :else
      (let [p (nth (:pc st) i)]
        (cond (= p 0) (-> st (assoc-in [:flag i] 0) (assoc-in [:pc i] 1))
              (= p 1) (-> st (assoc-in [:flag i] 1) (assoc-in [:pc i] 2) (assoc-in [:aux i] 0))
              (= p 2) (let [t (nth (:aux st) i)]
                        (cond (>= t i) (-> st (assoc-in [:aux i] (inc i)) (assoc-in [:pc i] 3))
                              (= (nth (:flag st) t) 1) (assoc-in st [:pc i] 4)
                              :else (update-in st [:aux i] inc)))
              (= p 4) (-> st (assoc-in [:flag i] 0) (assoc-in [:pc i] 5))
              (= p 5) (if (= (nth (:flag st) (nth (:aux st) i)) 1) st
                        (-> st (assoc-in [:pc i] 1) (assoc-in [:aux i] 0)))
              :else (let [t (nth (:aux st) i)]
                      (cond (>= t n) (-> st (assoc-in [:region i] 2) (update-in [:entries i] inc))
                            (= (nth (:flag st) t) 1) st
                            :else (update-in st [:aux i] inc))))))))
(defn -main [input]
  (when-not (str/blank? input)
    (let [toks (tokenize input)
          algo (nth toks 0) n (Long/parseLong (nth toks 1)) T (Long/parseLong (nth toks 2))
          st0 {:region (vec (repeat n 0)) :pc (vec (repeat n 0)) :aux (vec (repeat n 0))
               :entries (vec (repeat n 0)) :flag (vec (repeat n 0)) :level (vec (repeat n 0))
               :choosing (vec (repeat n 0)) :b (vec (repeat n 1)) :c (vec (repeat n 1))
               :number (vec (repeat n 0)) :victim (vec (repeat n 0)) :turn 0 :k 0}]
      (loop [s 0 st st0 mc 0]
        (if (>= s T)
          (do (doseq [i (range n)] (println (str "P" i ": entries=" (nth (:entries st) i))))
              (println (str "MAXCONCURRENT: " mc)) (println (str "STEPS: " T)))
          (let [st2 (one-step st (mod s n) algo n)
                cc (count (filter #(= % 2) (:region st2)))]
            (recur (inc s) st2 (max mc cc))))))))
(-main (slurp *in*))
