;; markowitz: Markowitz mean-variance efficient frontier, minimum-variance
;; & tangency portfolios plus Black-Litterman posterior blending for one
;; view. Small Gaussian solver with first-max partial pivoting.
;; Batch 6 (T381). Input: n, mu, cov, rf, 3 targets, tau, P row, Q, omega.
(require '[clojure.string :as str])
(def separators #"\s+")
(defn tokenize [input] (if (str/blank? input) [] (str/split (str/trim input) separators)))
(defn fmt [s & xs] (String/format java.util.Locale/US s (to-array xs)))
(defn pivot [A k n]
  (loop [i (inc k) piv k best (Math/abs (nth (nth A k) k))]
    (if (>= i n) piv
      (if (> (Math/abs (nth (nth A i) k)) best) (recur (inc i) i (Math/abs (nth (nth A i) k))) (recur (inc i) piv best)))))
(defn matswap [A k p] (assoc (assoc A k (nth A p)) p (nth A k)))
(defn vecswap [v k p] (assoc (assoc v k (nth v p)) p (nth v k)))
(defn gsolve [A0 b0]
  (let [n (count A0)
        [A b] (loop [k 0 A A0 b b0]
                (if (>= k n) [A b]
                  (let [p (pivot A k n) A2 (matswap A k p) b2 (vecswap b k p)]
                    (recur (inc k)
                      (loop [i (inc k) a A2]
                        (if (>= i n) a
                          (let [f (/ (nth (nth a i) k) (nth (nth a k) k))]
                            (recur (inc i) (loop [j k a2 a]
                                              (if (>= j n) a2 (recur (inc j) (assoc-in a2 [i j] (- (nth (nth a2 i) j) (* f (nth (nth a2 k) j)))))))))))
                      (loop [i (inc k) bb b2]
                        (if (>= i n) bb
                          (let [f (/ (nth (nth A2 i) k) (nth (nth A2 k) k))]
                            (recur (inc i) (assoc bb i (- (nth bb i) (* f (nth bb k))))))))))))]
    (loop [k (dec n) x (vec (repeat n 0.0))]
      (if (< k 0) x
        (recur (dec k) (assoc x k (/ (- (nth b k) (reduce + 0.0 (map (fn [j] (* (nth (nth A k) j) (nth x j))) (range (inc k) n)))) (nth (nth A k) k))))))))
(defn ginverse [A0]
  (let [n (count A0)]
    (loop [k 0 A A0 I (mapv (fn [i] (mapv (fn [j] (if (= i j) 1.0 0.0)) (range n))) (range n))]
      (if (>= k n) I
        (let [p (pivot A k n) A2 (matswap A k p) I2 (matswap I k p)
              d (nth (nth A2 k) k)
              A3 (reduce (fn [a j] (-> a (assoc-in [k j] (/ (nth (nth a k) j) d)))) A2 (range n))
              I3 (reduce (fn [a j] (assoc-in a [k j] (/ (nth (nth a k) j) d))) I2 (range n))]
          (recur (inc k)
            (loop [i 0 a A3]
              (if (>= i n) a
                (if (= i k) (recur (inc i) a)
                  (let [f (nth (nth a i) k)]
                    (recur (inc i) (loop [j 0 a2 a]
                                      (if (>= j n) a2
                                        (recur (inc j) (assoc-in a2 [i j] (- (nth (nth a2 i) j) (* f (nth (nth a2 k) j))))))))))))
            (loop [i 0 a I3]
              (if (>= i n) a
                (if (= i k) (recur (inc i) a)
                  (let [f (nth (nth A3 i) k)]
                    (recur (inc i) (loop [j 0 a2 a]
                                      (if (>= j n) a2
                                        (recur (inc j) (assoc-in a2 [i j] (- (nth (nth a2 i) j) (* f (nth (nth a2 k) j))))))))))))))))))
(defn -main [input]
  (let [toks (tokenize input)]
    (when (seq toks)
      (let [p (atom 0) nxt (fn [] (let [v (nth toks @p)] (swap! p inc) v))
            n (Long/parseLong (nxt))
            mu (mapv (fn [_] (Double/parseDouble (nxt))) (range n))
            S (mapv (fn [_] (mapv (fn [_] (Double/parseDouble (nxt))) (range n))) (range n))
            rf (Double/parseDouble (nxt))
            tg (mapv (fn [_] (Double/parseDouble (nxt))) (range 3))
            tau (Double/parseDouble (nxt))
            P (mapv (fn [_] (Double/parseDouble (nxt))) (range n))
            Q (Double/parseDouble (nxt)) om (Double/parseDouble (nxt))
            A (mapv (fn [i] (mapv (fn [j] (cond (< i n) (if (< j n) (nth (nth S i) j) 1.0) (< j n) 1.0 :else 0.0)) (range (inc n)))) (range (inc n)))
            b (assoc (vec (repeat (inc n) 0.0)) n 1.0)
            sol (gsolve A b)
            ex (mapv (fn [i] (- (nth mu i) rf)) (range n))
            tw0 (gsolve (mapv vec S) ex)
            ts (reduce + tw0) tw (mapv #(/ % ts) tw0)
            fvar (mapv (fn [f]
                         (let [F (mapv (fn [i] (mapv (fn [j]
                                                        (cond (< i n) (cond (< j n) (nth (nth S i) j) (= j n) (nth mu i) :else 1.0)
                                                              (= i n) (cond (< j n) (nth mu j) :else 0.0)
                                                              :else (cond (< j n) 1.0 :else 0.0))) (range (+ n 2)))) (range (+ n 2)))
                               fb (-> (vec (repeat (+ n 2) 0.0)) (assoc n (nth tg f)) (assoc (inc n) 1.0))
                               w (gsolve F fb)]
                           (reduce + (mapcat (fn [i] (map (fn [j] (* (nth w i) (nth (nth S i) j) (nth w j))) (range n))) (range n))))) (range 3))
            tS (mapv (fn [i] (mapv (fn [j] (* (nth (nth S i) j) tau)) (range n))) (range n))
            iTS (ginverse tS)
            M (mapv (fn [i] (mapv (fn [j] (+ (nth (nth iTS i) j) (/ (* (nth P i) (nth P j)) om))) (range n))) (range n))
            rhs (mapv (fn [i] (+ (reduce + (map (fn [j] (* (nth (nth iTS i) j) (nth mu j))) (range n))) (/ (* (nth P i) Q) om))) (range n))
            iM (ginverse M)
            bl (mapv (fn [i] (reduce + (map (fn [j] (* (nth (nth iM i) j) (nth rhs j))) (range n)))) (range n))]
        (println (str "MINVAR:" (apply str (map (fn [i] (str " " (fmt "%.6f" (nth sol i)))) (range n)))))
        (println (str "TANGENCY:" (apply str (map (fn [x] (str " " (fmt "%.6f" x))) tw))))
        (println (str "FRONTIER:" (apply str (map (fn [f] (str " " (fmt "%.6f" (nth tg f)) "," (fmt "%.6f" (nth fvar f)))) (range 3)))))
        (println (str "BL:" (apply str (map (fn [x] (str " " (fmt "%.6f" x))) bl))))))))
(-main (slurp *in*))
