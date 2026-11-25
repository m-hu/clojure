;; feature_importance: MDI (max single-split variance reduction per
;; feature), MDA (R^2 drop under deterministic cyclic shift of the column
;; for model Xw) and SFI (single-feature OLS R^2). Batch 6 (T370).
;; Input: n p, X rows, y, model weights w.
(require '[clojure.string :as str])
(def separators #"\s+")
(defn tokenize [input] (if (str/blank? input) [] (str/split (str/trim input) separators)))
(defn fmt [s & xs] (String/format java.util.Locale/US s (to-array xs)))
(defn r2score [y p]
  (let [n (count y) mu (/ (reduce + y) n)
        sse (reduce + (map (fn [a b] (let [d (- a b)] (* d d))) y p))
        sst (reduce + (map (fn [a] (let [d (- a mu)] (* d d))) y))]
    (if (zero? sst) (if (zero? sse) 1.0 0.0) (- 1.0 (/ sse sst)))))
(defn mdi-one [X y j n tot]
  (let [idx (sort-by (fn [i] [(nth (nth X i) j) i]) (range n))]
    (reduce max 0.0
      (map (fn [s]
             (if (= (nth (nth X (nth idx s)) j) (nth (nth X (nth idx (dec s))) j)) 0.0
             (let [nl s nr (- n s)
                   ml (/ (reduce + (map #(nth y %) (take s idx))) nl)
                   mr (/ (reduce + (map #(nth y %) (drop s idx))) nr)
                   sl (reduce + (map (fn [t] (let [d (- (nth y t) ml)] (* d d))) (take s idx)))
                   sr (reduce + (map (fn [t] (let [d (- (nth y t) mr)] (* d d))) (drop s idx)))]
               (- tot sl sr)))) (range 1 n)))))
(defn -main [input]
  (let [toks (tokenize input)]
    (when (seq toks)
      (let [n (Long/parseLong (nth toks 0)) p (Long/parseLong (nth toks 1))
            X (mapv (fn [r] (mapv (fn [c] (Double/parseDouble (nth toks (+ 2 (+ (* r p) c))))) (range p))) (range n))
            y (mapv (fn [i] (Double/parseDouble (nth toks (+ 2 (* n p) i)))) (range n))
            w (mapv (fn [j] (Double/parseDouble (nth toks (+ 2 (* n p) n j)))) (range p))
            mu (/ (reduce + y) n)
            tot (reduce + (map (fn [a] (let [d (- a mu)] (* d d))) y))
            base (mapv (fn [r] (reduce + (map * (nth X r) w))) (range n))
            r2b (r2score y base)
            mdi (mapv (fn [j] (mdi-one X y j n tot)) (range p))
            mda (mapv (fn [j]
                        (let [pr (mapv (fn [r]
                                         (reduce + (map (fn [k]
                                                          (* (if (= k j) (nth (nth X (mod (inc r) n)) k) (nth (nth X r) k))
                                                             (nth w k))) (range p)))) (range n))]
                          (- r2b (r2score y pr)))) (range p))
            sfi (mapv (fn [j]
                        (let [mx (/ (reduce + (map #(nth % j) X)) n)
                              cov (reduce + (map (fn [r] (* (- (nth (nth X r) j) mx) (- (nth y r) mu))) (range n)))
                              vv (reduce + (map (fn [r] (let [d (- (nth (nth X r) j) mx)] (* d d))) (range n)))
                              slope (if (zero? vv) 0.0 (/ cov vv))
                              icept (- mu (* slope mx))
                              pr (mapv (fn [r] (+ icept (* slope (nth (nth X r) j)))) (range n))]
                          (r2score y pr))) (range p))]
        (println (str "MDI:" (apply str (map (fn [v] (str " " (fmt "%.6f" v))) mdi))))
        (println (str "MDA:" (apply str (map (fn [v] (str " " (fmt "%.6f" v))) mda))))
        (println (str "SFI:" (apply str (map (fn [v] (str " " (fmt "%.6f" v))) sfi))))))))
(-main (slurp *in*))
