;; garch: EWMA (RiskMetrics) & GARCH(1,1) volatility filtering; GARCH
;; parameters by variance-targeted grid MLE (alpha 0.01..0.30, beta
;; 0.60..0.95, first maximum). Batch 6 (T383). Input: M lambda, returns.
(require '[clojure.string :as str])
(def separators #"\s+")
(defn tokenize [input] (if (str/blank? input) [] (str/split (str/trim input) separators)))
(defn fmt [s & xs] (String/format java.util.Locale/US s (to-array xs)))
(defn -main [input]
  (let [toks (tokenize input)]
    (when (seq toks)
      (let [M (Long/parseLong (nth toks 0)) lam (Double/parseDouble (nth toks 1))
            r (mapv (fn [k] (Double/parseDouble (nth toks (+ 2 k)))) (range M))
            mu (/ (reduce + r) M)
            vr (/ (reduce + (map (fn [x] (let [d (- x mu)] (* d d))) r)) (dec M))
            ewma (loop [t 0 v vr] (if (>= t M) v (recur (inc t) (+ (* lam v) (* (- 1.0 lam) (nth r t) (nth r t))))))
            res (loop [ai 1 bom 0.0 ba 0.0 bb 0.0 bll -1e300 bgf vr]
                  (if (> ai 30) [bom ba bb bll bgf]
                    (let [a (* ai 0.01)
                          r2 (loop [bi 60 o bom x ba y bb l bll g bgf]
                               (if (> bi 95) [o x y l g]
                                 (let [b (* bi 0.01)]
                                   (if (>= (+ a b) 1.0) (recur (inc bi) o x y l g)
                                     (let [om (* vr (- 1.0 a b))
                                           [vv ll] (loop [t 0 vv vr ll 0.0]
                                                     (if (>= t M) [vv ll]
                                                       (recur (inc t)
                                                         (+ om (* a (nth r t) (nth r t)) (* b vv))
                                                         (+ ll (* -0.5 (+ (Math/log vv) (/ (* (nth r t) (nth r t)) vv)))))))]
                                       (if (> ll l)
                                         (recur (inc bi) om a b ll vv)
                                         (recur (inc bi) o x y l g)))))))]
                      (recur (inc ai) (nth r2 0) (nth r2 1) (nth r2 2) (nth r2 3) (nth r2 4)))))]
        (println (str "EWMA: " (fmt "%.6f" ewma)))
        (println (str "GARCH: " (fmt "%.6f" (nth res 0)) " " (fmt "%.6f" (nth res 1)) " " (fmt "%.6f" (nth res 2)) " " (fmt "%.6f" (nth res 3))))
        (println (str "GFILTER: " (fmt "%.6f" (nth res 4))))))))
(-main (slurp *in*))
