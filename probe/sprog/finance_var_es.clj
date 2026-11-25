;; var_es: Value-at-Risk (parametric delta-normal, historical simulation,
;; Monte Carlo with LCG+Box-Muller) & Expected Shortfall + Kupiec POF and
;; Christoffersen independence backtests. Acklam inverse-normal CDF.
;; Batch 6 (T382). Input: M alpha MCN, returns.
(require '[clojure.string :as str])
(def separators #"\s+")
(defn tokenize [input] (if (str/blank? input) [] (str/split (str/trim input) separators)))
(defn fmt [s & xs] (String/format java.util.Locale/US s (to-array xs)))
(defn lcg-next [x] (unchecked-add (unchecked-multiply x 6364136223846793005) 1442695040888963407))
(defn phiinv [p]
  (let [a1 -3.969683028665376e+01 a2 2.209460984245205e+02 a3 -2.759285104469687e+02
        a4 1.383577518672690e+02 a5 -3.066479806614716e+01 a6 2.506628277459239e+00
        b1 -5.447609879822406e+01 b2 1.615858368580409e+02 b3 -1.556989798598866e+02
        b4 6.680131188771972e+01 b5 -1.328068155288572e+01
        c1 -7.784894002430293e-03 c2 -3.223964580411365e-01 c3 -2.400758277161838e+00
        c4 -2.549732539343734e+00 c5 4.374664141464968e+00 c6 2.938163982698783e+00
        d1 7.784695709041462e-03 d2 3.224671290700398e-01 d3 2.445134137142996e+00 d4 3.754408661907416e+00]
    (cond (< p 0.02425)
      (let [q (Math/sqrt (* -2.0 (Math/log p)))]
        (/ (+ (* (+ (* (+ (* (+ (* (+ (* c1 q) c2) q) c3) q) c4) q) c5) q) c6)
           (+ (* (+ (* (+ (* (+ (* d1 q) d2) q) d3) q) d4) q) 1.0)))
      (<= p 0.97575)
      (let [q (- p 0.5) r (* q q)]
        (/ (* (+ (* (+ (* (+ (* (+ (* (+ (* a1 r) a2) r) a3) r) a4) r) a5) r) a6) q)
           (+ (* (+ (* (+ (* (+ (* (+ (* b1 r) b2) r) b3) r) b4) r) b5) r) 1.0)))
      :else
      (let [q (Math/sqrt (* -2.0 (Math/log (- 1.0 p))))]
        (- (/ (+ (* (+ (* (+ (* (+ (* (+ (* c1 q) c2) q) c3) q) c4) q) c5) q) c6)
              (+ (* (+ (* (+ (* (+ (* d1 q) d2) q) d3) q) d4) q) 1.0)))))))
(defn tlog [a b] (if (zero? b) 0.0 (* b (Math/log a))))
(defn -main [input]
  (let [toks (tokenize input)]
    (when (seq toks)
      (let [M (Long/parseLong (nth toks 0)) alpha (Double/parseDouble (nth toks 1))
            MCN (Long/parseLong (nth toks 2))
            r (mapv (fn [k] (Double/parseDouble (nth toks (+ 3 k)))) (range M))
            mu (/ (reduce + r) M)
            vv (/ (reduce + (map (fn [x] (let [d (- x mu)] (* d d))) r)) (dec M))
            sd (Math/sqrt vv)
            z (phiinv alpha)
            pvar (- (+ mu (* sd z)))
            pes (- (/ (* sd (Math/exp (/ (* (- z) z) 2.0))) (Math/sqrt (* 2.0 Math/PI)) alpha) mu)
            s (sort r)
            idx (max 0 (dec (long (Math/ceil (* alpha M)))))
            hvar (- (nth s idx))
            hes (- (/ (reduce + (take (inc idx) s)) (inc idx)))
            sim (loop [lcg 42 out []]
                  (if (>= (count out) MCN) out
                    (let [l1 (lcg-next lcg) u1 (/ (double (inc (unsigned-bit-shift-right l1 11))) 9007199254740992.0)
                          l2 (lcg-next l1) u2 (/ (double (inc (unsigned-bit-shift-right l2 11))) 9007199254740992.0)
                          rad (Math/sqrt (* -2.0 (Math/log u1)))
                          z0 (+ mu (* sd rad (Math/cos (* 2.0 Math/PI u2))))
                          z1 (+ mu (* sd rad (Math/sin (* 2.0 Math/PI u2))))]
                      (recur l2 (if (>= (inc (count out)) MCN) (conj out z0) (conj out z0 z1))))))
            ss (sort sim)
            j (max 0 (dec (long (Math/ceil (* alpha MCN)))))
            mvar (- (nth ss j))
            mes (- (/ (reduce + (take (inc j) ss)) (inc j)))
            hit (mapv (fn [x] (if (< x (- hvar)) 1 0)) r)
            x (reduce + hit)
            kup (+ (* -2.0 (+ (tlog (- 1.0 alpha) (- M x)) (tlog alpha x)))
                   (* 2.0 (+ (tlog (- 1.0 (/ (double x) M)) (- M x)) (tlog (/ (double x) M) x))))
            tr (map (fn [i] [(nth hit (dec i)) (nth hit i)]) (range 1 M))
            n00 (count (filter #(= % [0 0]) tr)) n01 (count (filter #(= % [0 1]) tr))
            n10 (count (filter #(= % [1 0]) tr)) n11 (count (filter #(= % [1 1]) tr))
            pi0 (if (pos? (+ n00 n01)) (/ (double n01) (+ n00 n01)) 0.0)
            pi1 (if (pos? (+ n10 n11)) (/ (double n11) (+ n10 n11)) 0.0)
            pi (/ (double (+ n01 n11)) (dec M))
            chr (+ (* -2.0 (+ (tlog (- 1.0 pi) (+ n00 n10)) (tlog pi (+ n01 n11))))
                   (* 2.0 (+ (tlog (- 1.0 pi0) n00) (tlog pi0 n01) (tlog (- 1.0 pi1) n10) (tlog pi1 n11))))]
        (println (str "PVAR: " (fmt "%.6f" pvar) " PES: " (fmt "%.6f" pes)))
        (println (str "HVAR: " (fmt "%.6f" hvar) " HES: " (fmt "%.6f" hes)))
        (println (str "MVAR: " (fmt "%.6f" mvar) " MES: " (fmt "%.6f" mes)))
        (println (str "KUPIEC: " (fmt "%.6f" kup)))
        (println (str "CHRIST: " (fmt "%.6f" chr)))))))
(-main (slurp *in*))
