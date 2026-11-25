;; jones_mueller: Jones & Mueller-Stokes polarization calculus — cascade of
;; polarizers (P), waveplates (W) and rotators (R); system Jones J, Mueller
;; M = A(J x J*)A^-1, output Jones and Stokes vectors. Batch 6 (T389).
(require '[clojure.string :as str])
(def separators #"\s+")
(defn tokenize [input] (if (str/blank? input) [] (str/split (str/trim input) separators)))
(defn fmt [s & xs] (String/format java.util.Locale/US s (to-array xs)))
(defn sn [x] (if (< (Math/abs (double x)) 5e-7) 0.0 x))
(defn cmul [a b] [(- (* (nth a 0) (nth b 0)) (* (nth a 1) (nth b 1)))
                  (+ (* (nth a 0) (nth b 1)) (* (nth a 1) (nth b 0)))])
(defn cadd [a b] [(+ (nth a 0) (nth b 0)) (+ (nth a 1) (nth b 1))])
(defn mmul [A B n]
  (mapv (fn [i] (mapv (fn [j]
                (loop [k 0 s [0.0 0.0]]
                  (if (>= k n) s (recur (inc k) (cadd s (cmul (nth (nth A i) k) (nth (nth B k) j)))))))
              (range n))) (range n)))
(defn -main [input]
  (let [toks (tokenize input)]
    (when (seq toks)
      (let [PI Math/PI
            pos (atom 0)
            nxt (fn [] (let [t (nth toks @pos)] (swap! pos inc) t))
            E (Long/parseLong (nxt))
            J (loop [e 0 J [[[1.0 0.0] [0.0 0.0]] [[0.0 0.0] [1.0 0.0]]]]
                (if (>= e E) J
                  (let [kind (nxt)
                        T (cond (= kind "P") (let [a (* (Double/parseDouble (nxt)) PI (/ 1.0 180.0))
                                                      c (Math/cos a) s (Math/sin a)]
                                                  [[[(double (* c c)) 0.0] [(double (* c s)) 0.0]]
                                                   [[(double (* c s)) 0.0] [(double (* s s)) 0.0]]])
                                (= kind "W") (let [d (* (Double/parseDouble (nxt)) PI (/ 1.0 180.0))
                                                   a (* (Double/parseDouble (nxt)) PI (/ 1.0 180.0))
                                                   c (Math/cos a) s (Math/sin a)
                                                   h (/ d 2.0)
                                                   R [[[c 0.0] [s 0.0]] [[(- s) 0.0] [c 0.0]]]
                                                   RI [[[c 0.0] [(- s) 0.0]] [[s 0.0] [c 0.0]]]
                                                   D [[[(Math/cos h) (Math/sin h)] [0.0 0.0]]
                                                      [[0.0 0.0] [(Math/cos h) (- (Math/sin h))]]]]
                                               (mmul RI (mmul D R 2) 2))
                                :else (let [a (* (Double/parseDouble (nxt)) PI (/ 1.0 180.0))
                                            c (Math/cos a) s (Math/sin a)]
                                        [[[c 0.0] [(- s) 0.0]] [[s 0.0] [c 0.0]]]))]
                    (recur (inc e) (mmul T J 2)))))
            exr (Double/parseDouble (nxt)) exi (Double/parseDouble (nxt))
            eyr (Double/parseDouble (nxt)) eyi (Double/parseDouble (nxt))
            S (mapv (fn [_] (Double/parseDouble (nxt))) (range 4))
            ox (cadd (cmul (nth (nth J 0) 0) [exr exi]) (cmul (nth (nth J 0) 1) [eyr eyi]))
            oy (cadd (cmul (nth (nth J 1) 0) [exr exi]) (cmul (nth (nth J 1) 1) [eyr eyi]))
            K (mapv (fn [i] (mapv (fn [j]
                    (let [ii (quot i 2) kk (rem i 2) jj (quot j 2) ll (rem j 2)
                          z (nth (nth J kk) ll)]
                      (cmul (nth (nth J ii) jj) [(nth z 0) (- (nth z 1))]))) (range 4))) (range 4))
            ad [[1.0 0.0 0.0 1.0] [1.0 0.0 0.0 -1.0] [0.0 1.0 1.0 0.0] [0.0 0.0 0.0 0.0]]
            A (mapv (fn [i] (mapv (fn [j]
                    (cond (and (= i 3) (= j 1)) [0.0 1.0] (and (= i 3) (= j 2)) [0.0 -1.0]
                          :else [(nth (nth ad i) j) 0.0])) (range 4))) (range 4))
            AI [[[0.5 0.0] [0.5 0.0] [0.0 0.0] [0.0 0.0]]
                [[0.0 0.0] [0.0 0.0] [0.5 0.0] [0.0 -0.5]]
                [[0.0 0.0] [0.0 0.0] [0.5 0.0] [0.0 0.5]]
                [[0.5 0.0] [-0.5 0.0] [0.0 0.0] [0.0 0.0]]]
            M (mmul A (mmul K AI 4) 4)
            O (mapv (fn [i] (loop [j 0 s 0.0] (if (>= j 4) s (recur (inc j) (+ s (* (nth (nth (nth M i) j) 0) (nth S j))))))) (range 4))
            jf (fn [z] (fmt " %.6f %.6f" (sn (+ (nth z 0) 0.0)) (sn (+ (nth z 1) 0.0))))]
        (println (str "JONES:" (jf (nth (nth J 0) 0)) (jf (nth (nth J 0) 1)) (jf (nth (nth J 1) 0)) (jf (nth (nth J 1) 1))))
        (println (str "MUELLER:" (apply str (map (fn [i] (apply str (map (fn [j] (fmt " %.6f" (sn (+ (nth (nth (nth M i) j) 0) 0.0)))) (range 4)))) (range 4)))))
        (println (fmt "OUT_J: %.6f %.6f %.6f %.6f" (sn (+ (nth ox 0) 0.0)) (sn (+ (nth ox 1) 0.0)) (sn (+ (nth oy 0) 0.0)) (sn (+ (nth oy 1) 0.0))))
        (println (str "OUT_S:" (apply str (map (fn [i] (fmt " %.6f" (sn (+ (nth O i) 0.0)))) (range 4)))))))))
(-main (slurp *in*))
