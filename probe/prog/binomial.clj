(require '[clojure.string :as str])
(defn fmt [s & xs] (String/format java.util.Locale/US s (to-array xs)))
(defn -main [input]
  (when-not (str/blank? input)
    (let [toks (str/split (str/trim input) #"\s+")
          S (Double/parseDouble (nth toks 0)) K (Double/parseDouble (nth toks 1)) T (Double/parseDouble (nth toks 2))
          r (Double/parseDouble (nth toks 3)) sig (Double/parseDouble (nth toks 4)) N (Integer/parseInt (nth toks 5))
          tp (nth toks 6) style (nth toks 7)
          dt (/ T N) u (Math/exp (* sig (Math/sqrt dt))) d (/ 1.0 u) disc (Math/exp (- (* r dt)))
          p-raw (/ (- (Math/exp (* r dt)) d) (- u d)) p (max 0.0 (min 1.0 p-raw))
          init-vals (mapv (fn [j] (let [ST (* S (Math/pow u j) (Math/pow d (- N j)))] (if (= tp "call") (max (- ST K) 0.0) (max (- K ST) 0.0)))) (range (inc N)))
          result (loop [vals init-vals step (dec N)]
                   (if (< step 0) vals
                     (let [nvals (mapv (fn [j] (let [hold (* disc (+ (* p (nth vals (inc j))) (* (- 1.0 p) (nth vals j))))]
                                                  (if (= style "american")
                                                    (let [ST (* S (Math/pow u j) (Math/pow d (- step j))) intr (if (= tp "call") (max (- ST K) 0.0) (max (- K ST) 0.0))]
                                                      (max hold intr)) hold))) (range (inc step)))]
                       (recur (vec (concat nvals (repeat (- (count vals) (count nvals)) 0.0))) (dec step)))))]
      (println (fmt "%.4f" (nth result 0))))))
(-main (slurp *in*))
