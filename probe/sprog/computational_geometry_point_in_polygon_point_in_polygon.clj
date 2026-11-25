;; Ray Casting & Winding Number Point-in-Polygon (de Berg Ch 6).
;;
;; Time: O(N * Q) for N polygon vertices and Q query points.
;; Space: O(N) to store the polygon vertices.
;; Purity: core Clojure only, zero host interop; exact integer cross products.
;; Robustness: blank stdin prints nothing; malformed input prints an error
;;   message to *err* and exits normally.
(ns computational-geometry.point-in-polygon.point-in-polygon
  (:require [clojure.string :as str]))

(def digit-values
  "Digit character to numeric value."
  (zipmap "0123456789" (range)))

(defn whitespace?
  "True for ASCII whitespace separating input tokens."
  [ch]
  (or (= ch \space) (= ch \tab) (= ch \newline) (= ch \return)
      (= ch \formfeed) (= ch (char 11))))

(defn tokenize
  "Split s on ASCII whitespace."
  [s]
  (if (nil? s)
    []
    (filterv seq (clojure.string/split s #"[ \t\n\r\f\x0B]+"))))

(defn parse-long-token
  "Parse a signed 64-bit integer token; nil when malformed or out of range."
  [tok]
  (when (some? tok)
    (let [cs (vec tok)
          n (count cs)]
      (when (pos? n)
        (let [c0 (nth cs 0)
              neg (or (= c0 \-) (= c0 \+))
              want-neg (= c0 \-)
              digits (subvec cs (if neg 1 0))]
          (when (and (seq digits) (every? digit-values digits))
            (let [stripped (vec (drop-while #(= % \0) digits))
                  len (count stripped)
                  lim (if want-neg "9223372036854775808" "9223372036854775807")]
              (when (or (< len 19)
                        (and (= len 19) (<= (compare (apply str stripped) lim) 0)))
                (let [v (reduce (fn [acc ch] (- (* acc 10) (digit-values ch)))
                                0 stripped)]
                  (if want-neg v (- v)))))))))))

(defn cross
  "Twice the signed area of triangle a-b-p using arbitrary-precision math."
  [[ax ay] [bx by] [px py]]
  (-' (*' (-' bx ax) (-' py ay)) (*' (-' by ay) (-' px ax))))

(defn on-segment?
  "True when p lies on closed segment a-b."
  [[ax ay :as a] [bx by :as b] [px py :as p]]
  (and (zero? (cross a b p))
       (<= (min ax bx) px (max ax bx))
       (<= (min ay by) py (max ay by))))

(defn classify-point
  "Return BOUNDARY, INSIDE, or OUTSIDE for query point p against polygon poly."
  [poly [_ py :as p]]
  (let [n (count poly)]
    (loop [i 0 winding 0]
      (if (= i n)
        (if (zero? winding) "OUTSIDE" "INSIDE")
        (let [[_ ay :as a] (nth poly i)
              [_ by :as b] (nth poly (rem (inc i) n))]
          (if (on-segment? a b p)
            "BOUNDARY"
            (let [c (cross a b p)
                  dw (cond
                       (and (<= ay py) (> by py) (pos? c)) 1
                       (and (> ay py) (<= by py) (neg? c)) -1
                       :else 0)]
              (recur (inc i) (+ winding dw)))))))))

(defn die
  "Print error message to *err* and return nil."
  [msg]
  (binding [*out* *err*]
    (println msg)))

(defn -main
  [& _args]
  (let [content (slurp *in*)]
    (when-not (str/blank? content)
      (let [tokens (tokenize content)
            n (parse-long-token (first tokens))
            q (parse-long-token (second tokens))]
        (cond
          (or (nil? n) (nil? q))
          (die "error: expected integer N and Q header")

          (or (< n 3) (neg? q))
          (die "error: requires N >= 3 and Q >= 0")

          (< (count (drop 2 tokens)) (* 2 (+ n q)))
          (die "error: insufficient coordinate tokens")

          :else
          (let [coords (take (* 2 (+ n q)) (drop 2 tokens))
                pairs (vec (for [[xs ys] (partition 2 coords)]
                             [(parse-long-token xs) (parse-long-token ys)]))]
            (if (some (fn [[x y]] (or (nil? x) (nil? y))) pairs)
              (die "error: malformed coordinate token")
              (let [poly (subvec pairs 0 n)
                    queries (subvec pairs n (+ n q))]
                (doseq [pt queries]
                  (println (classify-point poly pt)))))))))))

(-main)
