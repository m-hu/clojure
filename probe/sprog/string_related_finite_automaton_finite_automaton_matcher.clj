;; Finite-Automaton String Matcher (CLRS Ch 32.3).
;; Input: a count Q followed by Q pairs of <pattern> <text> lowercase tokens.
;; Output: space-separated 0-based starting indices of matches per query, or -1.
;; Time complexity: O(m |Sigma| + n) per query; Space complexity: O(m |Sigma|).
;; Pure clojure with no host interop.
(require '[clojure.string :as str])

(def digits
  "Digit chars mapped to their numeric values."
  (zipmap "0123456789" (range 10)))

(def alphabet
  "Lowercase ASCII alphabet chars."
  (vec "abcdefghijklmnopqrstuvwxyz"))

(def char-idx
  "Lowercase ASCII chars mapped to 0..25."
  (zipmap alphabet (range 26)))

(defn- blank-char?
  "True for whitespace chars that split tokens."
  [c]
  (or (= c \space) (= c \tab) (= c \newline) (= c \return) (= c \formfeed)))

(defn- tokenize
  "Splits s on whitespace; returns a vector of token strings."
  [s]
  (if (nil? s)
    []
    (filterv seq (clojure.string/split s #"[ \t\n\r\f]+"))))

(defn- parse-int
  "Reads token tok as a decimal integer inside lo to hi, else nil."
  [tok lo hi]
  (when (and (string? tok) (seq tok))
    (let [cs (seq tok)
          lead (first cs)
          signed (or (= lead \-) (= lead \+))
          ds (if signed (next cs) cs)]
      (when (seq ds)
        (let [mag (reduce (fn [acc c]
                            (let [d (get digits c)]
                              (if (or (nil? acc) (nil? d))
                                nil
                                (+' (*' acc 10) d))))
                          0 ds)]
          (when (some? mag)
            (let [v (if (= lead \-) (- mag) mag)]
              (when (and (<= lo v) (<= v hi)) v))))))))

(defn- report
  "Prints msg on stderr."
  [msg]
  (binding [*out* *err*] (println msg)))

(defn- prefix-table
  "Computes KMP prefix function pi for char vector pat in O(m) time."
  [pat]
  (let [m (count pat)]
    (loop [i 1 j 0 pi (transient (vec (repeat m 0)))]
      (if (>= i m)
        (persistent! pi)
        (let [j (loop [j j]
                  (if (and (pos? j) (not= (nth pat i) (nth pat j)))
                    (recur (nth pi (dec j)))
                    j))
              j (if (= (nth pat i) (nth pat j)) (inc j) j)]
          (assoc! pi i j)
          (recur (inc i) j pi))))))

(defn- compute-transition-function
  "Builds transition table delta[0..m][0..25] in O(m |Sigma|) time."
  [pat]
  (let [m (count pat)
        pi (prefix-table pat)]
    (loop [q 0 delta []]
      (if (> q m)
        delta
        (let [row (mapv (fn [c]
                          (let [ch (nth alphabet c)]
                            (cond
                              (and (< q m) (= (nth pat q) ch)) (inc q)
                              (zero? q) 0
                              :else (nth (nth delta (nth pi (dec q))) c))))
                        (range 26))]
          (recur (inc q) (conj delta row)))))))

(defn- finite-automaton-matcher
  "Returns vector of 0-based match indices of pat in text."
  [pat text]
  (let [m (count pat)
        n (count text)
        delta (compute-transition-function pat)]
    (loop [i 0 q 0 acc []]
      (if (>= i n)
        acc
        (let [c (get char-idx (nth text i) 0)
              nq (nth (nth delta q) c)
              acc (if (= nq m) (conj acc (- (inc i) m)) acc)]
          (recur (inc i) nq acc))))))

(defn -main
  "Parses input and runs finite-automaton string matcher on each query."
  [data]
  (when-not (str/blank? data)
    (let [toks (tokenize data)
          q (parse-int (first toks) 0 2147483647)]
      (if (or (nil? q) (< (count toks) (inc' (*' 2 q))))
        (report "error: expected Q followed by Q pattern and text pairs")
        (let [lines (map (fn [[pat text]]
                           (let [res (finite-automaton-matcher (vec pat) (vec text))]
                             (if (empty? res)
                               "-1"
                               (str/join " " res))))
                         (partition 2 (take (* 2 q) (rest toks))))]
          (when (seq lines)
            (print (str (str/join "\n" lines) "\n"))))))))

(let [data (slurp *in*)] (-main data))
