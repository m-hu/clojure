(require '[clojure.string :as str])
;; lzw_compression: LZW dictionary coding via ternary trie expansion.
;; Pure Clojure with canonical input parsing.

(def digit-values
  (zipmap "0123456789" (range)))

(def separators
  #{\space \tab \newline \return \formfeed \u000B})

(defn tokenize
  [s]
  (if (nil? s)
    []
    (filterv seq (clojure.string/split s #"[ \t\n\r\f\x0B]+"))))

(def min-long (bit-shift-left 1 63))
(def max-long (bit-not min-long))

(defn parse-token
  [tok]
  (when (and (string? tok) (re-matches #"[+-]?[0-9]+" tok))
    (parse-long tok)))

(defn lzw-encode [s]
  (let [init-d (into {} (for [i (range 256)] [(str (char i)) i]))]
    (loop [chars (seq s) w "" d init-d nxt 256 codes []]
      (if (empty? chars)
        (if (seq w) (conj codes (get d w)) codes)
        (let [c (str (first chars)) wc (str w c)]
          (if (contains? d wc)
            (recur (rest chars) wc d nxt codes)
            (recur (rest chars) c (assoc d wc nxt) (inc nxt) (conj codes (get d w)))))))))
(defn -main [input]
  (let [toks (filter seq (str/split input #"\s+")) s (first toks)]
    (when (seq s)
      (println (str/join " " (lzw-encode s))))))
(-main (slurp *in*))