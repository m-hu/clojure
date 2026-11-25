;; Commentz-Walter Multi-Pattern Matcher (Gusfield Ch 3.5).
;; Input: N followed by N pattern strings and a text string.
;; Output: N lines, line i has space-separated 0-based start indices of P_i in text, or -1.
;; Time complexity: O(sum |P_i| + n * max |P_i|) worst-case, sublinear average; Space: O(sum |P_i|).
;; Pure clojure with no host interop.
(require '[clojure.string :as str])

(def digits
  "Digit chars mapped to their numeric values."
  (zipmap "0123456789" (range 10)))

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

(defn- insert-reversed-pattern
  "Inserts reversed pattern pat with index pid into [trie char-min-depth]."
  [[trie char-min-depth] [pid pat]]
  (let [rpat (vec (reverse pat))
        m (count rpat)]
    (loop [k 0 u 0 trie trie cmin char-min-depth]
      (if (>= k m)
        [(update-in trie [u :out] conj pid) cmin]
        (let [ch (nth rpat k)
              dep (inc k)
              cmin (if (and (>= dep 2) (< dep (get cmin ch (inc m))))
                     (assoc cmin ch dep)
                     cmin)
              nxt (get-in trie [u :next ch])]
          (if (some? nxt)
            (recur (inc k) nxt trie cmin)
            (let [new-id (count trie)
                  trie (-> trie
                           (assoc-in [u :next ch] new-id)
                           (conj {:next {} :out []}))]
              (recur (inc k) new-id trie cmin))))))))

(defn- commentz-walter
  "Runs Commentz-Walter matcher for patterns on text."
  [patterns text]
  (let [num-p (count patterns)
        empty-res (vec (repeat num-p []))
        n (count text)]
    (if (or (zero? num-p) (zero? n))
      empty-res
      (let [w-min (reduce min (map count patterns))]
        (if (or (zero? w-min) (> w-min n))
          empty-res
          (let [init-cmin (zipmap (seq "abcdefghijklmnopqrstuvwxyz")
                                  (repeat (inc w-min)))
                [trie cmin] (reduce insert-reversed-pattern
                                    [[{:next {} :out []}] init-cmin]
                                    (map-indexed vector patterns))]
            (loop [j (dec w-min) res empty-res]
              (if (>= j n)
                res
                (let [res (loop [u 0 pos j res res]
                            (let [ch (nth text pos)
                                  nxt (get-in trie [u :next ch])]
                              (if (nil? nxt)
                                res
                                (let [outs (:out (nth trie nxt))
                                      res (reduce (fn [r pid]
                                                    (update r pid conj pos))
                                                  res outs)]
                                  (if (zero? pos)
                                    res
                                    (recur nxt (dec pos) res))))))
                      shift (max 1 (dec (get cmin (nth text j) (inc w-min))))]
                  (recur (+ j shift) res))))))))))

(defn -main
  "Parses input and runs Commentz-Walter multi-pattern matcher."
  [data]
  (when-not (str/blank? data)
    (let [toks (tokenize data)
          n (parse-int (first toks) 1 2147483647)]
      (if (or (nil? n) (< (count toks) (+' n 2)))
        (report "error: expected N followed by N patterns and a text string")
        (let [patterns (mapv vec (take n (rest toks)))
              text (vec (nth toks (inc n)))
              res (commentz-walter patterns text)
              lines (map (fn [matches]
                           (if (empty? matches) "-1" (str/join " " matches)))
                         res)]
          (print (str (str/join "\n" lines) "\n")))))))

(let [data (slurp *in*)] (-main data))
