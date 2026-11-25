;; Ukkonen online suffix tree with distinct substring count and
;; substring queries.
;; Input: text token, then an optional query count with that many
;; pattern tokens. Output: node count, distinct substring count,
;; then YES or NO per query.
;; Linear time construction in the text length with linear space,
;; linear time per query in the pattern length, which is optimal.
;; Pure clojure with no host interop.
(require '[clojure.string :as str])

(def digits
  "Digit chars mapped to their numeric values."
  (zipmap "0123456789" (range 10)))

(defn- blank-char?
  "True for the whitespace chars that split tokens."
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

(defn- node-end
  "End index of node, resolving open leaves to leaf-end."
  [node leaf-end]
  (if (:leaf node) leaf-end (:end node)))

(defn- edge-len
  "Length of the edge into node u with leaves open to leaf-end."
  [nodes u leaf-end]
  (if (zero? u)
    0
    (let [nd (nth nodes u)]
      (inc (- (node-end nd leaf-end) (:start nd))))))

(defn- add-node
  "Appends a node and returns [nodes new-index]."
  [nodes start end leaf?]
  [(conj nodes {:link 0 :start start :end end :leaf leaf? :next {}})
   (count nodes)])

(defn- follow-link
  "Active point step after an extension, following suffix links."
  [nodes active edge alen pos rem]
  (cond
    (and (zero? active) (pos? alen)) [active (inc (- pos rem)) (dec alen)]
    (zero? active) [active edge alen]
    :else [(:link (nth nodes active)) edge alen]))

(defn- build-tree
  "Suffix tree nodes for char vector t with the online method."
  [t]
  (let [n (count t)]
    (loop [pos 0
           nodes [{:link 0 :start 0 :end -1 :leaf false :next {}}]
           active 0 edge 0 alen 0 rem 0]
      (if (>= pos n)
        nodes
        (let [[nodes active edge alen rem]
              (loop [nodes nodes active active edge edge
                     alen alen rem (inc rem) last-new -1]
                (if (zero? rem)
                  [nodes active edge alen rem]
                  (let [edge (if (zero? alen) pos edge)
                        cur-char (nth t edge)
                        nxt (get (:next (nth nodes active)) cur-char)]
                    (if (nil? nxt)
                      (let [[nodes leaf] (add-node nodes pos -1 true)
                            nodes (assoc-in nodes [active :next cur-char] leaf)
                            nodes (if (not= last-new -1)
                                    (assoc-in nodes [last-new :link] active)
                                    nodes)
                            rem (dec rem)
                            [active edge alen]
                            (follow-link nodes active edge alen pos rem)]
                        (recur nodes active edge alen rem -1))
                      (let [elen (edge-len nodes nxt pos)]
                        (if (>= alen elen)
                          (recur nodes nxt (+ edge elen) (- alen elen)
                                 rem last-new)
                          (let [epos (+ (:start (nth nodes nxt)) alen)]
                            (if (= (nth t epos) (nth t pos))
                              [(if (and (not= last-new -1) (not (zero? active)))
                                 (assoc-in nodes [last-new :link] active)
                                 nodes)
                               active edge (inc alen) rem]
                              (let [split-end (+ (:start (nth nodes nxt)) alen -1)
                                    [nodes split]
                                    (add-node nodes (:start (nth nodes nxt))
                                              split-end false)
                                    [nodes leaf] (add-node nodes pos -1 true)
                                    nodes (-> nodes
                                              (assoc-in [active :next cur-char] split)
                                              (assoc-in [split :next (nth t pos)] leaf)
                                              (update-in [nxt :start] + alen)
                                              (assoc-in [split :next (nth t epos)] nxt))
                                    nodes (if (not= last-new -1)
                                            (assoc-in nodes [last-new :link] split)
                                            nodes)
                                    rem (dec rem)
                                    [active edge alen]
                                    (follow-link nodes active edge alen pos rem)]
                                (recur nodes active edge alen rem split))))))))))]
          (recur (inc pos) nodes active edge alen rem))))))

(defn- count-distinct
  "Count of distinct substrings by summing edge lengths."
  [nodes leaf-end]
  (loop [stack [0] acc 0]
    (if (empty? stack)
      acc
      (let [u (peek stack)
            kids (vals (:next (nth nodes u)))]
        (recur (into (pop stack) kids)
               (reduce + acc (map #(edge-len nodes % leaf-end) kids)))))))

(defn- find-pattern
  "True when char vector pat occurs in the tree text."
  [nodes t pat leaf-end]
  (let [m (count pat)]
    (loop [cur 0 idx 0]
      (if (>= idx m)
        true
        (let [nxt (get (:next (nth nodes cur)) (nth pat idx))]
          (if (nil? nxt)
            false
            (let [start (:start (nth nodes nxt))
                  len (edge-len nodes nxt leaf-end)
                  res (loop [k 0 j idx]
                        (cond
                          (or (>= k len) (>= j m)) (- j idx)
                          (= (nth t (+ start k)) (nth pat j))
                          (recur (inc k) (inc j))
                          :else -1))]
              (if (= res -1)
                false
                (recur nxt (+ idx res))))))))))

(defn -main
  "Parses data and prints tree stats plus query answers.
  Malformed input prints a message on stderr and returns normally,
  leaving the exit code at zero since nonzero exit needs host calls."
  [data]
  (when-not (str/blank? data)
    (try
      (let [toks (tokenize data)
            t (vec (first toks))
            nodes (build-tree t)
            leaf-end (dec (count t))
            head [(str "Nodes: " (count nodes))
                  (str "Distinct substrings: " (count-distinct nodes leaf-end))]
            qr (next toks)]
        (if (nil? qr)
          (print (str (str/join "\n" head) "\n"))
          (let [q (parse-int (first qr) -2147483648 2147483647)]
            (if (nil? q)
              (print (str (str/join "\n" head) "\n"))
              (let [want (max 0 q)
                    pats (take want (rest qr))]
                (if (< (count pats) want)
                  (do
                    (print (str (str/join "\n" head) "\n"))
                    (report "error: expected one pattern per query"))
                  (let [tails (map #(if (find-pattern nodes t (vec %) leaf-end)
                                      "YES" "NO")
                                   pats)]
                    (print (str (str/join "\n" (concat head tails)) "\n")))))))))
      (catch Exception _ (report "error: invalid input")))))

(let [data (slurp *in*)] (-main data))
