;; Chandy-Lamport snapshot simulation over recorded balances and channels.
;; Reads num-procs num-events, num-procs balances, then TRANSFER u v amt /
;; RECEIVE u v / SNAPSHOT_INIT p events.
;; Every event costs near-constant amortized time apart from marker broadcasts,
;; which run once per process, so time is O(events + M + n n) for M app messages
;; and space holds the live plus recorded traffic.
;; Blank input prints nothing. Invalid input prints a note to stderr and
;; returns normally with no stdout. Exit stays 0 since a nonzero exit would
;; need host interop, which is retired.

(require '[clojure.string :as str])

(def digit-values
  "Map from digit char to its numeric value."
  (zipmap "0123456789" (range)))

(defn whitespace?
  "True for ASCII whitespace chars."
  [ch]
  (<= (compare ch \space) 0))

(defn tokenize
  "Split string s on ASCII whitespace into a vector of token strings."
  [s]
  (if (nil? s)
    []
    (filterv seq (clojure.string/split s #"[\x00-\x20]+"))))

(defn parse-int
  "Parse an optional-sign decimal token into an integer, else nil."
  [tok]
  (when (seq tok)
    (let [chs (seq tok)
          [neg? ds] (cond (= (first chs) \-) [true (next chs)]
                          (= (first chs) \+) [false (next chs)]
                          :else [false chs])]
      (when (seq ds)
        (let [mag (reduce (fn [acc ch]
                            (when (some? acc)
                              (let [d (get digit-values ch nil)]
                                (when (some? d)
                                  (+' (*' acc 10) d)))))
                          0
                          ds)]
          (when (some? mag)
            (if neg? (-' mag) mag)))))))

(defn fail!
  "Print msg to stderr and return nil. Stdout stays empty and exit stays 0."
  [msg]
  (binding [*out* *err*]
    (println msg))
  nil)

(defn empty-queue
  "Fresh empty FIFO queue."
  []
  {:data [] :off 0})

(defn queue-empty?
  "True when q holds no messages."
  [q]
  (>= (:off q) (count (:data q))))

(defn queue-push
  "Enqueue x on q."
  [q x]
  (update q :data conj x))

(defn queue-pop
  "Dequeue from q. Returns [message next-q]."
  [q]
  [(nth (:data q) (:off q)) (update q :off inc)])

(defn valid-proc?
  "True when p names one of n processes."
  [n p]
  (and (some? p) (integer? p) (<= 0 p) (< p n)))

(defn push-chan
  "Enqueue msg on channel u to v inside state."
  [state u v msg]
  (update-in state [:chans [u v]] (fn [q] (queue-push (or q (empty-queue)) msg))))

(defn on-marker
  "Handle a marker arriving at v from u. Returns the next state."
  [{:keys [recorded balance] :as state} u v n]
  (if (contains? recorded v)
    (update state :recording disj [u v])
    (reduce (fn [st k]
              (if (= k v)
                st
                (-> st
                    (update :recording (fn [r] (if (= k u) r (conj r [k v]))))
                    (push-chan v k "MARKER"))))
            (-> state
                (update :recorded conj v)
                (assoc-in [:recbal v] (nth balance v)))
            (range n))))

(defn on-snapshot-init
  "Handle SNAPSHOT_INIT at p. Returns the next state."
  [{:keys [recorded] :as state} p n]
  (if (contains? recorded p)
    state
    (reduce (fn [st k]
              (if (= k p)
                st
                (-> st
                    (update :recording conj [k p])
                    (push-chan p k "MARKER"))))
            (-> state
                (update :recorded conj p)
                (assoc-in [:recbal p] (nth (:balance state) p)))
            (range n))))

(defn on-transfer
  "Handle TRANSFER of amt from u to v. Returns the next state."
  [state u v amt]
  (-> state
      (update-in [:balance u] - amt)
      (push-chan u v (str amt))))

(defn on-receive
  "Handle RECEIVE at v from u. Returns the next state."
  [{:keys [chans recording] :as state} u v n]
  (let [q (get chans [u v])]
    (if (or (nil? q) (queue-empty? q))
      state
      (let [[msg q2] (queue-pop q)
            state2 (assoc-in state [:chans [u v]] q2)]
        (if (= msg "MARKER")
          (on-marker state2 u v n)
          (let [amt (parse-int msg)
                state3 (update-in state2 [:balance v] + amt)]
            (if (contains? recording [u v])
              (update-in state3 [:recchans [u v]] (fn [ms] (conj (or ms []) msg)))
              state3)))))))

(defn report-lines
  "Final snapshot lines plus the conserved total."
  [{:keys [recbal recchans]} n]
  (let [rec-msgs (for [u (range n) v (range n) msg (get recchans [u v] [])]
                   {:u u :v v :msg msg})
        total (+' (reduce +' 0 recbal)
                  (reduce +' 0 (map (comp parse-int :msg) rec-msgs)))]
    (conj (into (vec (map-indexed (fn [i b] (str "P" i ": " b)) recbal))
                (map (fn [{:keys [u v msg]}] (str "CHANNEL " u "->" v ": " msg)))
                rec-msgs)
          (str "TOTAL: " total))))

(defn simulate
  "Build output lines from toks, or fail to stderr and return nil."
  [toks]
  (if (< (count toks) 2)
    (fail! "Invalid input: expected num-procs num-events ...")
    (let [n (parse-int (nth toks 0))
          n-events (parse-int (nth toks 1))]
      (cond
        (or (nil? n) (neg? n))
        (fail! "Invalid input: num-procs must be a non-negative integer")
        (or (nil? n-events) (neg? n-events))
        (fail! "Invalid input: num-events must be a non-negative integer")
        (< (count toks) (+' 2 n))
        (fail! "Invalid input: fewer balances than num-procs claims")
        :else
        (let [bals (mapv parse-int (take n (drop 2 toks)))]
          (if (some nil? bals)
            (fail! "Invalid input: balances must be integers")
            (let [init {:balance bals
                        :recorded #{}
                        :recbal (vec (repeat n 0))
                        :chans {}
                        :recchans {}
                        :recording #{}}]
              (loop [e 0, ti (+ 2 n), state init]
                (if (>= e n-events)
                  (report-lines state n)
                  (if (>= ti (count toks))
                    (fail! "Invalid input: fewer event tokens than num-events claims")
                    (let [ev (nth toks ti)]
                      (cond
                        (= ev "TRANSFER")
                        (if (>= (+ ti 3) (count toks))
                          (fail! "Invalid input: TRANSFER needs u v amt")
                          (let [u (parse-int (nth toks (+ ti 1)))
                                v (parse-int (nth toks (+ ti 2)))
                                amt (parse-int (nth toks (+ ti 3)))]
                            (cond
                              (or (not (valid-proc? n u)) (not (valid-proc? n v)))
                              (fail! "Invalid input: TRANSFER endpoint out of range")
                              (nil? amt)
                              (fail! "Invalid input: TRANSFER amount must be an integer")
                              :else
                              (recur (inc e) (+ ti 4) (on-transfer state u v amt)))))
                        (= ev "RECEIVE")
                        (if (>= (+ ti 2) (count toks))
                          (fail! "Invalid input: RECEIVE needs u v")
                          (let [u (parse-int (nth toks (+ ti 1)))
                                v (parse-int (nth toks (+ ti 2)))]
                            (if (or (not (valid-proc? n u)) (not (valid-proc? n v)))
                              (fail! "Invalid input: RECEIVE endpoint out of range")
                              (recur (inc e) (+ ti 3) (on-receive state u v n)))))
                        (= ev "SNAPSHOT_INIT")
                        (if (>= (+ ti 1) (count toks))
                          (fail! "Invalid input: SNAPSHOT_INIT needs a process")
                          (let [p (parse-int (nth toks (+ ti 1)))]
                            (if-not (valid-proc? n p)
                              (fail! "Invalid input: SNAPSHOT_INIT process out of range")
                              (recur (inc e) (+ ti 2) (on-snapshot-init state p n)))))
                        :else
                        (fail! (str "Invalid input: unknown event " ev))))))))))))))

(defn -main
  "Entry point: tokenize INPUT and print result lines. Blank input prints
   nothing. Invalid input prints a note to stderr with no stdout."
  [input]
  (when-not (str/blank? input)
    (let [lines (simulate (tokenize input))]
      (when (some? lines)
        (run! println lines)))))

(-main (slurp *in*))
