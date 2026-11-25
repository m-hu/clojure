;; https://www.hackerrank.com/challenges/count-strings
;; Count strings of length l matching a regex over a and b symbols.
;; Input: a count t then t pairs of <regex> <l> tokens.
;; Output: one answer per line, modulo 1000000007.
(require '[clojure.string :as str])

(def MOD 1000000007)

(def digits
  "Digit chars mapped to their numeric values."
  (zipmap "0123456789" (range 10)))

(defn- tokenize
  "Splits s on whitespace; returns a vector of token strings."
  [s]
  (if (nil? s)
    []
    (let [v (str/split s #"[ \t\n\r\f]+")
          n (count v)
          start (if (and (pos? n) (str/blank? (nth v 0))) 1 0)
          end (if (and (> n start) (str/blank? (nth v (dec n)))) (dec n) n)]
      (subvec v start end))))

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

(defn- attach
  "Appends node to kids, folding into an open union when present."
  [kids node]
  (let [n (count kids)]
    (if (and (pos? n)
             (let [b (peek kids)]
               (and (= (:op b) :union) (= (count (:kids b)) 1))))
      (assoc kids (dec n) (update (peek kids) :kids conj node))
      (conj kids node))))

(defn- parse-regex
  "Parses regex string r into nodes with op, ch and kids fields."
  [r]
  (let [n (count r)]
    (loop [i 0 root [] stack []]
      (if (>= i n)
        {:op :concat :ch nil :kids root}
        (let [c (nth r i)]
          (cond
            (= c \()
            (recur (inc i) root (conj stack []))

            (= c \))
            (let [cur {:op :concat :ch nil :kids (peek stack)}
                  stack (pop stack)]
              (if (seq stack)
                (recur (inc i) root
                       (assoc stack (dec (count stack)) (attach (peek stack) cur)))
                (recur (inc i) (attach root cur) stack)))

            :else
            (let [in-par (boolean (seq stack))
                  kids (if in-par (peek stack) root)
                  nk (cond
                       (= c \*)
                       (assoc kids (dec (count kids))
                              (assoc (peek kids) :op :closure))
                       (= c \|)
                       (assoc kids (dec (count kids))
                              {:op :union :ch nil :kids [(peek kids)]})
                       :else
                       (attach kids {:op :concat :ch c :kids []}))]
              (if in-par
                (recur (inc i) root (assoc stack (dec (count stack)) nk))
                (recur (inc i) nk stack)))))))))

(defn- build-nfa
  "Thompson NFA for parsed root. Returns adjacency plus in and out
  states. Edge labels are 0 for empty, 1 for a, 2 for b."
  [root]
  (letfn [(fresh [adj] [(count adj) (conj! adj [])])
          (edge [adj u v lab] (do (assoc! adj u (conj (nth adj u) [v lab])) adj))
          (gen [nd adj]
            (let [[in adj] (fresh adj)
                  [out adj] (fresh adj)
                  op (:op nd) ch (:ch nd) kids (:kids nd)]
              (cond
                (= op :concat)
                (if (empty? kids)
                  [in out (cond (= ch \a) (edge adj in out 1)
                                (= ch \b) (edge adj in out 2)
                                :else adj)]
                  (loop [ks kids prev-out nil fst true adj adj]
                    (if (empty? ks)
                      [in out (edge adj prev-out out 0)]
                      (let [[ci co adj] (gen (first ks) adj)
                            adj (if fst
                                  (edge adj in ci 0)
                                  (edge adj prev-out ci 0))]
                        (recur (rest ks) co false adj)))))

                (= op :closure)
                (if (empty? kids)
                  (let [[si adj] (fresh adj)
                        [so adj] (fresh adj)
                        adj (cond (= ch \a) (edge adj si so 1)
                                  (= ch \b) (edge adj si so 2)
                                  :else adj)]
                    [in out (-> adj
                                (edge in si 0)
                                (edge in out 0)
                                (edge so si 0)
                                (edge so out 0))])
                  (let [[si so adj]
                        (loop [ks kids sub-in nil prev-out nil fst true adj adj]
                          (if (empty? ks)
                            [sub-in prev-out adj]
                            (let [[ci co adj] (gen (first ks) adj)]
                              (if fst
                                (recur (rest ks) ci co false
                                       (edge adj in ci 0))
                                (recur (rest ks) sub-in co false
                                       (edge adj prev-out ci 0))))))]
                    [in out (-> adj
                                (edge in out 0)
                                (edge so out 0)
                                (edge so si 0))]))

                (= op :union)
                (let [[li lo adj] (gen (nth kids 0) adj)
                      [ri ro adj] (gen (nth kids 1) adj)]
                  [in out (-> adj
                              (edge in li 0)
                              (edge in ri 0)
                              (edge lo out 0)
                              (edge ro out 0))]))))]
    (let [[in out adj] (gen root (transient []))]
      [(persistent! adj) in out])))

;; The subset construction mirrors the python twin: each DFA state
;; is the sorted vector of its NFA states, and a transition moves on
;; one label then closes over empty edges from the moved states only
;; (no closure table over every NFA state up front).

(defn- step-closure
  "The sorted NFA states reached from DFA state cur (a sorted vector)
  by one edge labelled tr, closed under empty edges ([] when none)."
  [adj n cur tr]
  (let [seen (transient (vec (repeat n false)))
        nc (count cur)
        q (loop [k 0 q (transient [])]
            (if (= k nc)
              q
              (let [es (nth adj (nth cur k))
                    ne (count es)]
                (recur (inc k)
                       (loop [i 0 q q]
                         (if (= i ne)
                           q
                           (let [e (nth es i)
                                 v (nth e 0)]
                             (if (and (= (nth e 1) tr) (not (nth seen v)))
                               (do (assoc! seen v true)
                                   (recur (inc i) (conj! q v)))
                               (recur (inc i) q)))))))))]
    (loop [qi 0 q q]
      (if (= qi (count q))
        (vec (sort (persistent! q)))
        (let [es (nth adj (nth q qi))
              ne (count es)]
          (recur (inc qi)
                 (loop [i 0 q q]
                   (if (= i ne)
                     q
                     (let [e (nth es i)
                           v (nth e 0)]
                       (if (and (zero? (nth e 1)) (not (nth seen v)))
                         (do (assoc! seen v true)
                             (recur (inc i) (conj! q v)))
                         (recur (inc i) q)))))))))))

(defn- has-state?
  "True when sorted state vector st holds NFA state s."
  [st s]
  (let [n (count st)]
    (loop [i 0]
      (cond (= i n) false
            (= (nth st i) s) true
            :else (recur (inc i))))))

(defn- nfa-to-dfa
  "Subset construction over adj from nfa-in, breadth first like the
  twin. Returns start, ends and rows ([a-next b-next] per state)."
  [adj nfa-in nfa-out]
  (let [n (count adj)
        s0 (let [seen (transient (vec (repeat n false)))]
             (assoc! seen nfa-in true)
             (loop [qi 0 q (transient [nfa-in])]
               (if (= qi (count q))
                 (vec (sort (persistent! q)))
                 (let [es (nth adj (nth q qi))
                       ne (count es)]
                   (recur (inc qi)
                          (loop [i 0 q q]
                            (if (= i ne)
                              q
                              (let [e (nth es i)
                                    v (nth e 0)]
                                (if (and (zero? (nth e 1)) (not (nth seen v)))
                                  (do (assoc! seen v true)
                                      (recur (inc i) (conj! q v)))
                                  (recur (inc i) q))))))))))]
    (loop [states (transient [s0])
           idx (transient {s0 0})
           rows (transient [])
           qi 0]
      (if (= qi (count states))
        (let [all (persistent! states)
              nd (count all)]
          {:start 0
           :ends (loop [d 0 ends #{}]
                   (if (= d nd)
                     ends
                     (recur (inc d) (if (has-state? (nth all d) nfa-out) (conj ends d) ends))))
           :dfa (persistent! rows)})
        (let [cur (nth states qi)
              ta (step-closure adj n cur 1)
              ia (get idx ta)
              states (if ia states (conj! states ta))
              ia (or ia (dec (count states)))
              idx (if (= ia (dec (count states))) (assoc! idx ta ia) idx)
              tb (step-closure adj n cur 2)
              ib (get idx tb)
              states (if ib states (conj! states tb))
              ib (or ib (dec (count states)))
              idx (if (= ib (dec (count states))) (assoc! idx tb ib) idx)]
          (recur states idx (conj! rows [ia ib]) (inc qi)))))))

(defn- minimize-dfa
  "Minimizes the 2-symbol DFA (start=0, ends set, dfa vector of [a-next b-next])
   via Moore partition refinement and prunes dead states that cannot reach ends."
  [dfa ends]
  (let [n (count dfa)
        init-cls (mapv #(if (contains? ends %) 1 0) (range n))
        cls (loop [cls init-cls prev-cnt -1]
              (let [[new-cls cnt]
                    (loop [i 0 mp {} out (transient []) next-id 0]
                      (if (= i n)
                        [(persistent! out) next-id]
                        (let [row (nth dfa i)
                              sig [(nth cls i)
                                   (nth cls (nth row 0))
                                   (nth cls (nth row 1))]
                              id (get mp sig -1)]
                          (if (not= id -1)
                            (recur (inc i) mp (conj! out id) next-id)
                            (recur (inc i) (assoc mp sig next-id) (conj! out next-id) (inc next-id))))))]
                (if (= cnt prev-cnt)
                  new-cls
                  (recur new-cls cnt))))
        num-cls (inc (reduce max 0 cls))
        rep (loop [i (dec n) r (vec (repeat num-cls 0))]
              (if (neg? i)
                r
                (recur (dec i) (assoc r (nth cls i) i))))
        min-dfa (mapv (fn [c]
                        (let [u (nth rep c)
                              row (nth dfa u)]
                          [(nth cls (nth row 0))
                           (nth cls (nth row 1))]))
                      (range num-cls))
        min-ends ( into #{} (for [c (range num-cls)
                                  :when (contains? ends (nth rep c))]
                              c))
        ;; Find which minimized states can reach min-ends
        alive (loop [alive min-ends]
                (let [alive2 (reduce (fn [acc c]
                                       (let [row (nth min-dfa c)]
                                         (if (or (contains? acc (nth row 0))
                                                 (contains? acc (nth row 1)))
                                           (conj acc c)
                                           acc)))
                                     alive
                                     (range num-cls))]
                  (if (= (count alive2) (count alive))
                    alive
                    (recur alive2))))]
    {:dfa min-dfa :ends min-ends :alive alive}))

(def MOD8 8000000112000000392)

(defn- vmult
  "Multiplies 1xN vector v by NxN matrix b modulo MOD."
  [v b n]
  (let [r (transient (vec (repeat n 0)))]
    (loop [k 0]
      (when (< k n)
        (let [vk (nth v k)]
          (when-not (zero? vk)
            (let [kb (* k n)]
              (loop [j 0]
                (when (< j n)
                  (let [bkj (nth b (+ kb j))]
                    (when-not (zero? bkj)
                      (let [s (+ (nth r j) (* vk bkj))]
                        (assoc! r j (if (>= s MOD8) (- s MOD8) s)))))
                  (recur (inc j)))))))
        (recur (inc k))))
    (loop [j 0]
      (when (< j n)
        (assoc! r j (rem (nth r j) MOD))
        (recur (inc j))))
    (persistent! r)))

(defn- mmult
  [a b n]
  (let [r (transient (vec (repeat (* n n) 0)))]
    (loop [i 0]
      (when (< i n)
        (let [xb (* i n)]
          (loop [k 0]
            (when (< k n)
              (let [aik (nth a (+ xb k))]
                (when-not (zero? aik)
                  (let [kb (* k n)]
                    (loop [j 0]
                      (when (< j n)
                        (let [bkj (nth b (+ kb j))]
                          (when-not (zero? bkj)
                            (let [o (+ xb j)
                                  s (+ (nth r o) (* aik bkj))]
                              (assoc! r o (if (>= s MOD8) (- s MOD8) s)))))
                        (recur (inc j)))))))
              (recur (inc k))))
          (loop [j 0]
            (when (< j n)
              (let [o (+ xb j)]
                (assoc! r o (rem (nth r o) MOD)))
              (recur (inc j)))))
        (recur (inc i))))
    (persistent! r)))

(defn- mexp-row0
  "Computes row 0 of m^p modulo MOD using O(n^2) vector-matrix steps for odd bits."
  [m p n]
  (let [row0 (assoc (vec (repeat n 0)) 0 1)]
    (loop [rv row0 mp m p p]
      (if (<= p 0)
        rv
        (let [rv (if (odd? p) (vmult rv mp n) rv)
              p (bit-shift-right p 1)
              mp (if (> p 0) (mmult mp mp n) mp)]
          (recur rv mp p))))))

(def ^:private dp-max
  "Lengths up to this count paths by stepping the DFA directly
  (l steps over 2 edges per state), cheaper than the matrix power
  and its minimization at these sizes."
  256)

(defn- count-dp
  "Strings of length l (0 to dp-max) accepted by the DFA: path
  counts from state 0 stepped l times, summed over ends."
  [dfa ends l]
  (let [n (count dfa)]
    (loop [k 0 c (assoc (vec (repeat n 0)) 0 1)]
      (if (= k l)
        (rem (reduce (fn [acc e] (+ acc (nth c e))) 0 ends) MOD)
        (recur (inc k)
               (persistent!
                (loop [u 0 nc (transient (vec (repeat n 0)))]
                  (if (= u n)
                    nc
                    (let [cu (nth c u)]
                      (if (zero? cu)
                        (recur (inc u) nc)
                        (let [row (nth dfa u)
                              a (nth row 0)
                              b (nth row 1)]
                          (assoc! nc a (rem (+ (nth nc a) cu) MOD))
                          (assoc! nc b (rem (+ (nth nc b) cu) MOD))
                          (recur (inc u) nc))))))))))))

(defn- count-strings
  [r l]
  (let [[adj nfa-in nfa-out] (build-nfa (parse-regex r))
        {:keys [dfa ends]} (nfa-to-dfa adj nfa-in nfa-out)]
    (if (<= 0 l dp-max)
      (count-dp dfa ends l)
      (let [{:keys [dfa ends alive]} (minimize-dfa dfa ends)]
        (if (not (contains? alive 0))
          0
          (let [alive-vec (vec (sort alive))
                n (count alive-vec)
                remap (into {} (map-indexed (fn [idx old-c] [old-c idx]) alive-vec))
                m (persistent!
                    (reduce (fn [m idx]
                              (let [old-c (nth alive-vec idx)
                                    row (nth dfa old-c)]
                                (reduce (fn [m nxt-c]
                                          (if-let [nxt-idx (get remap nxt-c)]
                                            (let [cell (+ (* idx n) nxt-idx)]
                                              (assoc! m cell (inc (nth m cell))))
                                            m))
                                        m row)))
                            (transient (vec (repeat (* n n) 0)))
                            (range n)))
                row0 (mexp-row0 m l n)]
            (rem (reduce (fn [acc old-e]
                           (if-let [e-idx (get remap old-e)]
                             (+ acc (nth row0 e-idx))
                             acc))
                         0 ends)
                 MOD)))))))

(defn -main
  [data]
  (when-not (str/blank? data)
    (try
      (let [toks (tokenize data)
            t (parse-int (first toks) 0 2147483647)]
        (cond
          (nil? t)
          (report "error: expected a count followed by regex length pairs")
          :else
          (let [rest-toks (subvec toks 1)
                avail (count rest-toks)
                want (* 2 t)]
            (if (and (< avail want) (odd? avail))
              (report "error: expected a count followed by regex length pairs")
              (let [pairs (partition 2 (take (min want avail) rest-toks))
                    bad (some (fn [[r l]]
                                (nil? (parse-int l -9223372036854775808 9223372036854775807)))
                              pairs)]
                (if bad
                  (report "error: expected a decimal length per regex")
                  (let [out (map (fn [[r l]]
                                   (str (count-strings r (parse-int l -9223372036854775808 9223372036854775807))))
                                 pairs)]
                    (when (seq out)
                      (print (str (str/join "\n" out) "\n"))))))))))
      (catch Exception _ (report "error: invalid input")))))

(let [data (slurp *in*)] (-main data))
