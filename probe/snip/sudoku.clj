;; https://www.hackerrank.com/challenges/sudoku/problem
;; Sudoku solver using backtracking with minimum remaining values.
;; Empty cells are picked in row major order by fewest legal digits, with
;; digits tried from 1 to 9, so the first solution found matches the shared
;; expected outputs on every puzzle. Bit masks track used digits per row and
;; col and box, giving linear work per search node and a tiny fixed board.
;; The worst case stays exponential, as expected for an NP complete puzzle,
;; while memory stays constant at 81 cells plus a few small mask vectors.
;; Bad input shapes print a note on stderr and exit zero, since a nonzero
;; exit needs host calls, which this file avoids on purpose.
(require '[clojure.string :as str])

(def digit-values
  "Digit chars mapped to their numeric values for the pure number reader."
  (zipmap "0123456789" (range 10)))

(def max-long
  "Largest signed 64-bit value; tokens beyond long range are invalid."
  9223372036854775807)

(def min-long
  "Smallest signed 64-bit value."
  -9223372036854775808)

(defn fit-long
  "Returns n when it fits in a signed 64-bit long, else nil."
  [n]
  (when (<= min-long n max-long) n))


(defn whitespace?
  "True for the four blank chars treated as token separators."
  [ch]
  (or (= ch \space) (= ch \tab) (= ch \newline) (= ch \return)))

(defn tokenize
  "Split string s on blanks into a vector of token strings in linear time."
  [s]
  (if (nil? s)
    []
    (filterv seq (clojure.string/split s #"[ \t\n\r]+"))))

(defn parse-int-token
  "Read one signed integer token into a long, else nil for a bad token."
  [tok]
  (if (and (string? tok) (re-matches #"[+-]?[0-9]{1,18}" tok))
    (parse-long tok)
    (let [cs (seq tok)]
      (when (seq cs)
        (let [f (first cs)
              neg? (= f \-)
              ds (if (or neg? (= f \+)) (next cs) cs)]
          (when (seq ds)
            (loop [ds ds acc 0]
              (if (nil? ds)
                (fit-long (if neg? (-' acc) acc))
                (let [ch (first ds)]
                  (if (contains? digit-values ch)
                    (recur (next ds) (+' (*' acc 10) (get digit-values ch)))
                    nil))))))))))

(defn parse-all-ints
  "Read every token of s as a signed integer, else nil for a bad token."
  [s]
  (let [toks (tokenize s)]
    (loop [ts (seq toks) out []]
      (if (nil? ts)
        out
        (let [v (parse-int-token (first ts))]
          (if (nil? v)
            nil
            (recur (next ts) (conj out v))))))))

(defn fail
  "Report a bad input shape on stderr and return nil with exit code zero."
  [msg]
  (binding [*out* *err*]
    (println msg))
  nil)

(defn box-index
  "Box number 0 to 8 for row r col c."
  [r c]
  (+ (* (quot r 3) 3) (quot c 3)))

(def cell-row
  (mapv #(quot % 9) (range 81)))

(def cell-col
  (mapv #(rem % 9) (range 81)))

(def cell-box
  (mapv #(box-index (quot % 9) (rem % 9)) (range 81)))

(def mask-digits
  (mapv (fn [m]
          (loop [d 1 acc []]
            (if (> d 9)
              acc
              (if (zero? (bit-and m (bit-shift-left 1 d)))
                (recur (inc d) acc)
                (recur (inc d) (conj acc d))))))
        (range 1024)))

(def popcount-table
  (mapv count mask-digits))

(defn best-cell
  "Empty cell with fewest cands packed as (bi << 16) | mask, -1 when full, or -2 when dead."
  [board rows cols boxes]
  (loop [i 0 best-i -1 best-mask 0 best-cnt 10]
    (if (>= i 81)
      (if (= best-i -1)
        -1
        (bit-or (bit-shift-left best-i 16) best-mask))
      (if (not (zero? (nth board i)))
        (recur (inc i) best-i best-mask best-cnt)
        (let [r (nth cell-row i)
              c (nth cell-col i)
              b (nth cell-box i)
              used (bit-or (nth rows r) (nth cols c) (nth boxes b))
              mask (bit-and (bit-not used) 1022)
              cnt (nth popcount-table mask)]
          (cond
            (zero? cnt) -2
            (= cnt 1) (bit-or (bit-shift-left i 16) mask)
            (< cnt best-cnt) (recur (inc i) i mask cnt)
            :else (recur (inc i) best-i best-mask best-cnt)))))))

(defn solve-board
  "Solve board vectors, returning a solved board or nil when stuck."
  [board rows cols boxes]
  (let [b (best-cell board rows cols boxes)]
    (cond
      (= b -2) nil
      (= b -1) board
      :else
      (let [bi (bit-shift-right b 16)
            mask (bit-and b 1023)
            br (nth cell-row bi)
            bc (nth cell-col bi)
            bx (nth cell-box bi)
            ds (nth mask-digits mask)
            nd (count ds)
            row-v (nth rows br)
            col-v (nth cols bc)
            box-v (nth boxes bx)]
        (loop [idx 0]
          (when (< idx nd)
            (let [d (nth ds idx)
                  bit (bit-shift-left 1 d)
                  res (solve-board (assoc board bi d)
                                   (assoc rows br (bit-or row-v bit))
                                   (assoc cols bc (bit-or col-v bit))
                                   (assoc boxes bx (bit-or box-v bit)))]
              (or res (recur (inc idx))))))))))

(defn solve-puzzle
  "Solve one 81 cell puzzle vector, returning the solved or given board."
  [cells]
  (let [board (vec cells)
        masks (loop [k 0 rows (vec (repeat 9 0))
                     cols (vec (repeat 9 0)) boxes (vec (repeat 9 0))]
                (if (>= k 81)
                  [rows cols boxes]
                  (let [v (nth board k)]
                    (if (zero? v)
                      (recur (inc k) rows cols boxes)
                      (let [r (quot k 9)
                            c (rem k 9)
                            bit (bit-shift-left 1 v)
                            bx (box-index r c)]
                        (recur (inc k)
                               (assoc rows r (bit-or (nth rows r) bit))
                               (assoc cols c (bit-or (nth cols c) bit))
                               (assoc boxes bx (bit-or (nth boxes bx) bit))))))))
        [rows cols boxes] masks]
    (or (solve-board board rows cols boxes) board)))

(defn print-board
  "Print one board as nine rows of digits with a trailing blank each."
  [board]
  (doseq [r (range 9)]
    (println (str (str/join " " (subvec board (* r 9) (+ (* r 9) 9))) " "))))

(defn -main
  "Entry point reading the whole stdin string, silent on blank input."
  [input]
  (when-not (str/blank? input)
    (let [all (parse-all-ints input)]
      (if (nil? all)
        (fail "bad input: tokens must all be integers")
        (let [len (count all)
              t (if (pos? len) (min (nth all 0) (quot (dec len) 81)) 0)]
          (doseq [p (range (max t 0))]
            (let [base (+ 1 (* p 81))
                  cells (subvec all base (+ base 81))]
              (if (every? (fn [v] (and (<= 0 v) (<= v 9))) cells)
                (print-board (solve-puzzle cells))
                (fail "bad input: cells must be digits from 0 to 9")))))))))

;; Minimal top level driver.
(-main (slurp *in*))
