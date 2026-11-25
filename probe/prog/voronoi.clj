;; Voronoi vertices through Delaunay duality.
;;
;; Time: O(n log n) expected -- the vertices are exactly the Delaunay
;;   circumcenters, so a randomized incremental Delaunay build
;;   (Hilbert-order insertion with a walk from the previous triangle)
;;   dominates; the old code tested every triple against every point
;;   (O(n to the fourth)). This file outputs vertices only, for which
;;   duality matches the sweep line while staying simple and exact.
;; Space: O(n).
;; Purity: core Clojure only, no host interop. Triangles live in
;;   persistent maps and parsing plus two-decimal rendering are
;;   hand-written.
;; Robustness: blank stdin prints nothing; malformed input prints a
;;   message to stderr and returns normally (exit 0 -- a nonzero exit
;;   needs host interop, which this port forbids).
(ns computational-geometry.voronoi.fortune-voronoi
  (:require [clojure.string :as str]))

(def EPS 1e-9)

(def digit-values
  "Digit character to numeric value."
  (zipmap "0123456789" (range)))

(defn whitespace?
  "True for the ASCII whitespace that separates input tokens."
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
  "Parse an integer token; nil when malformed or outside int64 range."
  [tok]
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
                (if want-neg v (- v))))))))))

(defn parse-double-token
  "Parse a decimal token; nil when malformed."
  [tok]
  (cond
    (= tok "NaN") (/ 0.0 0.0)
    (or (= tok "Infinity") (= tok "+Infinity")) (/ 1.0 0.0)
    (= tok "-Infinity") (/ -1.0 0.0)
    :else
    (let [cs (vec tok)
          n (count cs)]
      (when (pos? n)
        (let [c0 (nth cs 0)
              signed (or (= c0 \-) (= c0 \+))
              want-neg (= c0 \-)
              i0 (if signed 1 0)]
          (when (< i0 n)
            (let [epos (loop [i i0]
                         (cond (>= i n) -1
                               (let [c (nth cs i)]
                                 (or (= c \e) (= c \E))) i
                               :else (recur (inc i))))
                  mend (if (= epos -1) n epos)
                  mant (subvec cs i0 mend)
                  dots (count (filter #(= % \.) mant))
                  md (vec (remove #(= % \.) mant))]
              (when (and (<= dots 1)
                         (seq md)
                         (every? digit-values md))
                (let [dot-at (loop [i i0]
                               (cond (>= i mend) -1
                                     (= (nth cs i) \.) i
                                     :else (recur (inc i))))
                      intc (if (= dot-at -1)
                             (subvec cs i0 mend)
                             (subvec cs i0 dot-at))
                      fracc (if (= dot-at -1) [] (subvec cs (inc dot-at) mend))
                      expv (when (not (= epos -1))
                             (let [r (subvec cs (inc epos) n)
                                   rc (count r)]
                               (when (pos? rc)
                                 (let [s0 (nth r 0)
                                       es (or (= s0 \-) (= s0 \+))
                                       ed (subvec r (if es 1 0))]
                                   (when (and (seq ed) (every? digit-values ed))
                                     (if (> (count ed) 6)
                                       (if (= s0 \-) :neg-huge :pos-huge)
                                       (let [v (reduce (fn [a c]
                                                         (+ (* a 10) (digit-values c)))
                                                       0 ed)]
                                         (if (= s0 \-) (- v) v))))))))]
                  (when (or (= epos -1) (some? expv))
                    (let [alld (vec (concat intc fracc))
                          sig (vec (drop-while #(= % \0) alld))]
                      (if (empty? sig)
                        (if want-neg (* -1.0 0.0) 0.0)
                        (let [keep (take 15 sig)
                              lead (reduce (fn [a c]
                                             (+ (* a 10.0)
                                                (double (digit-values c))))
                                           0.0 keep)
                              ex (+ (if (number? expv) expv 0)
                                    (- (count fracc))
                                    (- (count sig) (count keep)))
                              ex (cond (= expv :pos-huge) 999
                                       (= expv :neg-huge) -999
                                       :else ex)
                              mag (cond (> ex 308) (/ 1.0 0.0)
                                        (< ex -324) 0.0
                                        :else (loop [i 0 v lead]
                                                (cond (= i ex) v
                                                      (< i ex) (recur (inc i) (* v 10.0))
                                                      :else (recur (dec i) (/ v 10.0)))))]
                          (if want-neg (- mag) mag))))))))))))))

(defn double-to-bigint
  "Exact integer value of an integral double beyond long range."
  [v]
  (loop [q v k 0]
    (if (< q 9007199254740992.0)
      (loop [i 0 acc (bigint (long q))]
        (if (= i k) acc (recur (inc i) (*' acc 2))))
      (recur (/ q 2.0) (inc k)))))

(defn format-fixed
  "Render x with exactly k decimals, halves up."
  [x k]
  (cond
    (not (== x x)) "NaN"
    (and (not (zero? x)) (== x (* x 2.0)))
    (if (neg? x) "-Infinity" "Infinity")
    :else
    (let [neg (neg? (/ 1.0 x))
          a (if neg (- x) x)]
      (if (>= a 9.0e18)
        (str (when neg "-") (str (double-to-bigint a)) "."
             (apply str (repeat k \0)))
        (let [pow10k (loop [i 0 v 1.0]
                       (if (= i k) v (recur (inc i) (* v 10.0))))
              ip (long a)
              frac (- a (double ip))
              r (long (+ (* frac pow10k) 0.5))
              [ip r] (if (>= r (long pow10k)) [(inc ip) 0] [ip r])
              fs (str r)
              pad (subs "00000000000000000000" 0 (- k (count fs)))]
          (str (when neg "-") (str ip) "." pad fs))))))

(defn abs-num
  "Absolute value."
  [x]
  (if (neg? x) (- x) x))

(defn point-cmp
  "Order points by x then y, within tolerance."
  [[x1 y1] [x2 y2]]
  (if (> (abs-num (- x1 x2)) EPS)
    (compare x1 x2)
    (if (> (abs-num (- y1 y2)) EPS)
      (compare y1 y2)
      0)))

(defn orient
  "Twice the signed area of triangle a-b-c."
  [[ax ay] [bx by] [cx cy]]
  (let [ax (double ax) ay (double ay)
        bx (double bx) by (double by)
        cx (double cx) cy (double cy)]
    (- (* (- bx ax) (- cy ay)) (* (- by ay) (- cx ax)))))

(defn circumcircle
  "Circumcenter and squared radius, or nil for collinear triples."
  [[ax ay] [bx by] [cx cy]]
  (let [ax (double ax) ay (double ay)
        bx (double bx) by (double by)
        cx (double cx) cy (double cy)
        d (* 2.0 (+ (* ax (- by cy)) (* bx (- cy ay)) (* cx (- ay by))))]
    (when (> (abs-num d) EPS)
      (let [ux (/ (+ (* (+ (* ax ax) (* ay ay)) (- by cy))
                     (* (+ (* bx bx) (* by by)) (- cy ay))
                     (* (+ (* cx cx) (* cy cy)) (- ay by)))
                  d)
            uy (/ (+ (* (+ (* ax ax) (* ay ay)) (- cx bx))
                     (* (+ (* bx bx) (* by by)) (- ax cx))
                     (* (+ (* cx cx) (* cy cy)) (- bx ax)))
                  d)
            r-sq (+ (* (- ax ux) (- ax ux)) (* (- ay uy) (- ay uy)))]
        [[ux uy] r-sq]))))

(defn edge-key
  "Canonical undirected key for the edge u-v."
  [u v]
  (if (< u v) [u v] [v u]))

(defn tri-edges
  "The three edge keys of triangle [a b c]."
  [[a b c]]
  [(edge-key a b) (edge-key b c) (edge-key c a)])

(defn inside-or-on?
  "True when p lies inside triangle t or on its border."
  [allpts [a b c] p]
  (let [pa (nth allpts a) pb (nth allpts b) pc (nth allpts c)]
    (and (>= (orient pa pb p) (- EPS))
         (>= (orient pb pc p) (- EPS))
         (>= (orient pc pa p) (- EPS)))))

(defn locate
  "Id of a triangle containing p, walking from tid0 with a full-scan
  fallback so location always succeeds."
  [tris edge-map allpts tid0 p]
  (let [start (if (contains? tris tid0) tid0 (first (keys tris)))]
    (or (loop [cur start seen #{start} steps 0]
          (when (<= steps (count tris))
            (let [[a b c] (get tris cur)
                  pa (nth allpts a) pb (nth allpts b) pc (nth allpts c)
                  o1 (orient pa pb p)
                  o2 (orient pb pc p)
                  o3 (orient pc pa p)]
              (if (and (>= o1 (- EPS)) (>= o2 (- EPS)) (>= o3 (- EPS)))
                cur
                (let [bad (cond (< o1 (- EPS)) (edge-key a b)
                                (< o2 (- EPS)) (edge-key b c)
                                :else (edge-key c a))
                      nxt (first (disj (get edge-map bad #{}) cur))]
                  (when (and (some? nxt) (not (contains? seen nxt)))
                    (recur nxt (conj seen nxt) (inc steps))))))))
        (some (fn [[tid t]] (when (inside-or-on? allpts t p) tid)) tris)
        (first (keys tris)))))

(defn in-cavity?
  "True when triangle t must be rebuilt for the insertion of p: p sits
  strictly inside or on its circumcircle."
  [allpts t p]
  (let [[a b c] t
        pi (first p)]
    (if (or (= a pi) (= b pi) (= c pi))
      true
      (let [cc (circumcircle (nth allpts a) (nth allpts b) (nth allpts c))]
        (if (nil? cc)
          (inside-or-on? allpts t (second p))
          (let [[[ux uy] r-sq] cc
                [px py] (second p)
                px (double px) py (double py)
                d-sq (+ (* (- px ux) (- px ux)) (* (- py uy) (- py uy)))]
            (< d-sq (+ r-sq EPS))))))))

(defn cavity-of
  "Triangles whose circumcircle covers p, grown from the seed."
  [tris edge-map allpts seed p]
  (loop [queue [seed] in #{seed}]
    (if (empty? queue)
      in
      (let [tid (peek queue)
            queue (pop queue)
            nbrs (mapcat #(disj (get edge-map % #{}) tid)
                         (tri-edges (get tris tid)))]
        (recur (into queue (remove #(or (contains? in %)
                                        (not (in-cavity? allpts (get tris %) p)))
                                   nbrs))
               (into in (filter #(in-cavity? allpts (get tris %) p) nbrs)))))))

(defn insert-point
  "Insert coord index pi into the triangulation; returns the new state."
  [tris edge-map allpts next-tid last-tid pi]
  (let [p (nth allpts pi)
        seed (locate tris edge-map allpts last-tid p)
        cav (cavity-of tris edge-map allpts seed [pi p])
        counts (frequencies (mapcat #(tri-edges (get tris %)) cav))
        bound (vec (for [[e c] counts :when (= c 1)] e))
        cav-pairs (mapv (fn [tid] [tid (get tris tid)]) cav)
        tris (reduce dissoc tris cav)
        edge-map (reduce (fn [em [tid t]]
                           (reduce (fn [em e]
                                     (let [s (disj (get em e #{}) tid)]
                                       (if (empty? s) (dissoc em e) (assoc em e s))))
                                   em (tri-edges t)))
                         edge-map cav-pairs)
        [tris edge-map next-tid tids]
        (loop [es bound tris tris edge-map edge-map next-tid next-tid tids []]
          (if (empty? es)
            [tris edge-map next-tid tids]
            (let [[u v] (peek es)
                  t (if (neg? (orient p (nth allpts u) (nth allpts v)))
                      [pi v u]
                      [pi u v])]
              (recur (pop es)
                     (assoc tris next-tid t)
                     (reduce (fn [em e]
                               (update em e (fnil conj #{}) next-tid))
                             edge-map (tri-edges t))
                     (inc next-tid)
                     (conj tids next-tid)))))]
    [tris edge-map next-tid (first tids) allpts]))

(defn hilbert-index
  "Hilbert order of the 16-bit cell (ix, iy)."
  [ix iy]
  (loop [s 32768 x ix y iy d 0]
    (if (zero? s)
      d
      (let [rx (if (pos? (bit-and x s)) 1 0)
            ry (if (pos? (bit-and y s)) 1 0)
            d (+ d (* s s (bit-xor (* 3 rx) ry)))
            [x y] (if (zero? ry)
                    (let [[x y] (if (= rx 1) [(- s 1 x) (- s 1 y)] [x y])]
                      [y x])
                    [x y])]
        (recur (quot s 2) x y d)))))

(defn insertion-order
  "Coord indices sorted along the Hilbert curve."
  [pts]
  (let [n (count pts)
        xs (map first pts) ys (map second pts)
        minx (reduce min xs) maxx (reduce max xs)
        miny (reduce min ys) maxy (reduce max ys)
        sx (- (double maxx) (double minx))
        sy (- (double maxy) (double miny))]
    (vec (sort-by
          (fn [i]
            (let [[x y] (nth pts i)
                  ix (if (zero? sx) 0
                         (min 65535 (max 0 (long (* (/ (- (double x) (double minx)) sx)
                                                     65535.0)))))
                  iy (if (zero? sy) 0
                         (min 65535 (max 0 (long (* (/ (- (double y) (double miny)) sy)
                                                     65535.0)))))]
              (hilbert-index ix iy)))
          (range n)))))

(defn smallest-combo
  "Lexicographically smallest index triple holding one index per list."
  [la lb lc]
  (let [i (min (reduce min la) (reduce min lb) (reduce min lc))
        rest-lists (vec (remove #(contains? (set %) i) [la lb lc]))
        j (min (reduce min (first rest-lists)) (reduce min (second rest-lists)))
        last-list (first (remove #(contains? (set %) j) rest-lists))
        k (reduce min last-list)]
    [i j k]))

(defn voronoi-vertices
  "Voronoi vertices over raw points via Delaunay duality. Each center is
  evaluated with its vertices in original-index order, reproducing the
  old triple loop bit for bit, and centers are preordered by their
  smallest triple so dedup survivors match too."
  [raw]
  (let [by-coord (reduce (fn [m [i p]] (update m p (fnil conj []) i))
                         {} (map-indexed vector raw))
        pts (vec (keys by-coord))
        n (count pts)]
    (if (< n 3)
      []
      (let [xs (map #(double (first %)) pts)
            ys (map #(double (second %)) pts)
            minx (reduce min xs) maxx (reduce max xs)
            miny (reduce min ys) maxy (reduce max ys)
            dd (max (- maxx minx) (- maxy miny) 1.0)
            mm (+ (* 1000.0 dd) 10.0)
            cx (/ (+ minx maxx) 2.0) cy (/ (+ miny maxy) 2.0)
            allpts (vec (conj pts [(- cx mm) (- cy mm)]
                              [(+ cx mm) (- cy mm)] [cx (+ cy mm)]))
            tris {0 [n (+ n 1) (+ n 2)]}
            edge-map (reduce (fn [em e] (assoc em e #{0}))
                             {} (tri-edges [n (+ n 1) (+ n 2)]))
            order (insertion-order pts)
            [tris _ _ _ _]
            (loop [ids order tris tris edge-map edge-map next-tid 1 last-tid 0]
              (if (empty? ids)
                [tris edge-map next-tid last-tid allpts]
                (let [[tris edge-map next-tid last-tid _]
                      (insert-point tris edge-map allpts next-tid last-tid
                                    (first ids))]
                  (recur (rest ids) tris edge-map next-tid last-tid))))]
        (let [centers (for [[_ [a b c]] tris
                                :when (and (< a n) (< b n) (< c n))
                                :let [cc (circumcircle (nth allpts a)
                                                       (nth allpts b)
                                                       (nth allpts c))]
                                :when (some? cc)
                                :let [[i j k] (smallest-combo
                                               (get by-coord (nth pts a))
                                               (get by-coord (nth pts b))
                                               (get by-coord (nth pts c)))
                                      rc (circumcircle (nth raw i)
                                                       (nth raw j)
                                                       (nth raw k))]
                                :when (some? rc)]
                            [[i j k] (first rc)])]
          (vec (map second (sort-by first centers))))))))

(defn die
  "Report invalid input on stderr and return normally."
  [msg]
  (binding [*out* *err*]
    (println msg)))

(defn nan-coord?
  "True when any coordinate is not a number."
  [raw]
  (some (fn [[x y]] (or (not (== x x)) (not (== y y)))) raw))

(defn -main
  [& _args]
  (let [content (slurp *in*)]
    (when-not (str/blank? content)
      (let [tokens (tokenize content)
            n (parse-long-token (first tokens))]
        (cond
          (nil? n)
          (die "error: first token must be an integer point count")

          (or (> n 4611686018427387903) (< n -4611686018427387904))
          (die "error: point count out of range")

          :else
          (let [coords (take (* 2 n) (rest tokens))
                raw (vec (for [[xs ys] (partition 2 coords)]
                           [(parse-double-token xs) (parse-double-token ys)]))]
            (cond
              (some (fn [[a b]] (or (nil? a) (nil? b))) raw)
              (die "error: expected numeric coordinate pairs after the count")

              (not (= (count raw) (max n 0)))
              (die "error: point count does not match the supplied coordinates")

              (nan-coord? raw)
              (println 0)

              :else
              (let [centers (voronoi-vertices raw)
                    sorted-centers (sort point-cmp centers)
                    unique-centers (reduce (fn [acc pt]
                                             (if (and (seq acc)
                                                      (zero? (point-cmp (peek acc) pt)))
                                               acc
                                               (conj acc pt)))
                                           [] sorted-centers)]
                (println (count unique-centers))
                (doseq [[x y] unique-centers]
                  (println (str (format-fixed (double x) 2) " "
                                (format-fixed (double y) 2))))))))))))

(-main)
