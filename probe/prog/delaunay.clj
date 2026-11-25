;; Delaunay triangulation by randomized incremental insertion.
;;
;; Time: O(n log n) expected -- Hilbert-order insertion with a walk from
;;   the previous triangle makes location O(1) amortized expected, the
;;   Hilbert sort dominates, and total structural work is O(n)
;;   expected. The old code tested every triple against every point
;;   (O(n to the fourth)). Cocircular inputs add an output-sensitive
;;   term: each empty-circle cocircular group of m points retests its
;;   triples in O(m cubed n); with no four cocircular this term is zero.
;; Space: O(n).
;; Purity: core Clojure only, no host interop. Triangles live in
;;   persistent maps, orientation and incircle tests use the same
;;   formulas as the old code, and parsing is hand-written.
;; Robustness: blank stdin prints nothing; malformed input prints a
;;   message to stderr and returns normally (exit 0 -- a nonzero exit
;;   needs host interop, which this port forbids).
(ns computational-geometry.triangulation.delaunay
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
    (when (> (if (neg? d) (- d) d) EPS)
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

(defn bowyer-watson
  "Delaunay triangles over distinct coords as index triples."
  [pts]
  (let [n (count pts)
        xs (map #(double (first %)) pts)
        ys (map #(double (second %)) pts)
        minx (reduce min xs) maxx (reduce max xs)
        miny (reduce min ys) maxy (reduce max ys)
        dd (max (- maxx minx) (- maxy miny) 1.0)
        mm (+ (* 1000.0 dd) 10.0)
        cx (/ (+ minx maxx) 2.0) cy (/ (+ miny maxy) 2.0)
        allpts (conj pts [(- cx mm) (- cy mm)] [(+ cx mm) (- cy mm)] [cx (+ cy mm)])
        allpts (vec allpts)
        tris {0 [n (+ n 1) (+ n 2)]}
        edge-map (reduce (fn [em e] (assoc em e #{0}))
                         {} (tri-edges [n (+ n 1) (+ n 2)]))
        order (insertion-order pts)
        [tris _ _ _ _]
        (loop [ids order tris tris edge-map edge-map next-tid 1 last-tid 0 allpts allpts]
          (if (empty? ids)
            [tris edge-map next-tid last-tid allpts]
            (let [[tris edge-map next-tid last-tid allpts]
                  (insert-point tris edge-map allpts next-tid last-tid (first ids))]
              (recur (rest ids) tris edge-map next-tid last-tid allpts))))]
    (vec (for [[_ [a b c]] tris
               :when (and (< a n) (< b n) (< c n))
               :when (some? (circumcircle (nth allpts a)
                                          (nth allpts b)
                                          (nth allpts c)))]
           [a b c]))))

(defn uf-find
  "Union-find root without compression."
  [parent i]
  (loop [i i]
    (let [p (get parent i i)]
      (if (= p i) i (recur p)))))

(defn uf-union
  "Union the sets of a and b."
  [parent a b]
  (let [ra (uf-find parent a) rb (uf-find parent b)]
    (if (= ra rb) parent (assoc parent ra rb))))

(defn cocircular-quad?
  "True when d sits on the circle through a, b, c. The band is widened
  on purpose: joining is only a trigger, the exact retest decides."
  [pts a b c d]
  (let [cc (circumcircle (nth pts a) (nth pts b) (nth pts c))]
    (and (some? cc)
         (let [[[ux uy] r-sq] cc
               [dx dy] (nth pts d)
               dx (double dx) dy (double dy)
               dd (+ (* (- dx ux) (- dx ux)) (* (- dy uy) (- dy uy)))
               gap (if (neg? (- dd r-sq)) (- r-sq dd) (- dd r-sq))]
           (<= gap (* 4 EPS))))))

(defn cocircular-groups
  "Maximal point sets sharing one empty circle, found by joining the
  cocircular quads across interior triangulation edges."
  [pts tris]
  (let [use (reduce (fn [m [a b c]]
                      (-> m
                          (update (edge-key a b) (fnil conj []) c)
                          (update (edge-key b c) (fnil conj []) a)
                          (update (edge-key c a) (fnil conj []) b)))
                    {} tris)
        quads (vec (for [[[a b] opps] use
                         :when (= (count opps) 2)
                         :let [[c d] opps]
                         :when (cocircular-quad? pts a b c d)]
                     [a b c d]))
        members (into #{} (mapcat identity quads))
        parent (reduce (fn [p [a b c d]]
                         (-> p (uf-union a c) (uf-union a d)
                             (uf-union b c) (uf-union b d)))
                       {} quads)]
    (->> members
         (group-by #(uf-find parent %))
         vals
         (map vec)
         (filter #(>= (count %) 3)))))

(defn valid-combo?
  "True when original-index triple i<j<k has an empty circumcircle over
  all points, evaluated exactly like the old triple loop."
  [raw i j k]
  (let [cc (circumcircle (nth raw i) (nth raw j) (nth raw k))]
    (and (some? cc)
         (let [[[ux uy] r-sq] cc
               n (count raw)]
           (loop [m 0]
             (if (>= m n)
               true
               (if (or (= m i) (= m j) (= m k))
                 (recur (inc m))
                 (let [[mx my] (nth raw m)
                       mx (double mx) my (double my)
                       d-sq (+ (* (- mx ux) (- mx ux)) (* (- my uy) (- my uy)))]
                   (if (>= d-sq (- r-sq EPS))
                     (recur (inc m))
                     false)))))))))

(defn delaunay-tris
  "All empty-circle triples as original-index triples."
  [raw]
  (let [by-coord (reduce (fn [m [i p]] (update m p (fnil conj []) i))
                         {} (map-indexed vector raw))
        pts (vec (keys by-coord))
        n (count pts)]
    (if (< n 3)
      []
      (let [tris (bowyer-watson pts)
            groups (cocircular-groups pts tris)
            lists-of (fn [[a b c]]
                       [(get by-coord (nth pts a))
                        (get by-coord (nth pts b))
                        (get by-coord (nth pts c))])
            emit (fn [coord-tri retest?]
                   (let [[la lb lc] (lists-of coord-tri)]
                     (if (and (not retest?)
                              (= 1 (count la) (count lb) (count lc)))
                       [(vec (sort [(first la) (first lb) (first lc)]))]
                       (vec (for [i la j lb k lc
                                  :let [t (vec (sort [i j k]))]
                                  :when (valid-combo? raw (nth t 0) (nth t 1) (nth t 2))]
                              t)))))
            extra (mapcat (fn [g]
                            (let [m (count g)]
                              (for [a (range m) b (range (inc a) m)
                                    c (range (inc b) m)]
                                [(nth g a) (nth g b) (nth g c)])))
                          groups)]
        (vec (sort (into #{} (concat (mapcat #(emit % false) tris)
                                     (mapcat #(emit % true) extra)))))))))

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
              (let [tris (delaunay-tris raw)]
                (println (count tris))
                (doseq [[i j k] tris]
                  (println (str i " " j " " k)))))))))))

(-main)
