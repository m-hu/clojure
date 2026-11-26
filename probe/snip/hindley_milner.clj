
;; hindley_milner: Damas-Milner type inference (Algorithm W) for a
;; small functional language with lambdas, let-polymorphism,
;; conditionals and integer operations. S-expression syntax.
(ns hindley-milner
  (:require [clojure.string :as str] [clojure.set]))

(defn digit-values [s] (mapv #(Long/parseLong %) s))
(defn separators [s] nil)
(defn tokenize [s] (str/split (str/trim s) #"\s+"))
(defn parse-token [t] (Long/parseLong t))

(def ^:private freshc (atom 0))
(defn fresh [] (let [v @freshc] (swap! freshc inc) [:v v]))

(defn apply-s [s t]
  (cond (= (first t) :v) (if-let [u (get s (second t))] (apply-s s u) t)
        (= (first t) :c) t
        :else [:a (apply-s s (nth t 1)) (apply-s s (nth t 2))]))

(defn ftv [t]
  (cond (= (first t) :v) #{(second t)}
        (= (first t) :c) #{}
        :else (clojure.set/union (ftv (nth t 1)) (ftv (nth t 2)))))

(defn comp-s [s1 s2]
  (merge (into {} (map (fn [[k v]] [k (apply-s s1 v)]) s2))
         (into {} (remove (fn [[k _]] (contains? (set (keys s2)) k)) s1))))

(defn unify* [a b s]
  (let [ua (apply-s s a) ub (apply-s s b)]
    (cond (and (= (first ua) :v) (= (first ub) :v) (= (second ua) (second ub))) s
          (= (first ua) :v) (if (contains? (ftv ub) (second ua)) nil (assoc s (second ua) ub))
          (= (first ub) :v) (if (contains? (ftv ua) (second ub)) nil (assoc s (second ub) ua))
          (and (= (first ua) :c) (= (first ub) :c)) (when (= (second ua) (second ub)) s)
          (and (= (first ua) :a) (= (first ub) :a))
          (when-let [s1 (unify* (nth ua 1) (nth ub 1) s)]
            (unify* (nth ua 2) (nth ub 2) s1))
          :else nil)))

(declare parse-expr)

(defn skip-sp [s p]
  (loop [i p]
    (if (and (< i (count s)) (contains? #{\space \tab} (.charAt s i)))
      (recur (inc i)) i)))

(defn parse-atom [s p]
  (let [p2 (skip-sp s p)
        q (loop [i p2]
            (if (and (< i (count s))
                     (not (contains? #{\space \tab \( \)} (.charAt s i))))
              (recur (inc i)) i))]
    [(subs s p2 q) q]))

(defn parse-list [s p kw]
  (let [p0 (skip-sp s p)]
    (cond (= kw "lam")
          (let [p1 (skip-sp s p0)
                [v p2] (parse-atom s p1)
                [b p3 e3] (parse-expr s p2)
                p4 (skip-sp s p3)]
            (if (or e3 (>= p4 (count s)) (not= (.charAt s p4) \)))
              [nil p4 true]
              [{:k 1 :var v :a b} (inc p4) false]))
          (= kw "app")
          (let [[f p1 e1] (parse-expr s p0)
                [x p2 e2] (parse-expr s p1)
                p3 (skip-sp s p2)]
            (if (or e1 e2 (>= p3 (count s)) (not= (.charAt s p3) \)))
              [nil p3 true]
              [{:k 2 :a f :b x} (inc p3) false]))
          (= kw "let")
          (let [p1 (skip-sp s p0)
                [v p2] (parse-atom s p1)
                [val p3 e3] (parse-expr s p2)
                [bo p4 e4] (parse-expr s p3)
                p5 (skip-sp s p4)]
            (if (or e3 e4 (>= p5 (count s)) (not= (.charAt s p5) \)))
              [nil p5 true]
              [{:k 3 :var v :a val :b bo} (inc p5) false]))
          (= kw "if")
          (let [[c p1 e1] (parse-expr s p0)
                [tt p2 e2] (parse-expr s p1)
                [e p3 e3] (parse-expr s p2)
                p4 (skip-sp s p3)]
            (if (or e1 e2 e3 (>= p4 (count s)) (not= (.charAt s p4) \)))
              [nil p4 true]
              [{:k 4 :a c :b tt :c e} (inc p4) false]))
          (or (= kw "+") (= kw "=="))
          (let [[x p1 e1] (parse-expr s p0)
                [y p2 e2] (parse-expr s p1)
                p3 (skip-sp s p2)]
            (if (or e1 e2 (>= p3 (count s)) (not= (.charAt s p3) \)))
              [nil p3 true]
              [{:k 5 :op kw :a x :b y} (inc p3) false]))
          :else [nil p0 true])))

(defn parse-expr [s p]
  (let [p0 (skip-sp s p)]
    (if (>= p0 (count s))
      [nil p0 true]
      (if (= (.charAt s p0) \()
        (let [p1 (skip-sp s (inc p0))
              [kw p2] (parse-atom s p1)]
          (parse-list s p2 kw))
        (let [[a p2] (parse-atom s p0)]
          (if (empty? a) [nil p2 true] [{:k 0 :atom a} p2 false]))))))

(defn ftv-env [env]
  (apply clojure.set/union #{}
         (map (fn [[_ sc]] (clojure.set/difference (ftv (:t sc)) (set (:qs sc)))) env)))

(defn inst [sc]
  (let [s (into {} (map (fn [v] [v (fresh)]) (:qs sc)))]
    (apply-s s (:t sc))))

(defn gen [env t]
  (let [a (ftv t) b (ftv-env env)]
    {:qs (vec (sort (clojure.set/difference a b))) :t t}))

(defn apply-env [s env]
  (into {} (map (fn [[k sc]] [k {:qs (:qs sc) :t (apply-s s (:t sc))}]) env)))

(declare infer*)

(defn infer* [env e]
  (cond (= (:k e) 0)
        (let [a (:atom e)]
          (cond (and (seq a) (every? #(and (>= (int %) (int \0)) (<= (int %) (int \9))) a))
                [{} [:c "int"]]
                (or (= a "T") (= a "F")) [{} [:c "bool"]]
                (contains? env a) [{} (inst (get env a))]
                :else nil))
        (= (:k e) 1)
        (let [b (fresh)
              e2 (assoc env (:var e) {:qs [] :t b})
              r (infer* e2 (:a e))]
          (when r
            (let [[s1 t1] r]
              [s1 [:a (apply-s s1 b) t1]])))
        (= (:k e) 2)
        (when-let [[s1 t1] (infer* env (:a e))]
          (let [e2 (apply-env s1 env)]
            (when-let [[s2 t2] (infer* e2 (:b e))]
              (let [b (fresh)
                    s3 (unify* (apply-s s2 t1) [:a t2 b] {})]
                (when s3
                  [(comp-s s3 (comp-s s2 s1)) (apply-s s3 b)])))))
        (= (:k e) 3)
        (when-let [[s1 t1] (infer* env (:a e))]
          (let [e2 (apply-env s1 env)
                e3 (assoc e2 (:var e) (gen e2 t1))]
            (when-let [[s2 t2] (infer* e3 (:b e))]
              [(comp-s s2 s1) t2])))
        (= (:k e) 4)
        (when-let [[s1 tc] (infer* env (:a e))]
          (when-let [sa (unify* tc [:c "bool"] s1)]
            (let [e2 (apply-env sa env)]
              (when-let [[s2 tt] (infer* e2 (:b e))]
                (let [e3 (apply-env s2 e2)]
                  (when-let [[s3 te] (infer* e3 (:c e))]
                    (when-let [sb (unify* (apply-s s3 tt) te {})]
                      [(comp-s sb (comp-s (comp-s s3 s2) sa)) (apply-s sb te)])))))))
        (= (:k e) 5)
        (when-let [[s1 t1] (infer* env (:a e))]
          (when-let [sa (unify* t1 [:c "int"] s1)]
            (let [e2 (apply-env sa env)]
              (when-let [[s2 t2] (infer* e2 (:b e))]
                (when-let [sb (unify* t2 [:c "int"] s2)]
                  [(comp-s (comp-s (comp-s sb s2) sa) s1)
                   [:c (if (= (:op e) "+") "int" "bool")]])))))
        :else nil))

(defn collect-vars [t]
  (loop [stack [t] seen []]
    (if (empty? stack)
      seen
      (let [x (peek stack) rst (pop stack)]
        (cond (= (first x) :v) (if (some #(= % (second x)) seen)
                                 (recur rst seen)
                                 (recur rst (conj seen (second x))))
              (= (first x) :c) (recur rst seen)
              :else (recur (conj rst (nth x 2) (nth x 1)) seen))))))

(defn show-t [t nm]
  (cond (= (first t) :v) (get nm (second t))
        (= (first t) :c) (second t)
        :else (str "(" (show-t (nth t 1) nm) " -> " (show-t (nth t 2) nm) ")")))

(defn -main []
  (let [lines (str/split-lines (slurp *in*))
        t (parse-token (first lines))]
    (loop [li 1 tcases t sb (StringBuilder.)]
      (if (zero? tcases)
        (print (str sb))
        (let [li2 (loop [i li]
                    (if (and (< i (count lines)) (empty? (nth lines i)))
                      (recur (inc i)) i))]
          (if (>= li2 (count lines))
            (do (.append sb "HM TYPE-ERROR\n") (recur li2 (dec tcases) sb))
            (let [line (nth lines li2)
                  [e p2 bad] (parse-expr line 0)
                  p3 (skip-sp line p2)]
              (reset! freshc 0)
              (if (or bad (nil? e) (not= p3 (count line)))
                (do (.append sb "HM TYPE-ERROR\n") (recur (inc li2) (dec tcases) sb))
                (let [r (infer* {} e)]
                  (if (nil? r)
                    (do (.append sb "HM TYPE-ERROR\n") (recur (inc li2) (dec tcases) sb))
                    (let [[s tt] r
                          f (apply-s s tt)
                          vs (collect-vars f)
                          nm (into {} (map-indexed (fn [i v] [v (str (char (+ (int \a) i)))]) vs))]
                      (.append sb (str "HM " (show-t f nm) "\n"))
                      (recur (inc li2) (dec tcases) sb))))))))))))

(-main)
