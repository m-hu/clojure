;; steiner-triple: Steiner triple systems via backtracking search,
;; Kirkman schoolgirl resolution, and STS validation.
(ns steiner-triple (:require [clojure.string :as str]))
(defn digit-values [s] (mapv #(Character/digit ^char % 36) s))
(defn separators [s] (str/split s #"\r?\n"))
(defn tokenize [s]
  (let [t (str/trim s)]
    (if (str/blank? t) [] (vec (str/split t #"\s+")))))
(defn parse-token [t] t)
(defn as-long [t] (Long/parseLong t))
(defn sts-search [v]
  (let [triples (vec (for [a (range v) b (range (inc a) v) c (range (inc b) v)] [a b c]))
        by-pair (loop [i 0 m {}]
                  (if (>= i (count triples)) m
                    (let [[a b c] (nth triples i)]
                      (recur (inc i) (-> m
                                         (update [a b] (fnil conj []) i)
                                         (update [a c] (fnil conj []) i)
                                         (update [b c] (fnil conj []) i))))))
        need (quot (* v (dec v)) 6)]
    (letfn [(bt [covered chosen]
              (if (>= (count chosen) need) chosen
                (let [fp (first (for [a (range v) b (range (inc a) v) :when (not (covered [a b]))] [a b]))]
                  (loop [ts (get by-pair fp)]
                    (when (seq ts)
                      (let [t (nth triples (first ts))
                            [a b c] t
                            ps [[a b] [a c] [b c]]]
                        (if (some covered ps) (recur (rest ts))
                          (or (bt (into covered ps) (conj chosen t))
                              (recur (rest ts))))))))))]
      (sort (bt #{} [])))))
(defn kirkman [v triples]
  (let [per (quot v 3)
        triples (sort triples)
        nc (quot (+ (count triples) (dec per)) per)]
    (letfn [(bt [i classes]
              (if (>= i (count triples))
                (when (every? #(= per (count %)) classes) classes)
                (loop [ci 0 classes classes]
                  (when (< ci (count classes))
                    (let [c (nth classes ci)]
                      (if (>= (count c) per) (recur (inc ci) classes)
                        (let [used (set (mapcat identity c))
                              t (nth triples i)]
                          (if (some used t) (recur (inc ci) classes)
                            (or (bt (inc i) (assoc classes ci (conj c t)))
                                (when (seq c) (recur (inc ci) classes)))))))))))]
      (map sort (bt 0 (vec (repeat nc [])))))))
(defn sts-valid? [v triples]
  (and (= (count triples) (quot (* v (dec v)) 6))
       (loop [i 0 seen #{}]
         (if (>= i (count triples)) (= (count seen) (quot (* v (dec v)) 2))
           (let [[a b c] (nth triples i)]
             (if (not (and (<= 0 a) (< a b) (< b c) (< c v))) false
               (if (some seen [[a b] [a c] [b c]]) false
                 (recur (inc i) (conj seen [a b] [a c] [b c])))))))))
(defn -main []
  (let [toks (tokenize (slurp *in*))]
    (when (seq toks)
      (cond (= "S" (nth toks 0))
            (let [v (as-long (nth toks 1))]
              (doseq [[a b c] (sts-search v)] (println a b c)))
            (= "K" (nth toks 0))
            (let [v (as-long (nth toks 1))
                  triples (sts-search v)]
              (doseq [cls (kirkman v triples)]
                (println (str/join " " (mapcat (fn [[a b c]] [a b c]) cls)))))
            :else
            (let [v (as-long (nth toks 1)) nt (as-long (nth toks 2))
                  tr (vec (for [i (range nt)] [(as-long (nth toks (+ 3 (* 3 i))))
                                                (as-long (nth toks (+ 4 (* 3 i))))
                                                (as-long (nth toks (+ 5 (* 3 i))))]))]
              (println (if (sts-valid? v tr) "VALID" "INVALID")))))))
(-main)
