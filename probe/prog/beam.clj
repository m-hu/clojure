(ns beam-search (:require [clojure.string :as str]))

(defn attacking [b]
  (let [n (count b)]
    (loop [i 0 cnt 0]
      (if (= i n)
        cnt
        (let [bi (nth b i)
              cnt2 (loop [j (inc i) c cnt]
                     (if (= j n)
                       c
                       (let [bj (nth b j)
                             diff (- bi bj)
                             adiff (if (neg? diff) (- diff) diff)]
                         (if (or (= bi bj) (= adiff (- j i)))
                           (recur (inc j) (inc c))
                           (recur (inc j) c)))))]
          (recur (inc i) cnt2))))))

(defn row-attacking [b n r]
  (let [hits (transient (vec (repeat n 0)))]
    (loop [i 0]
      (if (= i n)
        hits
        (if (= i r)
          (recur (inc i))
          (let [bi (nth b i)
                d (- i r)
                dist (if (neg? d) (- d) d)
                c1 (- bi dist)
                c2 (+ bi dist)]
            (assoc! hits bi (inc (nth hits bi)))
            (when (>= c1 0)
              (assoc! hits c1 (inc (nth hits c1))))
            (when (< c2 n)
              (assoc! hits c2 (inc (nth hits c2))))
            (recur (inc i))))))))

(defn lcg-next [s]
  (unchecked-add (unchecked-multiply s 6364136223846793005) 1442695040888963407))

(defn beam-search [n k seed]
  (let [gen-boards (loop [s seed acc (transient []) i 0]
                     (if (= i k)
                       (persistent! acc)
                       (let [[board s2] (loop [b (transient []) s s j 0]
                                          (if (= j n)
                                            [(persistent! b) s]
                                            (let [ns (lcg-next s)]
                                              (recur (conj! b (int (Long/remainderUnsigned ns n)))
                                                     ns (inc j)))))]
                         (recur s2 (conj! acc board) (inc i)))))
        init-costs (mapv attacking gen-boards)
        [best-init bestc-init] (loop [i 1
                                      bb (nth gen-boards 0)
                                      bc (nth init-costs 0)]
                                 (if (= i k)
                                   [bb bc]
                                   (let [c (nth init-costs i)]
                                     (if (< c bc)
                                       (recur (inc i) (nth gen-boards i) c)
                                       (recur (inc i) bb bc)))))]
    (loop [beam gen-boards costs init-costs best best-init bestc bestc-init iter 0]
      (if (or (zero? bestc) (>= iter 100))
        [best bestc]
        (let [nb-beam (count beam)
              succs (loop [bi 0 acc (transient [])]
                      (if (= bi nb-beam)
                        (sort (persistent! acc))
                        (let [b (nth beam bi)
                              base-c (nth costs bi)
                              acc2 (loop [r 0 acc acc]
                                     (if (= r n)
                                       acc
                                       (let [br (nth b r)
                                             hits (row-attacking b n r)
                                             base-r (- base-c (nth hits br))
                                             acc3 (loop [c2 0 acc acc]
                                                    (if (= c2 n)
                                                      acc
                                                      (if (= c2 br)
                                                        (recur (inc c2) acc)
                                                        (let [nc (+ base-r (nth hits c2))
                                                              nb (assoc b r c2)]
                                                          (recur (inc c2) (conj! acc [nc nb]))))))]
                                         (recur (inc r) acc3))))]
                          (recur (inc bi) acc2))))
              [new-beam new-costs new-best new-bestc]
              (loop [ss succs nb-acc [] nc-acc [] seen #{} bb best bc bestc]
                (if (or (empty? ss) (= (count nb-acc) k))
                  [nb-acc nc-acc bb bc]
                  (let [entry (first ss)
                        c (nth entry 0)
                        nb (nth entry 1)]
                    (if (contains? seen nb)
                      (recur (rest ss) nb-acc nc-acc seen bb bc)
                      (let [[bb2 bc2] (if (< c bc) [nb c] [bb bc])]
                        (recur (rest ss)
                               (conj nb-acc nb)
                               (conj nc-acc c)
                               (conj seen nb)
                               bb2 bc2))))))]
          (recur new-beam new-costs new-best new-bestc (inc iter)))))))

(defn -main []
  (let [s (slurp *in*)]
    (when-not (str/blank? s)
      (let [v (str/split s #"\s+")
            nv (count v)
            start (if (and (pos? nv) (str/blank? (nth v 0))) 1 0)
            toks (subvec v start)
            t (parse-long (nth toks 0))]
        (loop [pos 1 tcases t out (transient [])]
          (if (zero? tcases)
            (print (str (str/join "\n" (persistent! out)) "\n"))
            (let [n (parse-long (nth toks pos))
                  k (parse-long (nth toks (inc pos)))
                  seed (parse-long (nth toks (+ pos 2)))
                  [best bestc] (beam-search n k seed)]
              (recur (+ pos 3) (dec tcases)
                     (-> out
                         (conj! (str/join " " best))
                         (conj! (str bestc)))))))))))

(-main)
