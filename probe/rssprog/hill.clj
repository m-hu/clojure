(ns hill-climbing
  (:require [clojure.string :as str]))

(defn attacking-pairs [board]
  (let [n (count board)]
    (loop [i 0 acc 0]
      (if (>= i n)
        acc
        (let [bi (nth board i)
              acc2 (loop [j (inc i) a acc]
                     (if (>= j n)
                       a
                       (let [bj (nth board j)
                             diff (- bi bj)
                             adiff (if (neg? diff) (- diff) diff)]
                         (recur (inc j)
                                (if (or (= bi bj) (= adiff (- j i)))
                                  (inc a)
                                  a)))))]
          (recur (inc i) acc2))))))

(defn row-conflicts [board n r col]
  (loop [i 0 cnt 0]
    (if (>= i n)
      cnt
      (if (= i r)
        (recur (inc i) cnt)
        (let [bi (nth board i)
              dr (let [x (- i r)] (if (neg? x) (- x) x))
              dc (let [x (- bi col)] (if (neg? x) (- x) x))]
          (recur (inc i) (if (or (= bi col) (= dc dr)) (inc cnt) cnt)))))))

(defn steepest-step [board cur]
  (let [n (count board)]
    (if (zero? cur)
      nil
      (loop [r 0 best-v cur best-r -1 best-c -1]
        (if (>= r n)
          (when (not= best-r -1) [best-r best-c best-v])
          (let [old-c (nth board r)
                base-v (- cur (row-conflicts board n r old-c))
                [bv br bc] (loop [c 0 bv best-v br best-r bc best-c]
                             (if (>= c n)
                               [bv br bc]
                               (if (= c old-c)
                                 (recur (inc c) bv br bc)
                                 (let [v (+ base-v (row-conflicts board n r c))]
                                   (if (< v bv)
                                     (recur (inc c) v r c)
                                     (recur (inc c) bv br bc))))))]
            (recur (inc r) bv br bc)))))))

(defn hill-climb [board]
  (loop [b board cur (attacking-pairs board)]
    (if-let [[r c nv] (steepest-step b cur)]
      (recur (assoc b r c) nv)
      [b cur])))

(defn -main []
  (let [toks (str/split (str/trim (slurp *in*)) #"\s+")
        parse #(Long/parseLong %)
        t (parse (first toks))
        nums (mapv parse (rest toks))]
    (loop [pos 0 tcases t sb (StringBuilder.)]
      (if (zero? tcases)
        (print (str sb))
        (let [n (nth nums pos)
              board (vec (map int (subvec nums (inc pos) (+ pos 1 n))))
              [final-board ap] (hill-climb board)]
          (.append sb (str/join " " final-board))
          (.append sb "\n")
          (.append sb (str ap))
          (.append sb "\n")
          (recur (+ pos 1 n) (dec tcases) sb))))))

(-main)
