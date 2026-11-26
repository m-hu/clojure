
;; backprop: perceptron, logistic regression and multilayer neural
;; network backpropagation. Fixed learning rates and SGD order;
;; tanh hidden layer, sigmoid output, MSE loss.
(ns backprop
  (:require [clojure.string :as str]))

(defn digit-values [s] (mapv #(Long/parseLong %) s))
(defn separators [s] nil)
(defn tokenize [s] (str/split (str/trim s) #"\s+"))
(defn parse-token [t] (Long/parseLong t))

(defn fmt6 [x]
  (String/format java.util.Locale/US "%.6f" (into-array Object [(double x)])))

(defn sigmoid [z] (/ 1.0 (+ 1.0 (Math/exp (- z)))))

(defn perceptron [n f e x y]
  (loop [i 0 w (vec (repeat f 0)) b 0]
    (if (= i e)
      [w b]
      (let [[w2 b2] (loop [j 0 ww w bb b]
                      (if (= j n)
                        [ww bb]
                        (let [yy (if (= (nth y j) 1) 1 -1)
                              act (+ bb (reduce + 0 (map * ww (nth x j))))]
                          (if (<= (* yy act) 0)
                            (recur (inc j) (mapv + ww (map #(* yy %) (nth x j))) (+ bb yy))
                            (recur (inc j) ww bb)))))]
        (recur (inc i) w2 b2)))))

(defn logistic [n f e x y]
  (loop [i 0 w (vec (repeat f 0.0)) b 0.0]
    (if (= i e)
      [w b]
      (let [[w2 b2] (loop [j 0 ww w bb b]
                      (if (= j n)
                        [ww bb]
                        (let [z (+ bb (reduce + 0.0 (map * ww (nth x j))))
                              pr (sigmoid z)
                              err (- (nth y j) pr)]
                          (recur (inc j)
                                 (mapv (fn [a xi] (+ a (* 0.5 err xi))) ww (nth x j))
                                 (+ bb (* 0.5 err))))))]
        (recur (inc i) w2 b2)))))

(defn mlp-step [xrow yy w1 b1 w2 b2]
  (let [a1 (mapv (fn [h]
                   (Math/tanh (+ (nth b1 h) (reduce + 0.0 (map * (nth w1 h) xrow)))))
                 (range (count w2)))
        z2 (+ b2 (reduce + 0.0 (map * w2 a1)))
        pr (sigmoid z2)
        d2 (* (- pr yy) pr (- 1.0 pr))
        w1b (mapv (fn [h]
                    (let [d1 (* d2 (nth w2 h) (- 1.0 (* (nth a1 h) (nth a1 h))))]
                      (mapv (fn [a xi] (- a (* 0.1 d1 xi))) (nth w1 h) xrow)))
                  (range (count w2)))
        b1b (mapv (fn [h]
                    (let [d1 (* d2 (nth w2 h) (- 1.0 (* (nth a1 h) (nth a1 h))))]
                      (- (nth b1 h) (* 0.1 d1))))
                  (range (count w2)))
        w2b (mapv (fn [h] (- (nth w2 h) (* 0.1 d2 (nth a1 h)))) (range (count w2)))
        b2b (- b2 (* 0.1 d2))]
    [w1b b1b w2b b2b]))

(defn mlp [n f h e x y]
  (loop [i 0
         w1 (mapv (fn [hh] (mapv (fn [j] (* 0.05 (+ hh j 1))) (range f))) (range h))
         b1 (vec (repeat h 0.0))
         w2 (mapv (fn [hh] (* 0.05 (inc hh))) (range h))
         b2 0.0]
    (if (= i e)
      [w1 b1 w2 b2]
      (let [[w1b b1b w2b b2b]
            (loop [j 0 a w1 bb b1 c w2 d b2]
              (if (= j n)
                [a bb c d]
                (let [[a2 bb2 c2 d2] (mlp-step (nth x j) (nth y j) a bb c d)]
                  (recur (inc j) a2 bb2 c2 d2))))]
        (recur (inc i) w1b b1b w2b b2b)))))

(defn mlp-pred [xrow w1 b1 w2 b2]
  (let [a1 (mapv (fn [h] (Math/tanh (+ (nth b1 h) (reduce + 0.0 (map * (nth w1 h) xrow)))))
                 (range (count w2)))
        z2 (+ b2 (reduce + 0.0 (map * w2 a1)))]
    (sigmoid z2)))

(defn parse-case [nums pos]
  (let [n (int (nth nums pos)) f (int (nth nums (+ pos 1)))
        h (int (nth nums (+ pos 2))) e (int (nth nums (+ pos 3)))
        p0 (+ pos 4)
        x (vec (for [i (range n)] (vec (map int (subvec nums (+ p0 (* i (inc f))) (+ p0 (* i (inc f)) f))))))
        y (vec (for [i (range n)] (int (nth nums (+ p0 (* i (inc f)) f)))))]
    [n f h e x y (+ p0 (* n (inc f)))]))

(defn -main []
  (let [nums (digit-values (tokenize (slurp *in*)))
        t (first nums)]
    (loop [pos 1 tcases t sb (StringBuilder.)]
      (if (zero? tcases)
        (print (str sb))
        (let [[n f h e x y p2] (parse-case nums pos)
              [pw pb] (perceptron n f e x y)
              [lw lb] (logistic n f e x y)
              [w1 b1 w2 b2] (mlp n f h e x y)
              ppreds (mapv (fn [xi] (if (pos? (+ pb (reduce + 0 (map * pw xi)))) 1 0)) x)
              lpreds (mapv (fn [xi] (if (>= (sigmoid (+ lb (reduce + 0.0 (map * lw xi)))) 0.5) 1 0)) x)
              mpreds (mapv (fn [xi] (if (>= (mlp-pred xi w1 b1 w2 b2) 0.5) 1 0)) x)
              mps (mapv #(mlp-pred % w1 b1 w2 b2) x)
              loss (/ (reduce + 0.0 (map (fn [yi pr] (* 0.5 (- yi pr) (- yi pr))) y mps)) n)]
          (.append sb "BACKPROP\n")
          (.append sb (str "P " (str/join " " (conj pw pb)) " " (str/join " " ppreds) "\n"))
          (.append sb (str "L " (str/join " " (map fmt6 (conj lw lb))) " " (str/join " " lpreds) "\n"))
          (.append sb (str "M " (str/join " " mpreds) " " (fmt6 loss) "\n"))
          (recur p2 (dec tcases) sb))))))

(-main)
