(defprotocol Shape (area [s]) (name-of [s]))
(defrecord Circle [r] Shape (area [_] (* Math/PI r r)) (name-of [_] "circle"))
(defrecord Rect [w h] Shape (area [_] (* w h)) (name-of [_] "rect"))
(defmulti describe :kind)
(defmethod describe :a [m] (str "A" (:v m)))
(defmethod describe :default [m] "other")
(let [shapes [(->Circle 1.0) (->Rect 2 3) (map->Rect {:w 1 :h 1})]]
  (println (map name-of shapes) (reduce + (map area shapes)) (:w (second shapes)) (assoc (first shapes) :r 2.0))
  (println (describe {:kind :a :v 1}) (describe {:kind :b}) (instance? Circle (first shapes)) (record? (first shapes))))
