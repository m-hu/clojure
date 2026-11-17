;; clojure.set's select and rename read their relation's metadata, which
;; is data, not var metadata: the program keeps no def metadata, and the
;; relations' metadata survives exactly as under run (#271).
(ns user (:require [clojure.set :as set]))
(def rel (with-meta #{{:a 1 :b 2} {:a 3 :b 4}} {:source :test}))
(println (meta (set/select (fn [m] (odd? (:a m))) rel)))
(println (meta (set/rename rel {:a :x})) (sort-by :x (set/rename rel {:a :x})))
(println (set/union #{1} #{2}) (set/subset? #{1} #{1 2}) (set/superset? #{1 2} #{1}))
