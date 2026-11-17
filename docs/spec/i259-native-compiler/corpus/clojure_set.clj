;; #271: a leading `ns` requiring clojure.set compiles the vendored
;; library into the program (keeping the Lite boot); every fn answers
;; exactly what run's interpreted library answers, non-set inputs, the
;; variadic arities, and thrown errors included.
(ns corpus.clojure-set
  (:require [clojure.set :as set]))

(println (set/union) (set/union #{1}) (set/union #{1 2} #{2 3})
         (set/union #{1} #{2} #{3 4} #{5}) (set/union [1 2] [3])
         (set/union nil #{1}) (set/union #{1} nil))
(println (set/intersection #{1 2 3} #{2 3 4}) (set/intersection #{1 2} #{3})
         (set/intersection #{1 2 3} #{2 3} #{3 4}) (set/intersection #{1}))
(println (set/difference #{1 2 3} #{2}) (set/difference #{1} #{1 2 3})
         (set/difference #{1 2 3 4} #{1} #{2}) (set/difference #{1}))
(println (set/select odd? #{1 2 3 4}) (set/subset? #{1} #{1 2})
         (set/subset? #{1 3} #{1 2}) (set/superset? #{1 2} #{1})
         (set/superset? #{1} #{1 2}))
(println (set/project #{{:a 1 :b 2} {:a 3 :b 4}} [:a])
         (set/rename-keys {:a 1 :b 2} {:a :x}) (set/rename-keys {:a 1 :b 2} {:a :b :b :a})
         (set/rename #{{:a 1}} {:a :z}))
(println (set/index #{{:a 1 :b 2} {:a 1 :b 3} {:a 2 :b 2}} [:a])
         (set/map-invert {:a 1 :b 2}))
(println (set/join #{{:a 1 :b 2} {:a 2 :b 3}} #{{:a 1 :c 3}})
         (set/join #{{:a 1}} #{{:b 2}}) (set/join #{{:a 1}} #{{:b 1}} {:a :b})
         (set/join #{} #{{:a 1}}))
(println (try (set/intersection 5 #{1}) (catch Exception e (.getMessage e)))
         (try (set/difference #{1 2} [2]) (catch Exception e (.getMessage e)))
         (try (set/union #{1} 5) (catch Exception e (.getMessage e))))
(let [s #{1 2}]
  (println (identical? s (set/union s)) (identical? s (set/union s #{}))
           (identical? (set/union) (set/union))))
