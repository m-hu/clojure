;; #271: a program that needs the full boot anyway (here a macro) drops
;; the spliced clojure.set units: its require loads the interpreted
;; library, exactly as under run.
(ns corpus.clojure-set-full
  (:require [clojure.set :as set]))

(defmacro twice [x] `(do ~x ~x))

(twice (println (set/union #{1} #{2}) (set/difference #{1 2} #{2})
                (set/map-invert {:a 1})))
