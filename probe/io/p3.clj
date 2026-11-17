(require (quote [clojure.string :as str]))
(def sep #"\s+")
(println (reduce + (mapv parse-long (str/split (str/trim (slurp *in*)) sep))))
