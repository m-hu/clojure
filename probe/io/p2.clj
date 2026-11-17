(require (quote [clojure.string :as str]))
(def sep #"\s+")
(println (count (str/split (str/trim (slurp *in*)) sep)))
