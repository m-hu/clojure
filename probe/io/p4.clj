(require (quote [clojure.string :as str]))
(def sep #"\s+")
(let [xs (mapv parse-long (str/split (str/trim (slurp *in*)) sep))] (println (str/join " " (map (fn [x] (* 2 x)) xs))))
