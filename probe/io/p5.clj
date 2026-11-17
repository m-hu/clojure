(require (quote [clojure.string :as str]))
(def sep #"\s+")
(let [xs (mapv parse-long (str/split (str/trim (slurp *in*)) sep))] (doseq [x xs] (println (format "%.2f" (double x)))))
