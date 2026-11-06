(require '[clojure.string :as str])
(let [s (str/join " " (range 100000))]
  (println (reduce + (map parse-long (str/split s #" ")))))
