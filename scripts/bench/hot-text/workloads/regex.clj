(let [s "a1 b22 c333 d4444 e55555"]
  (println (re-find #"\d+" s) (re-seq #"[a-z](\d+)" s) (re-matches #"[a-z]\d+" "x12"))
  (println (count (re-seq #"\w+" (apply str (repeat 200 "word1 word2 ")))))
  (println (re-find (re-pattern "c(\\d)(\\d)") s) (clojure.string/split s #"\s+")))
