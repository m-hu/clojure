(let [s "Hello, World 123"]
  (println (int \a) (char 98) (Character/isDigit \5) (Character/isLetter \x) (Character/isWhitespace \space) (Character/toUpperCase \q))
  (println (count (filter #(Character/isDigit %) s)) (apply str (reverse s)) (.charAt s 4) (.length s) (.substring s 7) (.indexOf s "World"))
  (println (.toUpperCase s) (.trim "  x ") (.split "a,b,c" ",") (str (doto (StringBuilder.) (.append "ab") (.append 1))))
  (println (map int (take 3 s)) (frequencies (map #(Character/toLowerCase %) s)) (clojure.string/join (map char (range 97 102)))))
