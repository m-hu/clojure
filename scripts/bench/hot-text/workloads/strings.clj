(require '[clojure.string :as str])
(let [s "  The quick brown Fox jumps over the lazy dog  "
      t (str/trim s)
      words (str/split t #" ")]
  (println (count words) (str/join "," (map str/upper-case words)))
  (println (str/lower-case t) (str/capitalize "hello") (str/reverse "abc"))
  (println (str/starts-with? t "The") (str/ends-with? t "dog") (str/includes? t "Fox"))
  (println (str/index-of t "brown") (str/last-index-of t "o") (subs t 4 9))
  (println (str/replace t "o" "0") (str/replace t #"[aeiou]" "_") (str/blank? "  "))
  (println (str/split-lines "a\nb\nc") (str/triml "  x") (str/trimr "x  ") (str/escape "a<b" {\< "&lt;"}))
  (println (format "%d %5.2f %s %08d %x" 42 3.14159 "str" 7 255) (str 1 2.5 \c :k 'sym nil "s")))
