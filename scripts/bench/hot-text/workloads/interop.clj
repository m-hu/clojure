;; Host interop most programs reach: number parsing, StringBuilder output,
;; String/format with a locale, and char and string methods, in loops long
;; enough for the loop tier.
(def input (slurp *in*))

(defn build-report
  "A StringBuilder filled line by line with parsed and formatted numbers."
  [n]
  (let [sb (StringBuilder.)]
    (dotimes [i n]
      (let [l (Long/parseLong (str i))
            d (Double/parseDouble (str i ".25"))
            k (Integer/parseInt (str (rem i 7)))
            c (Character/digit (char (+ 48 (rem i 10))) 10)]
        (.append sb l)
        (.append sb " ")
        (.append sb (String/format java.util.Locale/US "%.3f" (to-array [d])))
        (.append sb " ")
        (.append sb (format "%d:%s" (+ k c) (if (even? i) "even" "odd")))
        (.append sb "\n")))
    sb))

(defn scan
  "Char and string methods over the report."
  [^String s]
  (loop [i 0 digits 0 spaces 0]
    (if (< i (.length s))
      (let [ch (.charAt s i)]
        (recur (inc i)
               (if (Character/isDigit ch) (inc digits) digits)
               (if (= ch \space) (inc spaces) spaces)))
      [digits spaces (.indexOf s "even") (.substring s 0 10) (.toUpperCase (subs s 0 4))
       (.contains s "odd") (.startsWith s "0") (.trim "  x  ") (.lastIndexOf s "\n")])))

(let [sb (build-report 120)
      s (.toString sb)]
  (println (count input) (.length s) (scan s) (count (str sb))
           (Long/remainderUnsigned -7 3) (Math/floorMod -7 3) (Long/MAX_VALUE) (Integer/MAX_VALUE)
           (Double/isNaN (/ 0.0 0.0)) (Math/round 2.5) (Math/ceil 1.2) (Math/hypot 3.0 4.0)))
