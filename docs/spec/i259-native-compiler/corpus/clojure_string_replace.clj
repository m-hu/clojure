;; #271: a program whose leading ns requires clojure.string and calls a
;; clojure.string fn no Lite native covers (replace) compiles the vendored
;; definitions it needs (replace, replace-by) into itself; every shape of
;; replace answers exactly what run's interpreted library answers.
(ns corpus.clojure-string-replace
  (:require [clojure.string :as str]))

(println (str/replace "a-b-c" \- \+) (str/replace "a.b.c" "." "::")
         (str/replace "0042" #"^0+" "") (str/replace "x1y22" #"\d+" "N"))
(println (str/replace "Almost Pig Latin" #"\b(\w)(\w+)\b" "$2$1ay")
         (str/replace "abc" #"z" "q") (str/replace "" #"a*" "-"))
(println (try (str/replace "a1b22" #"\d+" str) (catch Exception e (.getMessage e))))
(println (try (str/replace "abc" 5 "x") (catch Exception e (.getMessage e)))
         (try (str/replace nil "a" "b") (catch Exception e (class e))))
(println (str/join "," (map #(str/replace % #"[aeiou]" "") ["banana" "kiwi"])))
