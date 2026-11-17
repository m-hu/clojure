;; #271: zipmap of a string over the core (range) iterate or a counted long
;; range pairs chars with longs directly (no seq walk), and a core inc/inc'
;; iterate steps longs without calling; overflow, other fns, and other
;; inputs take the walk. Every result matches run.
(println (zipmap "0123456789" (range)) (zipmap "" (range)) (zipmap "abc" (range 2)) (zipmap "abcdef" (range 10 0 -3)))
(println (zipmap "xy" (iterate inc' 9223372036854775806)) (zipmap "xy" (iterate inc 5)) (zipmap "xyz" (iterate inc' 9223372036854775807)))
(println (try (zipmap "xyz" (iterate inc 9223372036854775806)) (catch Exception e (.getMessage e))))
(println (zipmap "ab" (iterate dec 0)) (zipmap [1 2] (range)) (zipmap "ab" [1 2]) (zipmap "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+/=" (range)))
(println (class (zipmap "0123456789" (range))) (count (zipmap (apply str (map char (range 65 200))) (range))) (get (zipmap "0123456789" (range)) \7))
(println (zipmap "ab" (rest (range))) (zipmap "ab" (iterate inc' 1.5)) (take 3 (iterate inc 1.5)) (nth (iterate inc' 0) 40))
