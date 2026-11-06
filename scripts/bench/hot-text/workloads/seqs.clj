(let [xs (range 100)]
  (println (partition 3 (take 10 xs)) (partition-all 4 (take 10 xs)) (interleave [1 2 3] [:a :b :c]))
  (println (concat [1 2] '(3 4)) (mapcat (fn [x] [x x]) [1 2]) (take-while #(< % 5) xs) (drop-while #(< % 95) xs))
  (println (distinct [1 2 1 3 2]) (reverse [1 2 3]) (last xs) (butlast [1 2 3]) (nth xs 50) (take 5 (iterate inc 0)))
  (println (take 4 (cycle [1 2])) (interpose :x [1 2 3]) (keep #(when (odd? %) %) (range 6)) (some #{3} xs) (every? number? xs))
  (println (map-indexed vector [:a :b]) (apply max xs) (apply + xs) (reduce max xs) (first (filter #(> % 42) xs)))
  (println (remove odd? (range 10)) (split-at 2 [1 2 3 4]) (split-with neg? [-1 0 1]) (take-nth 3 (range 10)) (zipmap [:a :b] [1 2]))
  (println (doall (for [x (range 3) y (range 3) :when (< x y)] [x y])) (count (seq "hello")) (empty? []) (not-empty [1])))
