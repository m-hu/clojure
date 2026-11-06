(let [sm (into (sorted-map) (map (fn [i] [(mod (* i 37) 101) i]) (range 200)))
      ss (into (sorted-set) (map #(mod (* % 13) 97) (range 300)))]
  (println (count sm) (first sm) (last sm) (take 3 (subseq sm > 50)) (take 2 (rsubseq sm < 20)))
  (println (count ss) (first ss) (last ss) (take 3 (disj ss 0 1 2)) (contains? ss 5))
  (println (keys (dissoc (assoc sm -1 :x) 0)) (sort-by - [3 1 2]) (sort > [5 3 9]) (compare "a" "b")))
