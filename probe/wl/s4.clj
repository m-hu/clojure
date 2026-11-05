(println (reduce + (map inc (filter odd? (range 300000)))))
