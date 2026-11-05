(let [n 20000
      words (mapv #(str "w" (mod (* % 31) 5000)) (range n))
      freq (frequencies words)]
  (println (count freq) (apply max (vals freq))))
