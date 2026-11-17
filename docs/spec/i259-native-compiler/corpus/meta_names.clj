;; #271: def names carrying metadata (defn-, type hints, :dynamic, rich
;; maps) pool without reading text back at startup; the vars behave and
;; their metadata reads exactly as under run (this program reads var
;; metadata, so every key stays, including ones that still read back).
(def answer 42)
(defn- helper [x] (inc x))
(defn ^String tagged [x] (str x))
(def ^{:k :v :n 1 :c \a :d 2.5 :z nil :f false :s "str" :q user/answer} rich 5)
(def ^:dynamic *dyn* 1)
(println (helper 1) (tagged 2) rich (binding [*dyn* 2] *dyn*))
(println (select-keys (meta #'rich) [:k :n :c :d :z :f :s :q :private]))
(println (:private (meta #'helper)) (:tag (meta #'tagged)) (:arglists (meta #'helper)) (:dynamic (meta #'*dyn*)))
