;; Copyright (c) Rich Hickey. All rights reserved.
;; The use and distribution terms for this software are covered by the
;; Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
;; which can be found in the file epl-v10.html at the root of this distribution.
;; By using this software in any fashion, you are agreeing to be bound by
;; the terms of this license.
;; You must not remove this notice, or any other, from this software.

;; gap-124 slice 1: port of clojure.tools.deps.edn
;; (org.clojure/tools.deps.edn 0.9.57, edn.clj) over string paths
;; and the `clojure.tools.deps.util.port` substrate (no
;; java.io/Reader/clojure.string here).
;;
;; Port adaptations (see docs/gap-124/slice-1.md A1/A2/A5/D1/R3):
;; - `read-edn` takes a string, not a Reader (A1); callers `slurp`
;;   first. Error *shapes* match upstream (ex-info + `:path`);
;;   message *texts* are port-canonical (R3).
;; - `validate` is hand-written: spec.alpha lacks
;;   `s/keys`/`s/or`/`s/merge`, so the specs.clj grammar verdicts
;;   are encoded directly (A2). Same accept/reject verdicts, with
;;   one deliberate narrowing: a present coord-kind key must be
;;   well-typed (`{:local/root 42}` is rejected — a spec-letter
;;   reading would accept it through the all-optional `:git`
;;   branch of the `s/or`, but BDD scenario 5 pins rejection).
;; - `canonicalize` walks maps/vectors only (A5 — the grammar has
;;   no maps under other nodes on valid input, and `read-deps`
;;   validates before canonicalizing).
;; - `root-deps` is the embedded value `{}` (D1 — the port binary
;;   IS the runtime, so the port root carries no defaults).
;; - Path joins pin `/` and the stderr warning pins `\n` (no
;;   `File/separator` / `line.separator` here); `user.home` reads
;;   `$HOME` (then `%USERPROFILE%`, then the cwd).
;; - Opts arrive `& opts` and normalize through `as-opts` (kwargs
;;   or a single map — upstream's own `read-deps` passes a map).
;; - No syntax-quote or `::keywords` anywhere (the loader reads
;;   the whole file before evaluating it — the spec_alpha.clj
;;   discipline).
(ns clojure.tools.deps.edn
  "Functions for reading, validating, and manipulating deps.edn
  files and data structures."
  (:require [clojure.tools.deps.util.dir :as dir]
            [clojure.tools.deps.util.port :as port]))

(def ^:private nl "\n")

(def ^:private file-separator "/")

(defn- as-opts
  "Normalize an `& opts` rest: kwargs (`:path p ...`) or a single
  map (upstream's own `read-deps` passes a map)."
  [opts]
  (if (and (= 1 (count opts)) (map? (first opts)))
    (first opts)
    (apply hash-map opts)))

(defn- join-space
  "`clojure.string/join` over `\" \"` (clojure.string is unported)."
  [msgs]
  (apply str (interpose " " msgs)))

(defn- printerrln
  "println to *err*"
  [& msgs]
  (binding [*out* *err*
            *print-readably* nil]
    (pr (str (join-space msgs) nl))
    (flush)))

;;;; Read

(defn read-edn
  "Read edn from string s, which should contain exactly one edn value.
  If source exists but is blank, nil is returned.
  Throws if source is unreadable or contains multiple values.

  Opts:
    :path String path to file being read, for error reporting (recommended)"
  [s & opts]
  (port/read-edn-string s (:path (as-opts opts))))

;;;; Validate (hand-written specs.clj verdicts — A2)

(defn- exclusions-problem
  [xs lib]
  (cond (not (or (sequential? xs) (set? xs)))
        (str "invalid :exclusions for " lib ": expected a collection of lib symbols, found " (pr-str xs))
        (not (every? symbol? xs))
        (str "invalid :exclusions for " lib ": expected lib symbols, found " (pr-str xs))
        :else nil))

(defn- coord-problem
  "First coord verdict problem, or nil when valid (nil, or a map
  whose present kind/generic keys are well-typed)."
  [lib coord]
  (cond (nil? coord) nil
        (not (map? coord))
        (str "invalid coord for " lib ": expected a map, found " (pr-str coord))
        (and (contains? coord :local/root) (not (string? (:local/root coord))))
        (str "invalid :local/root for " lib ": expected a string, found " (pr-str (:local/root coord)))
        (and (contains? coord :mvn/version) (not (string? (:mvn/version coord))))
        (str "invalid :mvn/version for " lib ": expected a string, found " (pr-str (:mvn/version coord)))
        (and (contains? coord :git/url) (not (string? (:git/url coord))))
        (str "invalid :git/url for " lib ": expected a string, found " (pr-str (:git/url coord)))
        (and (contains? coord :git/sha) (not (string? (:git/sha coord))))
        (str "invalid :git/sha for " lib ": expected a string, found " (pr-str (:git/sha coord)))
        (and (contains? coord :git/tag) (not (string? (:git/tag coord))))
        (str "invalid :git/tag for " lib ": expected a string, found " (pr-str (:git/tag coord)))
        (and (contains? coord :deps/root) (not (string? (:deps/root coord))))
        (str "invalid :deps/root for " lib ": expected a string, found " (pr-str (:deps/root coord)))
        (and (contains? coord :deps/manifest) (not (keyword? (:deps/manifest coord))))
        (str "invalid :deps/manifest for " lib ": expected a keyword, found " (pr-str (:deps/manifest coord)))
        (contains? coord :exclusions)
        (exclusions-problem (:exclusions coord) lib)
        :else nil))

(defn- deps-val-problem
  [deps-val where]
  (cond (not (map? deps-val))
        (str "expected " where " to be a map of lib to coord, found " (pr-str deps-val))
        :else (reduce (fn [acc [lib coord]]
                        (or acc
                          (when-not (symbol? lib)
                            (str "expected " where " keys to be lib symbols, found " (pr-str lib)))
                          (coord-problem lib coord)))
                nil deps-val)))

(defn- paths-problem
  [paths]
  (cond (not (vector? paths))
        (str "expected :paths to be a vector, found " (pr-str paths))
        :else (reduce (fn [acc v]
                        (or acc
                          (when-not (or (string? v) (keyword? v))
                            (str "expected :paths entries to be path strings or alias keywords, found " (pr-str v)))))
                nil paths)))

(defn- aliases-problem
  [aliases]
  (cond (not (map? aliases))
        (str "expected :aliases to be a map of alias keyword to alias map, found " (pr-str aliases))
        :else (reduce (fn [acc [k v]]
                        (or acc
                          (when-not (keyword? k)
                            (str "expected :aliases keys to be keywords, found " (pr-str k)))))
                nil aliases)))

(def ^:private repo-update-policies #{:daily :always :never})

(def ^:private repo-checksums #{:warn :fail :ignore})

(defn- repo-policy-problem
  [policy where]
  (cond (not (map? policy))
        (str "expected " where " to be a repo policy map, found " (pr-str policy))
        (and (contains? policy :enabled) (not (boolean? (:enabled policy))))
        (str "expected " where " :enabled to be boolean, found " (pr-str (:enabled policy)))
        (and (contains? policy :update)
          (not (or (contains? repo-update-policies (:update policy)) (int? (:update policy)))))
        (str "expected " where " :update to be a policy or interval int, found " (pr-str (:update policy)))
        (and (contains? policy :checksum) (not (contains? repo-checksums (:checksum policy))))
        (str "expected " where " :checksum to be :warn, :fail, or :ignore, found " (pr-str (:checksum policy)))
        :else nil))

(defn- repo-problem
  [id repo]
  (let [where (str "mvn repo " (pr-str id))]
    (cond (nil? repo) nil
          (not (map? repo))
          (str "expected " where " to be a map or nil, found " (pr-str repo))
          :else (or (when (contains? repo :url)
                      (when-not (string? (:url repo))
                        (str "expected " where " :url to be a string, found " (pr-str (:url repo)))))
                    (when (contains? repo :mvn/releases)
                      (repo-policy-problem (:mvn/releases repo) (str where " :mvn/releases")))
                    (when (contains? repo :mvn/snapshots)
                      (repo-policy-problem (:mvn/snapshots repo) (str where " :mvn/snapshots")))))))

(defn- mvn-repos-problem
  [repos]
  (cond (not (map? repos))
        (str "expected :mvn/repos to be a map of repo id to repo, found " (pr-str repos))
        :else (reduce (fn [acc [id repo]]
                        (or acc
                          (when-not (string? id)
                            (str "expected :mvn/repos keys to be strings, found " (pr-str id)))
                          (repo-problem id repo)))
                nil repos)))

(defn- ns-aliases-problem
  [aliases]
  (cond (not (map? aliases))
        (str "expected :ns-aliases to be a map of simple symbol to simple symbol, found " (pr-str aliases))
        :else (reduce (fn [acc [k v]]
                        (or acc
                          (when-not (and (simple-symbol? k) (simple-symbol? v))
                            (str "expected :ns-aliases entries to be simple symbol to simple symbol, found " (pr-str [k v])))))
                nil aliases)))

(defn- tools-usage-problem
  [usage]
  (cond (not (map? usage))
        (str "expected :tools/usage to be a map, found " (pr-str usage))
        :else (or (when (contains? usage :ns-default)
                    (when-not (simple-symbol? (:ns-default usage))
                      (str "expected :tools/usage :ns-default to be a simple symbol, found " (pr-str (:ns-default usage)))))
                  (when (contains? usage :ns-aliases)
                    (ns-aliases-problem (:ns-aliases usage))))))

(defn- prep-lib-problem
  [prep]
  (cond (not (map? prep))
        (str "expected :deps/prep-lib to be a map, found " (pr-str prep))
        (not (contains? prep :ensure))
        "expected :deps/prep-lib to contain :ensure"
        (not (string? (:ensure prep)))
        (str "expected :deps/prep-lib :ensure to be a string, found " (pr-str (:ensure prep)))
        (not (contains? prep :alias))
        "expected :deps/prep-lib to contain :alias"
        (not (keyword? (:alias prep)))
        (str "expected :deps/prep-lib :alias to be a keyword, found " (pr-str (:alias prep)))
        (not (contains? prep :fn))
        "expected :deps/prep-lib to contain :fn"
        (not (symbol? (:fn prep)))
        (str "expected :deps/prep-lib :fn to be a symbol, found " (pr-str (:fn prep)))
        :else nil))

(defn- check-deps-map
  "First deps-map verdict problem, or nil when valid."
  [deps-edn]
  (cond (nil? deps-edn) nil
        (not (map? deps-edn))
        (str "expected a deps.edn map or nil, found " (pr-str deps-edn))
        :else (or (when (contains? deps-edn :paths)
                    (paths-problem (:paths deps-edn)))
                  (when (contains? deps-edn :deps)
                    (deps-val-problem (:deps deps-edn) ":deps"))
                  (when (contains? deps-edn :aliases)
                    (aliases-problem (:aliases deps-edn)))
                  (when (contains? deps-edn :deps/prep-lib)
                    (prep-lib-problem (:deps/prep-lib deps-edn)))
                  (when (contains? deps-edn :mvn/local-repo)
                    (when-not (string? (:mvn/local-repo deps-edn))
                      (str "expected :mvn/local-repo to be a string, found " (pr-str (:mvn/local-repo deps-edn)))))
                  (when (contains? deps-edn :mvn/repos)
                    (mvn-repos-problem (:mvn/repos deps-edn)))
                  (when (contains? deps-edn :tools/usage)
                    (tools-usage-problem (:tools/usage deps-edn))))))

(defn validate
  "Validate a deps-edn map according to the specs, throw if invalid.
  Returns the deps-edn map.

  Opts:
    :path String path to file being read"
  [deps-edn & opts]
  (let [opts (as-opts opts)
        path (or (:path opts) "unknown path")]
    (if-let [problem (check-deps-map deps-edn)]
      (throw (ex-info (str "Error validating deps in " path ": " problem) {:path path}))
      deps-edn)))

;;;; Canonicalize

(defn- canonicalize-sym
  [s path]
  (if (simple-symbol? s)
    (let [cs (as-> (name s) n (symbol n n))]
      (printerrln "DEPRECATED: Libs must be qualified, change" s "=>" cs
        (if path (str "(" path ")") ""))
      cs)
    s))

(defn- canonicalize-exclusions
  [{:keys [exclusions] :as coord} path]
  (if (seq (filter simple-symbol? exclusions))
    (assoc coord :exclusions (mapv #(canonicalize-sym % path) exclusions))
    coord))

(defn- canonicalize-dep-map
  [deps-map path]
  (when deps-map
    (reduce-kv (fn [acc lib coord]
                 (let [new-lib (if (simple-symbol? lib) (canonicalize-sym lib path) lib)
                       new-coord (canonicalize-exclusions coord path)]
                   (assoc acc new-lib new-coord)))
      {} deps-map)))

(def ^:private dep-map-keys
  [:deps :default-deps :override-deps :extra-deps :classpath-overrides])

(defn- canonicalize-node
  [x path]
  (if (map? x)
    (reduce (fn [xr k]
              (if-let [xm (get xr k)]
                (assoc xr k (canonicalize-dep-map xm path))
                xr))
      x dep-map-keys)
    x))

(defn- canonicalize-walk
  "Scoped `postwalk` (A5): maps/vectors only, node fn applied
  bottom-up like upstream."
  [x path]
  (let [x (cond (map? x)
                (reduce-kv (fn [m k v] (assoc m k (canonicalize-walk v path))) {} x)
                (vector? x)
                (mapv #(canonicalize-walk % path) x)
                :else x)]
    (canonicalize-node x path)))

(defn canonicalize
  "Canonicalize a deps.edn map (convert simple lib symbols to qualified lib symbols).
  Returns the deps-edn map.

  Opts:
    :path String path to file being read"
  [deps-edn & opts]
  (canonicalize-walk deps-edn (:path (as-opts opts))))

;;;; Read deps with validation and canonicalization

(defn read-deps
  "Read, validate, and canonicalize the deps.edn at path string f.
  This is the primary entry point for reading.
  Use read-edn, validate, or canonicalize for individual steps.

  Opts: none for now"
  [f & opts]
  (let [opts (assoc (as-opts opts) :path f)]
    (when (port/file-exists? f)
      (-> (slurp f)
        (read-edn :path (:path opts))
        (validate :path (:path opts))
        (canonicalize :path (:path opts))))))

;;;; Dep chain lookups

(defn root-deps
  "Read the root deps.edn resource from the classpath at the path
  clojure/tools/deps/deps.edn"
  []
  {})

(defn- home-dir
  "Upstream `user.home`: `$HOME`, else `%USERPROFILE%`, else the cwd."
  []
  (or (port/getenv "HOME") (port/getenv "USERPROFILE") (port/current-dir)))

(defn user-config-dir
  "Use the same logic as clj to calculate the location of the user config dir.
  Note that it's possible no dir may exist at this location.
  Return path as string."
  []
  (let [config-env (port/getenv "CLJ_CONFIG")
        xdg-env (port/getenv "XDG_CONFIG_HOME")]
    (cond config-env config-env
          xdg-env (str xdg-env file-separator "clojure")
          :else (str (home-dir) file-separator ".clojure"))))

(defn user-deps-path
  "Use the same logic as clj to calculate the location of the user deps.edn.
  Note that it's possible no file may exist at this location.
  Returns path as string."
  []
  (str (user-config-dir) file-separator "deps.edn"))

(defn user-deps
  "Calculate the user deps.edn per user-deps-path, read and
  return the deps.edn data"
  []
  (-> (user-deps-path) dir/canonicalize read-deps))

(defn project-dir
  "Calculate the project directory, usually . by default, but
  may be pushed to a new directory using the
  clojure.tools.deps.util.dir API."
  []
  dir/*the-dir*)

(defn project-deps-path
  "Calculate the project deps.edn location. This
  is the deps.edn in the current directory, as defined by
  clojure.tools.deps.util.dir/*the-dir* - use with-dir
  to push a new local directory context around a call to
  project-deps-path."
  []
  (str (project-dir) file-separator "deps.edn"))

(defn project-deps
  "Calculate the project deps.edn location, read and return the deps.edn data.

  You may use clojure.tools.deps.util.dir/with-dir to read from a custom project
  dir, either absolute or relative to the current dir context."
  []
  (-> (project-deps-path) dir/canonicalize read-deps))

(defn- choose-deps
  [requested standard-fn]
  (cond
    (= :standard requested) (standard-fn)
    (string? requested) (-> requested dir/canonicalize read-deps)
    (or (nil? requested) (map? requested)) requested
    :else (throw (ex-info (format "Unexpected dep source: %s" (pr-str requested))
                   {:requested requested}))))

(defn create-edn-maps
  "Takes optional map of location sources, keys = :root :user :project :extra
  where each key may be:
    :standard (default) - to get the default source
    string - for file path to source, relative to current dir context
    nil - to omit
    map - a literal map to use

  Returns a set of deps edn maps with the same keys :root :user :project :extra.
  Keys may be missing if source was nil or file was missing."
  ([]
   (create-edn-maps nil))
  ([{:keys [root user project extra] :as params
     :or {root :standard, user :standard, project :standard}}]
   (let [root-edn (choose-deps root #(root-deps))
         user-edn (choose-deps user #(user-deps))
         project-edn (choose-deps project #(project-deps))
         extra-edn (choose-deps extra (constantly nil))]
     (cond-> {}
       root-edn (assoc :root root-edn)
       user-edn (assoc :user user-edn)
       project-edn (assoc :project project-edn)
       extra-edn (assoc :extra extra-edn)))))

;;;; deps edn manipulation

(defn- merge-or-replace
  "If maps, merge, otherwise replace"
  [& vals]
  (when (some identity vals)
    (reduce (fn [ret val]
              (if (and (map? ret) (map? val))
                (merge ret val)
                (or val ret)))
      nil vals)))

(defn merge-edns
  "Merge multiple deps edn maps from left to right into a single deps edn map."
  [deps-edn-maps]
  (apply merge-with merge-or-replace (remove nil? deps-edn-maps)))

;;;; Aliases

;; per-key binary merge-with rules

(def ^:private last-wins (comp last #(remove nil? %) vector))
(def ^:private append (comp vec concat))
(def ^:private append-unique (comp vec distinct concat))

(def ^:private merge-alias-rules
  {:deps merge ;; FUTURE: remove
   :replace-deps merge ;; formerly :deps
   :extra-deps merge
   :override-deps merge
   :default-deps merge
   :classpath-overrides merge
   :paths append-unique ;; FUTURE: remove
   :replace-paths append-unique ;; formerly :paths
   :extra-paths append-unique
   :jvm-opts append
   :main-opts last-wins
   :exec-fn last-wins
   :exec-args merge-or-replace
   :ns-aliases merge
   :ns-default last-wins})

(defn- choose-rule [alias-key val]
  (or (merge-alias-rules alias-key)
    (if (map? val)
      merge
      (fn [_v1 v2] v2))))

(defn merge-alias-maps
  "Like merge-with, but using custom per-alias-key merge function"
  [& ms]
  (reduce
    #(reduce
       (fn [m [k v]] (update m k (choose-rule k v) v))
       %1 %2)
    {} ms))

(defn combine-aliases
  "Find, read, and combine alias maps identified by alias keywords from
  a deps edn map into a single args map."
  [edn-map alias-kws]
  (->> alias-kws
    (map #(get-in edn-map [:aliases %]))
    (apply merge-alias-maps)))
