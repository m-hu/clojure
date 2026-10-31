;; Copyright (c) Rich Hickey. All rights reserved.
;; The use and distribution terms for this software are covered by the
;; Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
;; which can be found in the file epl-v10.html at the root of this distribution.
;; By using this software in any fashion, you are agreeing to be bound by
;; the terms of this license.
;; You must not remove this notice, or any other, from this software.

;; gap-124 slice 1: shell port of clojure.tools.deps.extensions
;; (org.clojure/tools.deps, extensions.clj + extensions/local.clj)
;; covering `lib-location` only — the full multimethod SPI
;; (`coord-type`, `canonicalize`, `manifest-type`, ...) arrives in
;; later slices. Dispatch keys on the coord kind
;; (`:local/root` -> `:local`, `:mvn/version` -> `:mvn`,
;; `:git/url`/`:git/sha`/`:git/tag` -> `:git`); mixed-kind coords
;; take the first match in that order (pinned, port-canonical).
;;
;; Port decisions (see docs/gap-124/slice-1.md R1/R2):
;; - `:mvn` is REFUSED (permanent redesign): Maven coords parse
;;   and validate (config compatibility), but resolving them
;;   throws an ex-info naming the lib with the source-only rule.
;; - `:git` is NOT-YET (temporary): it throws an ex-info
;;   deferring to the git slice.
;; - Unknown/non-map coords throw `throw-bad-coord`-shaped
;;   ex-infos (upstream extensions.clj parity).
;; No syntax-quote or `::keywords` anywhere (the loader reads the
;; whole file before evaluating it — the spec_alpha.clj discipline).
(ns clojure.tools.deps.extensions
  "Defines the SPI for new dep and manifest types - to implement, extend the
  multimethods here, then load those extensions before using the tools.deps API.
  (Slice 1 shell: `lib-location` only.)")

(defn- coord-type
  [coord]
  (cond (and (map? coord) (contains? coord :local/root)) :local
        (and (map? coord) (contains? coord :mvn/version)) :mvn
        (and (map? coord)
          (or (contains? coord :git/url)
            (contains? coord :git/sha)
            (contains? coord :git/tag))) :git
        :else :default))

(defmulti lib-location
  "Takes a coordinate and returns the location where the lib would be
  installed locally. Location keys:

  :base     local repo base directory path
  :path     path within base dir
  :type     coordinate type"
  (fn [lib coord config] (coord-type coord)))

(defmethod lib-location :local
  [_lib coord _config]
  {:base (:local/root coord)
   :path ""
   :type :local})

(defmethod lib-location :mvn
  [lib coord _config]
  (throw (ex-info (str "Source-only port: cannot resolve Maven coordinate for " lib)
           {:lib lib :coord coord})))

(defmethod lib-location :git
  [lib coord _config]
  (throw (ex-info (str "Cannot resolve git coordinate for " lib ": deferred to the git slice")
           {:lib lib :coord coord})))

(defmethod lib-location :default
  [lib coord _config]
  (throw (ex-info (if (map? coord)
                    (str "Coord of unknown type for " lib ": " (pr-str coord))
                    (str "Bad coordinate for library " lib ", expected map: " (pr-str coord)))
           {:lib lib :coord coord})))
