;; Copyright (c) Rich Hickey. All rights reserved.
;; The use and distribution terms for this software are covered by the
;; Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
;; which can be found in the file epl-v10.html at the root of this distribution.
;; By using this software in any fashion, you are agreeing to be bound by
;; the terms of this license.
;; You must not remove this notice, or any other, from this software.

;; gap-124 slice 1: port of clojure.tools.deps.util.dir
;; (org.clojure/tools.deps.edn 0.9.57, util/dir.clj) over string
;; paths — the port has no java.io.File, so `*the-dir*` holds an
;; absolute path string and `canonicalize`/`sub-path?` go through
;; the `clojure.tools.deps.util.port` substrate. Upstream's
;; DEPRECATED `as-canonical` is not ported (it is deprecated there
;; too, and the slice contract lists the four live vars only).
;;
;; Port adaptations (see docs/gap-124/slice-1.md A3/A4):
;; - `canonicalize` resolves against `*the-dir*` via the substrate
;;   2-arity `canonical-path` (prefix-walk when the leaf is
;;   missing — Rust `fs::canonicalize` requires existence).
;; - `sub-path?` compares canonical path strings (parent-walk over
;;   `parent-path`) instead of `Files/isSameFile` inode identity;
;;   a missing `*the-dir*` answers false (upstream: unreadable
;;   paths answer false). The parent split treats `/` and `\` as
;;   separators on every platform (uniform code — the coverage
;;   gate runs on Linux only, so platform-conditional branches
;;   are uncoverable).
;; - `with-dir` builds its expansion with `list` (no
;;   syntax-quote) and fully-qualified heads: the loader reads the
;;   whole file before evaluating it, so syntax-quote would
;;   qualify against the *requiring* namespace (the spec_alpha.clj
;;   discipline — no `::keywords` here either).
(ns clojure.tools.deps.util.dir
  "An abstraction for managing a 'current directory' stack.
  The dynamic var `*the-dir*` holds the current directory.
  Use `with-dir` to push a new absolute (or more usefully)
  relative directory onto the stack, which will pop when the
  scope exits. Use `canonicalize` to get the canonical path
  string of a path in terms of the current directory context."
  (:require [clojure.tools.deps.util.port :as port]))

(def ^:dynamic *the-dir*
  "Thread-local directory context for resolving relative directories.
  Defaults to current directory. Should always hold an absolute directory
  path string, never nil."
  (port/current-dir))

(defn canonicalize
  "Make canonical path string in terms of the current directory context.
  f may be either absolute or relative."
  [f]
  (port/canonical-path *the-dir* f))

(defmacro with-dir
  "Push directory into current directory context for execution of body."
  [dir & body]
  (apply list 'clojure.core/binding
    (vector 'clojure.tools.deps.util.dir/*the-dir*
      (list 'clojure.tools.deps.util.dir/canonicalize dir))
    body))

(defn- last-sep
  "Index of the last `/` or `\\` in p at or below from, or nil."
  [p from]
  (loop [i from]
    (cond (< i 0) nil
          (or (= (subs p i (inc i)) "/") (= (subs p i (inc i)) "\\")) i
          :else (recur (dec i)))))

(defn- parent-path
  "Parent of canonical path p, or nil at the stop (p is `/`, or p
  carries no separator below its last char)."
  [p]
  (let [n (count p)]
    (when (> n 1)
      (let [i (last-sep p (- n 2))]
        (cond (nil? i) nil
              (zero? i) "/"
              :else (subs p 0 i))))))

(defn sub-path?
  "True if the path is a sub-path of the current directory context.
  path may be either absolute or relative. Will return true if path
  has a parent that is the current directory context, false otherwise.
  Handles relative paths, .., ., etc. The sub-path does not need to
  exist on disk (but the current directory context must)."
  [path]
  (if (nil? path)
    false
    (if (port/file-exists? *the-dir*)
      (let [root (canonicalize *the-dir*)]
        (loop [check (canonicalize path)]
          (cond (nil? check) false
                (= root check) true
                :else (recur (parent-path check)))))
      false)))
