#!/usr/bin/env bash
# Probe (throwaway, #271): where do AOT binaries spend their Linux startup
# floor and small-input work, next to C / Rust / OCaml baselines?
# Callgrind Ir (≈ the h2h gate's Pin count), per-function inclusive and
# self costs, heap peak (massif), page-level RSS (massif --pages-as-heap),
# and GNU time max RSS for each program.
set -euo pipefail
root=$(pwd)
out=/tmp/out
mkdir -p "$out" /tmp/w
cd /tmp/w
cljrs="$root/rust/target/release/clojure-rs"

# --- programs ---------------------------------------------------------
cat > hello.clj <<'EOF'
(println 1)
EOF
cat > sum.clj <<'EOF'
(require '[clojure.string :as str])
(let [n (Long/parseLong (str/trim (read-line)))
      xs (mapv #(Long/parseLong %) (str/split (str/trim (read-line)) #"\s+"))]
  (println (reduce + 0 (take n xs))))
EOF
cat > grid.clj <<'EOF'
(require '[clojure.string :as str])
(defn solve [rows]
  (let [h (count rows) w (count (first rows))]
    (loop [r 0 acc 0]
      (if (< r h)
        (recur (inc r) (+ acc (count (filter #(= \# %) (nth rows r)))))
        acc))))
(let [[h _] (map #(Long/parseLong %) (str/split (read-line) #" "))
      rows (vec (repeatedly h read-line))]
  (println (solve rows)))
EOF
printf '5\n1 2 3 4 5\n' > sum.in
printf '3 4\n#..#\n.##.\n....\n' > grid.in
printf '' > empty.in
cat > hello.ml <<'EOF'
let () = print_endline "1"
EOF
cat > sum.ml <<'EOF'
let () =
  let n = int_of_string (String.trim (input_line stdin)) in
  let xs = String.split_on_char ' ' (String.trim (input_line stdin)) |> List.filter (fun s -> s <> "") |> List.map int_of_string in
  let rec go i acc = function [] -> acc | x :: r -> if i < n then go (i + 1) (acc + x) r else acc in
  print_int (go 0 0 xs); print_newline ()
EOF
printf 'int main(void){return 0;}\n' > empty.c
printf 'fn main(){}\n' > empty.rs
printf 'fn main(){println!("1");}\n' > hello.rs
printf 'fn main(){let mut s=String::new();std::io::stdin().read_line(&mut s).unwrap();println!("{}",s.len());}\n' > readline.rs

gcc -O2 -o c_empty empty.c
gcc -O2 -static -o c_empty_static empty.c || true
rustc -O -o rs_empty empty.rs
rustc -O -o rs_hello hello.rs
rustc -O -o rs_readline readline.rs
ocamlopt -O2 -o ml_hello hello.ml 2>/dev/null || ocamlopt -o ml_hello hello.ml
ocamlopt -O2 -o ml_sum sum.ml 2>/dev/null || ocamlopt -o ml_sum sum.ml
for p in hello sum grid; do
  "$cljrs" build --report "$p.clj" -o "aot_$p" 2>&1 | tail -1
done
"$cljrs" build hello.clj -o /dev/null --emit-ir "$out/aot_hello.ll"

# --- measurements -----------------------------------------------------
measure() { # name input cmd...
  local name=$1 input=$2; shift 2
  local rss
  rss=$(for _ in 1 2 3; do { /usr/bin/time -f '%M' "$@" < "$input" > /dev/null; } 2>&1 | tail -1; done | sort -n | head -1)
  valgrind --tool=callgrind --callgrind-out-file="$out/cg.$name" "$@" < "$input" > /dev/null 2> "$out/cg.$name.log" || true
  local ir
  ir=$(grep -E '^(summary|totals):' "$out/cg.$name" | head -1 | awk '{print $2}')
  callgrind_annotate --inclusive=yes "$out/cg.$name" 2>/dev/null | head -70 > "$out/incl.$name.txt" || true
  callgrind_annotate "$out/cg.$name" 2>/dev/null | head -70 > "$out/self.$name.txt" || true
  valgrind --tool=massif --massif-out-file="$out/massif.$name" "$@" < "$input" > /dev/null 2>/dev/null || true
  valgrind --tool=massif --pages-as-heap=yes --massif-out-file="$out/pages.$name" "$@" < "$input" > /dev/null 2>/dev/null || true
  local heap pages
  heap=$(ms_print "$out/massif.$name" 2>/dev/null | awk '/^ +[0-9]+ +[0-9,]+ +[0-9,]+ +[0-9,]+/{gsub(",","",$3); if ($3+0>m) m=$3+0} END{print m+0}')
  pages=$(ms_print "$out/pages.$name" 2>/dev/null | awk '/^ +[0-9]+ +[0-9,]+ +[0-9,]+ +[0-9,]+/{gsub(",","",$3); if ($3+0>m) m=$3+0} END{print m+0}')
  printf '%-16s ir=%-10s maxrss_kb=%-6s heap_peak_b=%-9s pages_peak_b=%s\n' "$name" "$ir" "$rss" "$heap" "$pages"
}

echo "== floors and programs"
measure c_empty empty.in ./c_empty
[ -x c_empty_static ] && measure c_static empty.in ./c_empty_static
measure rs_empty empty.in ./rs_empty
measure rs_hello empty.in ./rs_hello
measure ml_hello empty.in ./ml_hello
measure ml_sum sum.in ./ml_sum
measure aot_hello empty.in ./aot_hello
measure aot_sum sum.in ./aot_sum
measure aot_grid grid.in ./aot_grid

echo "== binary layout"
for b in c_empty rs_empty ml_hello aot_hello aot_sum; do
  echo "-- $b"; size "$b"; ldd "$b" 2>&1 | sed 's/^/   /'
done
readelf -S -W aot_hello | awk '$2 ~ /^\./ {print $2, $6}' | sort -k2 -r | head -20 > "$out/sections.aot_hello.txt" || true
nm --size-sort -S -C aot_hello 2>/dev/null | tail -60 > "$out/biggest-symbols.aot_hello.txt" || true

echo "== top inclusive (aot_hello)"; head -45 "$out/incl.aot_hello" 2>/dev/null || head -45 "$out/incl.aot_hello.txt"
echo "== top self (aot_hello)"; head -45 "$out/self.aot_hello.txt"
echo "== top self (aot_sum)"; head -45 "$out/self.aot_sum.txt"
echo "== top self (ml_sum)"; head -30 "$out/self.ml_sum.txt"

# --- post-boot RSS attribution: block each program on a FIFO stdin,
# snapshot /proc/PID/smaps, then release it.
snap() { # name cmd...
  local name=$1; shift
  rm -f /tmp/w/fifo; mkfifo /tmp/w/fifo
  "$@" < /tmp/w/fifo > /dev/null 2>&1 &
  local pid=$!
  exec 3> /tmp/w/fifo
  sleep 1
  cp "/proc/$pid/smaps" "$out/smaps.$name.txt" 2>/dev/null || true
  echo "-- $name (pid $pid) post-boot rollup"
  grep -E '^(Rss|Pss|Shared_Clean|Shared_Dirty|Private_Clean|Private_Dirty|Anonymous):' "/proc/$pid/smaps_rollup" 2>/dev/null || true
  echo "   per-mapping Rss (kB), largest first:"
  awk '/^[0-9a-f]+-[0-9a-f]+ /{name=($6==""?"[anon]":$6)} /^Rss:/{rss[name]+=$2} END{for(n in rss) if (rss[n]>0) printf "   %7d %s\n", rss[n], n}' "$out/smaps.$name.txt" | sort -rn | head -14
  exec 3>&-
  wait "$pid" || true
}
echo "== post-boot RSS"
snap rs_readline ./rs_readline
snap ml_sum ./ml_sum
snap aot_sum ./aot_sum
snap aot_grid ./aot_grid
