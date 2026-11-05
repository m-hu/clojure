// Probe-only build script: extra link args for the clojure-rs binary.
fn main() {
    println!("cargo:rerun-if-env-changed=PROBE_LINK_ARGS");
    if let Ok(args) = std::env::var("PROBE_LINK_ARGS") {
        for a in args.split_whitespace() {
            println!("cargo:rustc-link-arg-bins={a}");
        }
    }
}
