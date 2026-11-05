#!/bin/bash
# Provision the instrumentation backend (Intel Pin + the vendored
# inscount tool) for the h2h gate.
#
# Idempotent: when INSTRUMENTATION_DIR already holds a kit and a
# built tool (the workflow caches it across runs), it just
# re-exports the paths. Otherwise it downloads the kit from
# Intel, verifies its sha256, extracts it, and builds
# scripts/ci/instrumentation/inscount.cpp against it.
#
# Outputs (stdout for logs, $GITHUB_ENV when present):
#   INSTRUMENTATION_BIN   absolute path to the pin launcher
#   INSTRUMENTATION_TOOL  absolute path to obj-intel64/inscount.so
set -euo pipefail

PIN_URL="${PIN_URL:-https://software.intel.com/sites/landingpage/pintool/downloads/pin-external-4.4-99977-g54fbb8814-gcc-linux.tar.gz}"
PIN_SHA256="${PIN_SHA256:-3138183a5d2b51ea5c8f29edd148b67ac9c249e2c510cc0e9e2a5880f523a34c}"
PIN_TOPDIR="pin-external-4.4-99977-g54fbb8814-gcc-linux"
INSTRUMENTATION_DIR="${INSTRUMENTATION_DIR:-$HOME/instrumentation}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_SRC="$SCRIPT_DIR"

KIT_DIR="$INSTRUMENTATION_DIR/$PIN_TOPDIR"
INSTRUMENTATION_BIN="$KIT_DIR/pin"
TOOL_BUILD="$INSTRUMENTATION_DIR/toolbuild"
INSTRUMENTATION_TOOL="$TOOL_BUILD/obj-intel64/inscount.so"

need_tool() {
  [ ! -x "$INSTRUMENTATION_BIN" ] || [ ! -f "$INSTRUMENTATION_TOOL" ]
}

err() { echo "setup-instrumentation: $*" >&2; }

if need_tool; then
  command -v g++ >/dev/null || { err "g++ is required to build the inscount tool"; exit 1; }
  command -v curl >/dev/null || { err "curl is required to fetch the Pin kit"; exit 1; }
  [ -f "$TOOL_SRC/inscount.cpp" ] || { err "tool sources missing at $TOOL_SRC"; exit 1; }
  mkdir -p "$INSTRUMENTATION_DIR"
  if [ ! -x "$INSTRUMENTATION_BIN" ]; then
    err "downloading Pin 4.4 kit..."
    curl -sSfL --retry 3 --max-time 600 -o "$INSTRUMENTATION_DIR/pin-kit.tar.gz" "$PIN_URL"
    echo "$PIN_SHA256  $INSTRUMENTATION_DIR/pin-kit.tar.gz" | sha256sum -c - || {
      err "kit checksum mismatch"; exit 1
    }
    tar xzf "$INSTRUMENTATION_DIR/pin-kit.tar.gz" -C "$INSTRUMENTATION_DIR"
    rm -f "$INSTRUMENTATION_DIR/pin-kit.tar.gz"
  fi
  err "building inscount tool..."
  rm -rf "$TOOL_BUILD"
  mkdir -p "$TOOL_BUILD"
  cp "$TOOL_SRC/inscount.cpp" "$TOOL_SRC/makefile" "$TOOL_SRC/makefile.rules" "$TOOL_BUILD/"
  make -C "$TOOL_BUILD" PIN_ROOT="$KIT_DIR" >/tmp/instrumentation-tool-build.log 2>&1 || {
    tail -n 20 /tmp/instrumentation-tool-build.log >&2; err "tool build failed"; exit 1
  }
fi

[ -x "$INSTRUMENTATION_BIN" ] || { err "pin launcher missing at $INSTRUMENTATION_BIN"; exit 1; }
[ -f "$INSTRUMENTATION_TOOL" ] || { err "tool missing at $INSTRUMENTATION_TOOL"; exit 1; }
"$INSTRUMENTATION_BIN" -version >/dev/null 2>&1 || { err "pin launcher does not run"; exit 1; }

echo "INSTRUMENTATION_BIN=$INSTRUMENTATION_BIN"
echo "INSTRUMENTATION_TOOL=$INSTRUMENTATION_TOOL"
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "INSTRUMENTATION_BIN=$INSTRUMENTATION_BIN" >>"$GITHUB_ENV"
  echo "INSTRUMENTATION_TOOL=$INSTRUMENTATION_TOOL" >>"$GITHUB_ENV"
fi
