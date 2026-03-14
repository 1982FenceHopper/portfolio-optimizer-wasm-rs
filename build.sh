#!/bin/bash
set -e

CRATE_NAME="slsqp-wasm"
OUT_DIR="dist"

echo "Building Library (Rust -> WASM)"
wasm-pack build --target web --out-dir dist

echo "WASM built to dist/"
echo "JS bindings: dist/${CRATE_NAME}.js"
echo "WASM binary: dist/${CRATE_NAME}_bg.wasm"