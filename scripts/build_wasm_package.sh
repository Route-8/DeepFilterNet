#!/bin/sh -ex

# look at DeepFilterNet/.github/workflows/build_wasm.yml for enviroment setup
cd ./libDF/
RUSTFLAGS="-C target-feature=+simd128,+bulk-memory" wasm-pack build --no-opt --profile release-lto --target no-modules --features wasm,wasm-simd
