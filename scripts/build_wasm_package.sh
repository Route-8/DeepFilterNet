#!/bin/sh -ex

# look at DeepFilterNet/.github/workflows/build_wasm.yml for enviroment setup
cd ./libDF/
RUSTFLAGS="-C target-feature=+simd128,+bulk-memory" wasm-pack build --no-opt --profile release-lto --target no-modules --features wasm,wasm-simd
# wasm-pack's built-in wasm-opt invocation is skipped (--no-opt) so the binaryen feature flags
# stay explicit and deterministic. The enable flags cover the features rustc emits by default
# for wasm32-unknown-unknown plus the simd128/bulk-memory target features set above.
# --fast-math is deliberately omitted to keep IEEE semantics.
wasm-opt -O4 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int \
    --enable-sign-ext --enable-mutable-globals --enable-reference-types --enable-multivalue \
    pkg/df_bg.wasm -o pkg/df_bg.wasm
