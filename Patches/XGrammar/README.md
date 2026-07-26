# XGrammar patches

These patches target XGrammar commit `2ea71da4ccb997a06928c9fb69b99f330da56697` and are applied in order:

1. `0001-support-single-threaded-wasi.patch` adds synchronous compilation and cache behavior for plain `wasm32-unknown-wasip1`. Native platforms and `wasm32-unknown-wasip1-threads` retain XGrammar's threaded implementation.
2. `0002-support-wasi-without-cxx-exceptions.patch` avoids C++ exception-runtime dependencies that are unavailable in the Swift 6.3 WASI SDK. It also replaces exception-based Lark integer parsing and uses fatal logging for unrecoverable WASI parser errors.
