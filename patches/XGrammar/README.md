# XGrammar patches

These patches target XGrammar commit `97787376faee5ed8466cfad57c99855e4ce2f6aa` (v0.2.8) and are applied in order:

1. `0001-support-single-threaded-wasi.patch` adds synchronous compilation and cache behavior for plain `wasm32-unknown-wasip1`. Native platforms and `wasm32-unknown-wasip1-threads` retain XGrammar's threaded implementation.
2. `0002-support-wasi-without-cxx-exceptions.patch` avoids C++ exception-runtime dependencies that are unavailable in the Swift 6.3 WASI SDK. It also replaces exception-based Lark and EBNF number parsing and uses fatal logging for unrecoverable WASI parser errors.
