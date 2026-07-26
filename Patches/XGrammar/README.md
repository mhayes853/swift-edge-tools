# XGrammar patches

These patches target XGrammar commit `2ea71da4ccb997a06928c9fb69b99f330da56697` and are applied in order:

1. `0001-support-single-threaded-wasi.patch` adds synchronous compilation and cache behavior for plain `wasm32-unknown-wasip1`. Native platforms and `wasm32-unknown-wasip1-threads` retain XGrammar's threaded implementation.
2. `0002-support-wasi-without-cxx-exceptions.patch` avoids C++ exception-runtime dependencies that are unavailable in the Swift 6.3 WASI SDK. It also replaces exception-based Lark integer parsing and uses fatal logging for unrecoverable WASI parser errors.

The generic patch plugin prepends the exception compatibility header to generated C++ translation units after applying both patches. This keeps the exception patch focused instead of repeating an identical include hunk for every source file. Native builds retain XGrammar's exception behavior. On WASI, fatal XGrammar checks terminate the process rather than attempting to throw through an unavailable C++ exception runtime.
