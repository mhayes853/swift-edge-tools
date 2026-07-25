# XGrammar patches

`0001-support-single-threaded-wasi.patch` targets XGrammar commit `2ea71da4ccb997a06928c9fb69b99f330da56697` to add single-threaded support for `wasm32-unknown-wasip1` environments. `wasm32-unknown-wasip1-threads` has native thread support, and therefore continues to use XGrammar's thread pool.
