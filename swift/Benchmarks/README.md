# Benchmarks

Formal benchmarks for performance sensitive parts of the core framework that don't have to do with model performance. The CLI has a model performance benchmark harness in place, this exists for library functions.

Run the CPU sampling suite with:

```sh
swift package benchmark --target CPUSamplingBenchmarks
```

Run the MLX fused-versus-upstream sampling suite on macOS with:

```sh
swift package --disable-sandbox benchmark --target MLXSamplingBenchmarks
```

The MLX suite measures a complete sampling step, including prompt-backed penalty state and the
device-to-host synchronization needed to consume the sampled token. It does not include model
inference or grammar masking. Disabling SwiftPM's plugin sandbox is required so Metal devices are
visible to the benchmark subprocess.
