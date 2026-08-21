// swift-tools-version: 6.3;(experimentalCGen)

import PackageDescription

let package = Package(
  name: "edge-tools-benchmarks",
  platforms: [.macOS(.v14)],
  dependencies: [
    .package(
      name: "swift-edge-tools",
      path: "../..",
      traits: ["Foundation", "XGrammar", "MLX", "Llama"]
    ),
    .package(
      name: "benchmark",
      url: "https://github.com/ordo-one/package-benchmark",
      from: "1.9.2"
    ),
    .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.4"),
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3")
  ],
  targets: [
    .executableTarget(
      name: "CPUSamplingBenchmarks",
      dependencies: [
        .product(name: "EdgeTools", package: "swift-edge-tools"),
        .product(name: "Benchmark", package: "benchmark")
      ],
      path: "Benchmarks/CPUSamplingBenchmarks",
      plugins: [.plugin(name: "BenchmarkPlugin", package: "benchmark")]
    ),
    .executableTarget(
      name: "MLXSamplingBenchmarks",
      dependencies: [
        .product(name: "EdgeTools", package: "swift-edge-tools"),
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "Benchmark", package: "benchmark")
      ],
      path: "Benchmarks/MLXSamplingBenchmarks",
      plugins: [.plugin(name: "BenchmarkPlugin", package: "benchmark")]
    )
  ],
  swiftLanguageModes: [.v6]
)
