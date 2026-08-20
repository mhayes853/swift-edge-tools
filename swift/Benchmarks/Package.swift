// swift-tools-version: 6.3;(experimentalCGen)

import PackageDescription

let package = Package(
  name: "edge-tools-benchmarks",
  platforms: [.macOS(.v14)],
  dependencies: [
    .package(
      name: "swift-edge-tools",
      path: "../..",
      traits: ["Foundation", "XGrammar", "Llama"]
    ),
    .package(
      name: "benchmark",
      url: "https://github.com/ordo-one/package-benchmark",
      from: "1.9.2"
    )
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
    )
  ],
  swiftLanguageModes: [.v6]
)
