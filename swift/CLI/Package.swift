// swift-tools-version: 6.3;(experimentalCGen)

import PackageDescription

let package = Package(
  name: "edge",
  platforms: [.macOS(.v15)],
  products: [
    .executable(name: "edge", targets: ["edge"])
  ],
  dependencies: [
    .package(
      name: "swift-edge-tools",
      path: "../..",
      traits: ["MLX", "CoreML", "CoreAI", "ONNX"]
    ),
    .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.3"),
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
    .package(url: "https://github.com/pointfreeco/swift-custom-dump", from: "1.6.1"),
    .package(url: "https://github.com/pointfreeco/swift-snapshot-testing", from: "1.18.0")
  ],
  targets: [
    .executableTarget(
      name: "edge",
      dependencies: ["EdgeToolsCLI"],
      path: "Sources/edge"
    ),
    .target(
      name: "EdgeToolsCLI",
      dependencies: [
        .product(name: "EdgeTools", package: "swift-edge-tools"),
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "Hub", package: "swift-transformers"),
        .product(name: "Tokenizers", package: "swift-transformers"),
        .product(name: "MLX", package: "mlx-swift", condition: .when(platforms: [.macOS])),
        .product(name: "MLXNN", package: "mlx-swift", condition: .when(platforms: [.macOS])),
        .product(name: "MLXLLM", package: "mlx-swift-lm", condition: .when(platforms: [.macOS])),
        .product(
          name: "MLXHuggingFace",
          package: "mlx-swift-lm",
          condition: .when(platforms: [.macOS])
        ),
        .product(
          name: "MLXLMCommon",
          package: "mlx-swift-lm",
          condition: .when(platforms: [.macOS])
        )
      ],
      path: "Sources/EdgeToolsCLI"
    ),
    .testTarget(
      name: "EdgeToolsCLITests",
      dependencies: [
        "EdgeToolsCLI",
        .product(name: "CustomDump", package: "swift-custom-dump"),
        .product(name: "SnapshotTesting", package: "swift-snapshot-testing")
      ],
      path: "Tests/EdgeToolsCLITests",
      exclude: ["__Snapshots__"]
    )
  ],
  swiftLanguageModes: [.v6],
  cxxLanguageStandard: .cxx17
)
