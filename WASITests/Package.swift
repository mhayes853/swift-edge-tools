// swift-tools-version: 6.3

import PackageDescription

let package = Package(
  name: "EdgeToolsWASITests",
  dependencies: [
    .package(
      path: "..",
      traits: ["XGrammar", "System", "Atomics"]
    )
  ],
  targets: [
    .testTarget(
      name: "EdgeToolsWASITests",
      dependencies: [
        .product(name: "EdgeTools", package: "swift-edge-tools")
      ]
    )
  ]
)
