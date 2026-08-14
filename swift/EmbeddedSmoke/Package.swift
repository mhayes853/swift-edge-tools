// swift-tools-version: 6.4

import PackageDescription

let package = Package(
  name: "EdgeToolsEmbeddedSmoke",
  dependencies: [
    .package(name: "swift-edge-tools", path: "../..", traits: []),
    .package(url: "https://github.com/apple/swift-collections", from: "1.6.0")
  ],
  targets: [
    .executableTarget(
      name: "EdgeToolsEmbeddedSmoke",
      dependencies: [
        .product(name: "EdgeTools", package: "swift-edge-tools"),
        .product(name: "OrderedCollections", package: "swift-collections")
      ],
      // Embedded Swift keeps the Unicode tables backing String comparison in a separate archive.
      linkerSettings: [.linkedLibrary("swiftUnicodeDataTables")]
    )
  ]
)
