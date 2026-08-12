// swift-tools-version: 6.3

import PackageDescription

let package = Package(
  name: "EdgeToolsWASITests",
  dependencies: [
    .package(
      path: "../..",
      traits: ["XGrammar", "ONNXCore", "JS"]
    ),
    .package(
      url: "https://github.com/swiftwasm/JavaScriptKit",
      exact: "0.56.1"
    )
  ],
  targets: [
    .testTarget(
      name: "EdgeToolsWASITests",
      dependencies: [
        .product(name: "EdgeTools", package: "swift-edge-tools"),
        .product(name: "JavaScriptKit", package: "JavaScriptKit"),
        .product(name: "JavaScriptEventLoop", package: "JavaScriptKit"),
        .product(name: "JavaScriptEventLoopTestSupport", package: "JavaScriptKit"),
        .product(name: "JavaScriptBigIntSupport", package: "JavaScriptKit")
      ],
      exclude: ["Fixtures"]
    )
  ]
)
