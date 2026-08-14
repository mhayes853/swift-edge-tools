// swift-tools-version: 6.3

import Foundation
import PackageDescription

let needle2Only = ProcessInfo.processInfo.environment["EDGE_TOOLS_WASI_NEEDLE2_ONLY"] == "1"
let edgeToolsTraits: Set<Package.Dependency.Trait> =
  needle2Only
  ? ["JS", "Needle2"]
  : ["XGrammar", "JS", "Needle2"]
let testSources = needle2Only ? ["Needle2JSEngineTests.swift"] : nil
let testExcludes =
  needle2Only
  ? ["XGrammarWASITests.swift"]
  : []

let package = Package(
  name: "EdgeToolsWASITests",
  dependencies: [
    .package(
      name: "swift-edge-tools",
      path: "../..",
      traits: edgeToolsTraits
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
      exclude: testExcludes,
      sources: testSources
    )
  ]
)
