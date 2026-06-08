// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import CompilerPluginSupport
import PackageDescription

let package = Package(
  name: "swift-needle",
  platforms: [
    .macOS(.v14),
    .iOS(.v17),
    .tvOS(.v17),
    .watchOS(.v10),
    .visionOS(.v1)
  ],
  products: [
    .library(name: "Needle", targets: ["Needle"]),
    .library(name: "NeedleCore", targets: ["NeedleCore"])
  ],
  traits: [
    .trait(name: "SwiftNeedleXGrammar", description: "XGrammar-powered structured generation."),
    .trait(
      name: "SwiftNeedleSentencepiece",
      description: "Sentencepiece tokenizer implementation."
    ),
    .trait(
      name: "SwiftNeedleMLX",
      description: "MLX Support.",
      enabledTraits: ["SwiftNeedleSentencepiece"]
    ),
    .default(enabledTraits: ["SwiftNeedleXGrammar", "SwiftNeedleMLX", "SwiftNeedleSentencepiece"])
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.3"),
    .package(url: "https://github.com/pointfreeco/swift-snapshot-testing", from: "1.18.7"),
    .package(url: "https://github.com/pointfreeco/swift-custom-dump", from: "1.3.3"),
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
    .package(url: "https://github.com/mattt/swift-xgrammar", from: "0.1.0"),
    .package(url: "https://github.com/pointfreeco/swift-macro-testing", from: "0.6.5"),
    .package(url: "https://github.com/swiftlang/swift-syntax", "600.0.0"..<"603.0.0")
  ],
  targets: [
    .target(name: "Needle", dependencies: ["NeedleCore", "NeedleMacros"]),
    .target(
      name: "NeedleCore",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift", condition: .when(traits: ["SwiftNeedleMLX"])),
        .product(name: "MLXNN", package: "mlx-swift", condition: .when(traits: ["SwiftNeedleMLX"])),
        .product(
          name: "MLXLLM",
          package: "mlx-swift-lm",
          condition: .when(traits: ["SwiftNeedleMLX"])
        ),
        .product(
          name: "MLXLMCommon",
          package: "mlx-swift-lm",
          condition: .when(traits: ["SwiftNeedleMLX"])
        ),
        .product(
          name: "XGrammar",
          package: "swift-xgrammar",
          condition: .when(traits: ["SwiftNeedleXGrammar"])
        ),
        .target(name: "Sentencepiece", condition: .when(traits: ["SwiftNeedleSentencepiece"]))
      ],
      swiftSettings: [.enableExperimentalFeature("LifetimeDependence")]
    ),
    .binaryTarget(name: "Sentencepiece", path: "bin/sentencepiece.xcframework.zip"),
    .macro(
      name: "NeedleMacros",
      dependencies: [
        .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
        .product(name: "SwiftSyntax", package: "swift-syntax"),
        .product(name: "SwiftSyntaxBuilder", package: "swift-syntax"),
        .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
        .product(name: "SwiftDiagnostics", package: "swift-syntax")
      ]
    ),
    .testTarget(
      name: "NeedleMacrosTests",
      dependencies: ["NeedleMacros", .product(name: "MacroTesting", package: "swift-macro-testing")]
    ),
    .testTarget(
      name: "NeedleTests",
      dependencies: [
        "Needle",
        .product(name: "SnapshotTesting", package: "swift-snapshot-testing"),
        .product(name: "CustomDump", package: "swift-custom-dump")
      ],
      exclude: ["NeedleGenerationSchemaTests/__Snapshots__"],
      resources: [.copy("Resources")]
    )
  ],
  swiftLanguageModes: [.v6]
)
