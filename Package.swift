// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import CompilerPluginSupport
import PackageDescription

let package = Package(
  name: "swift-edge-tools",
  platforms: [
    .macOS(.v14),
    .iOS(.v17),
    .tvOS(.v17),
    .watchOS(.v10),
    .visionOS(.v1)
  ],
  products: [
    .library(name: "EdgeTools", targets: ["EdgeTools"])
  ],
  traits: [
    .trait(name: "Foundation", description: "Foundation-specific conveniences."),
    .trait(name: "System", description: "swift-system FilePath-based file I/O."),
    .trait(name: "Atomics", description: "Atomic engine generation coordination."),
    .trait(name: "XGrammar", description: "XGrammar-powered structured generation."),
    .trait(
      name: "Transformers",
      description: "swift-transformers tokenizer support.",
      enabledTraits: ["Foundation"]
    ),
    .trait(
      name: "FoundationModels",
      description: "Apple FoundationModels interoperability.",
      enabledTraits: ["Foundation"]
    ),
    .trait(
      name: "MLX",
      description: "MLX engine support.",
      enabledTraits: ["XGrammar", "Transformers", "Foundation", "Atomics"]
    ),
    .trait(
      name: "CoreAI",
      description: "CoreAI engine support (experimental).",
      enabledTraits: ["XGrammar", "Foundation", "Atomics"]
    ),
    .trait(
      name: "CoreML",
      description: "CoreML engine support.",
      enabledTraits: ["XGrammar", "Foundation", "Atomics"]
    ),
    .default(
      enabledTraits: [
        "Foundation", "XGrammar", "MLX", "FoundationModels", "CoreAI", "CoreML"
      ]
    )
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.3"),
    .package(url: "https://github.com/pointfreeco/swift-snapshot-testing", from: "1.18.7"),
    .package(url: "https://github.com/pointfreeco/swift-custom-dump", from: "1.3.3"),
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
    .package(url: "https://github.com/pointfreeco/swift-macro-testing", from: "0.6.5"),
    .package(url: "https://github.com/swiftlang/swift-syntax", "600.0.0"..<"603.0.0"),
    .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    .package(
      url: "https://github.com/ibireme/yyjson.git",
      revision: "de3700ab1778e236a8a571058463b6a5888cf262",
      traits: [
        "noIncrementalReader",
        "noUtilities",
        "noFastFloatingPoint",
        "strictStandardJSON"
      ]
    ),
    .package(url: "https://github.com/apple/swift-collections", from: "1.2.1"),
    .package(url: "https://github.com/apple/swift-atomics", from: "1.3.0"),
    .package(url: "https://github.com/apple/swift-system", from: "1.7.4")
  ],
  targets: [
    .target(
      name: "EdgeTools",
      dependencies: [
        "EdgeToolsMacros",
        .product(name: "yyjson", package: "yyjson"),
        .product(name: "HeapModule", package: "swift-collections"),
        .product(name: "OrderedCollections", package: "swift-collections"),
        .product(name: "Atomics", package: "swift-atomics", condition: .when(traits: ["Atomics"])),
        .product(name: "SystemPackage", package: "swift-system", condition: .when(traits: ["System"])),
        .product(name: "MLX", package: "mlx-swift", condition: .when(traits: ["MLX"])),
        .product(name: "MLXNN", package: "mlx-swift", condition: .when(traits: ["MLX"])),
        .product(name: "MLXLLM", package: "mlx-swift-lm", condition: .when(traits: ["MLX"])),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm", condition: .when(traits: ["MLX"])),
        .target(name: "CXGrammar", condition: .when(traits: ["XGrammar"])),
        .product(
          name: "Tokenizers",
          package: "swift-transformers",
          condition: .when(traits: ["Transformers"])
        )
      ],
    ),
    .target(
      name: "CXGrammar",
      path: "Sources/CXGrammar",
      sources: [
        "bridging.cc",
        "xgrammar/cpp/compiled_grammar.cc",
        "xgrammar/cpp/config.cc",
        "xgrammar/cpp/earley_parser.cc",
        "xgrammar/cpp/fsm.cc",
        "xgrammar/cpp/fsm_builder.cc",
        "xgrammar/cpp/grammar.cc",
        "xgrammar/cpp/grammar_builder.cc",
        "xgrammar/cpp/grammar_compiler.cc",
        "xgrammar/cpp/grammar_functor.cc",
        "xgrammar/cpp/grammar_matcher.cc",
        "xgrammar/cpp/grammar_parser.cc",
        "xgrammar/cpp/grammar_printer.cc",
        "xgrammar/cpp/json_schema_converter.cc",
        "xgrammar/cpp/json_schema_converter_ext.cc",
        "xgrammar/cpp/lark_converter.cc",
        "xgrammar/cpp/regex_converter.cc",
        "xgrammar/cpp/structural_tag.cc",
        "xgrammar/cpp/testing.cc",
        "xgrammar/cpp/tokenizer_info.cc",
        "xgrammar/cpp/support/logging.cc",
        "xgrammar/cpp/support/recursion_guard.cc"
      ],
      publicHeadersPath: "include",
      cxxSettings: [
        .headerSearchPath("."),
        .headerSearchPath("xgrammar/include"),
        .headerSearchPath("xgrammar/cpp"),
        .headerSearchPath("xgrammar/3rdparty/dlpack/include"),
        .headerSearchPath("xgrammar/3rdparty/picojson"),
        .define("XGRAMMAR_ENABLE_LOG_DEBUG", to: "0"),
        .define("XGRAMMAR_ENABLE_CPPTRACE", to: "0")
      ]
    ),
    .macro(
      name: "EdgeToolsMacros",
      dependencies: [
        .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
        .product(name: "SwiftSyntax", package: "swift-syntax"),
        .product(name: "SwiftSyntaxBuilder", package: "swift-syntax"),
        .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
        .product(name: "SwiftDiagnostics", package: "swift-syntax")
      ]
    ),
    .testTarget(
      name: "EdgeToolsMacrosTests",
      dependencies: [
        "EdgeToolsMacros", .product(name: "MacroTesting", package: "swift-macro-testing")
      ]
    ),
    .testTarget(
      name: "EdgeToolsTests",
      dependencies: [
        "EdgeTools",
        .product(name: "SnapshotTesting", package: "swift-snapshot-testing"),
        .product(name: "CustomDump", package: "swift-custom-dump"),
        .product(
          name: "Hub",
          package: "swift-transformers",
          condition: .when(traits: ["MLX"])
        )
      ],
      exclude: [
        "GenerationSchema/__Snapshots__",
        "Models/Needle/Engines/__Snapshots__",
        "Models/Needle/Engines/MLX/__Snapshots__",
        "Models/Needle/__Snapshots__",
        "Models/__Snapshots__"
      ],
      resources: [.process("Resources")]
    )
  ],
  swiftLanguageModes: [.v6],
  cxxLanguageStandard: .cxx20
)
