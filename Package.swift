// swift-tools-version: 6.3;(experimentalCGen)
// The swift-tools-version declares the minimum version of Swift required to build this package.

import CompilerPluginSupport
import PackageDescription

let package = Package(
  name: "swift-edge-tools",
  platforms: [.macOS(.v14), .iOS(.v17), .tvOS(.v17), .watchOS(.v10), .visionOS(.v1)],
  products: [
    .library(name: "EdgeTools", targets: ["EdgeTools"]),
    .library(name: "EdgeToolsXGrammar", targets: ["EdgeToolsXGrammar"])
  ],
  traits: [
    .trait(name: "Foundation", description: "Foundation-specific conveniences."),
    .trait(name: "Atomics", description: "Atomic engine generation coordination."),
    .trait(name: "JS", description: "JavaScriptKit interoperability."),
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
      description: "MLX model support.",
      enabledTraits: ["XGrammar", "Transformers", "Foundation", "Atomics"]
    ),
    .trait(
      name: "CoreAI",
      description: "CoreAI model support (experimental).",
      enabledTraits: ["XGrammar", "Foundation", "Atomics"]
    ),
    .trait(
      name: "CoreML",
      description: "CoreML model support.",
      enabledTraits: ["XGrammar", "Foundation", "Atomics"]
    ),
    .trait(
      name: "ONNXCore",
      description: """
        Needle ONNX model and runtime-provider protocols.

        (Only enable this trait if you want to use your own ONNX build. Otherwise, enable `ONNX` directly.)
        """,
      enabledTraits: ["XGrammar", "Atomics"]
    ),
    .trait(
      name: "ONNX",
      description: "Vendored ONNX Runtime support.",
      enabledTraits: ["ONNXCore", "Transformers"]
    ),
    .default(enabledTraits: ["Foundation"])
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.3"),
    .package(url: "https://github.com/pointfreeco/swift-snapshot-testing", from: "1.18.7"),
    .package(url: "https://github.com/pointfreeco/swift-custom-dump", from: "1.6.1"),
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
    .package(url: "https://github.com/pointfreeco/swift-macro-testing", from: "0.6.5"),
    .package(url: "https://github.com/swiftlang/swift-syntax", "600.0.0"..<"603.0.0"),
    .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    .package(
      url: "https://github.com/ibireme/yyjson.git",
      revision: "de3700ab1778e236a8a571058463b6a5888cf262",
      traits: ["noIncrementalReader", "noUtilities", "noFastFloatingPoint", "strictStandardJSON"]
    ),
    .package(url: "https://github.com/apple/swift-collections", from: "1.2.1"),
    .package(url: "https://github.com/apple/swift-atomics", from: "1.3.0"),
    .package(url: "https://github.com/swiftwasm/JavaScriptKit", exact: "0.56.1")
  ],
  targets: [
    .target(
      name: "EdgeTools",
      dependencies: [
        "EdgeToolsMacros",
        .target(
          name: "_EdgeToolsFoundation",
          condition: .when(traits: ["Foundation"])
        ),
        .product(name: "yyjson", package: "yyjson"),
        .product(name: "HeapModule", package: "swift-collections"),
        .product(name: "OrderedCollections", package: "swift-collections"),
        .product(name: "Atomics", package: "swift-atomics", condition: .when(traits: ["Atomics"])),
        .product(
          name: "JavaScriptKit",
          package: "JavaScriptKit",
          condition: .when(traits: ["JS"])
        ),
        .product(
          name: "JavaScriptEventLoop",
          package: "JavaScriptKit",
          condition: .when(traits: ["JS"])
        ),
        .product(
          name: "JavaScriptBigIntSupport",
          package: "JavaScriptKit",
          condition: .when(traits: ["JS"])
        ),
        .product(name: "MLX", package: "mlx-swift", condition: .when(traits: ["MLX"])),
        .product(name: "MLXNN", package: "mlx-swift", condition: .when(traits: ["MLX"])),
        .product(name: "MLXLLM", package: "mlx-swift-lm", condition: .when(traits: ["MLX"])),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm", condition: .when(traits: ["MLX"])),
        .target(name: "EdgeToolsXGrammar", condition: .when(traits: ["XGrammar"])),
        .target(
          name: "COnnxRuntime",
          condition: .when(platforms: [.macOS, .iOS, .linux, .android], traits: ["ONNXCore"])
        ),
        .product(
          name: "Tokenizers",
          package: "swift-transformers",
          condition: .when(
            // TODO: - watchOS has compilation issues in Hub https://github.com/huggingface/swift-huggingface/pull/58
            platforms: [.macOS, .iOS, .tvOS, .visionOS, .linux],
            traits: ["Transformers"]
          )
        )
      ],
      path: "swift/Sources/EdgeTools",
      swiftSettings: [
        .enableExperimentalFeature("Lifetimes"),
        .enableExperimentalFeature("AddressableTypes")
      ],
      linkerSettings: [.linkedFramework("CoreML", .when(platforms: [.macOS, .iOS], traits: ["ONNX"]))]
    ),
    .target(
      name: "_EdgeToolsFoundation",
      path: "swift/Sources/_EdgeToolsFoundation"
    ),
    .target(
      name: "EdgeToolsXGrammar",
      dependencies: ["CXGrammar"],
      path: "swift/Sources/EdgeToolsXGrammar"
    ),
    .target(
      name: "COnnxRuntime",
      dependencies: [
        .target(
          name: "onnxruntime",
          condition: .when(platforms: [.macOS, .iOS], traits: ["ONNX"])
        ),
        .target(
          name: "onnxruntimeNonApple",
          condition: .when(platforms: [.linux, .android], traits: ["ONNX"])
        )
      ],
      path: "swift/Sources/COnnxRuntime",
      exclude: ["LICENSE"],
      publicHeadersPath: "include"
    ),
    .binaryTarget(
      name: "onnxruntime",
      url: "https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.27.0.zip",
      checksum: "8c74edd600eafc3055de9e8f7a9602afee44ed516913cb5e132bca02cc34622c"
    ),
    .binaryTarget(
      name: "onnxruntimeNonApple",
      path: "bin/onnxruntime-webgpu-1.27.0.artifactbundle.zip"
    ),
    .target(
      name: "CXGrammar",
      path: "swift/Sources/CXGrammar",
      sources: ["bridging.cc"],
      publicHeadersPath: "include",
      cxxSettings: [
        .headerSearchPath("."),
        .headerSearchPath("xgrammar/include"),
        .headerSearchPath("xgrammar/cpp"),
        .headerSearchPath("xgrammar/cpp/support"),
        .headerSearchPath("xgrammar/3rdparty/dlpack/include"),
        .headerSearchPath("xgrammar/3rdparty/picojson"),
        .define("XGRAMMAR_ENABLE_LOG_DEBUG", to: "0"),
        .define("XGRAMMAR_ENABLE_CPPTRACE", to: "0"),
        .define("XGRAMMAR_CXX_EXCEPTIONS_ENABLED", to: "0", .when(platforms: [.wasi])),
        .define("PICOJSON_DISABLE_EXCEPTION", to: "1", .when(platforms: [.wasi])),
        .define("XGRAMMAR_LOG_CUSTOMIZE", to: "1", .when(platforms: [.wasi]))
      ],
      plugins: [.plugin(name: "PatchPlugin")]
    ),
    .plugin(name: "PatchPlugin", capability: .buildTool(), path: "swift/Plugins/PatchPlugin"),
    .macro(
      name: "EdgeToolsMacros",
      dependencies: [
        .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
        .product(name: "SwiftSyntax", package: "swift-syntax"),
        .product(name: "SwiftSyntaxBuilder", package: "swift-syntax"),
        .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
        .product(name: "SwiftParser", package: "swift-syntax"),
        .product(name: "SwiftDiagnostics", package: "swift-syntax")
      ],
      path: "swift/Sources/EdgeToolsMacros"
    ),
    .testTarget(
      name: "EdgeToolsMacrosTests",
      dependencies: [
        "EdgeToolsMacros", .product(name: "MacroTesting", package: "swift-macro-testing")
      ],
      path: "swift/Tests/EdgeToolsMacrosTests"
    ),
    .testTarget(
      name: "EdgeToolsTests",
      dependencies: [
        "EdgeTools",
        .target(
          name: "COnnxRuntime",
          condition: .when(platforms: [.macOS, .iOS, .linux, .android], traits: ["ONNX"])
        ),
        .product(name: "SnapshotTesting", package: "swift-snapshot-testing"),
        .product(name: "CustomDump", package: "swift-custom-dump"),
        .product(name: "Hub", package: "swift-transformers", condition: .when(traits: ["MLX"]))
      ],
      path: "swift/Tests/EdgeToolsTests",
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
  cxxLanguageStandard: .cxx17
)
