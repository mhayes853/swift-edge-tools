// swift-tools-version: 6.3;(experimentalCGen)
// The swift-tools-version declares the minimum version of Swift required to build this package.

import CompilerPluginSupport
import PackageDescription

var edgeToolsSwiftSettings: [SwiftSetting] = [
  .enableExperimentalFeature("Lifetimes"),
  .enableExperimentalFeature("AddressableTypes")
]
#if compiler(<6.4)
  edgeToolsSwiftSettings.append(.enableExperimentalFeature("SuppressedAssociatedTypes"))
#endif
let edgeToolsJavaScriptSwiftSettings = edgeToolsSwiftSettings + [
  .enableExperimentalFeature("Extern")
]

let package = Package(
  name: "swift-edge-tools",
  platforms: [.macOS(.v14), .iOS(.v17), .tvOS(.v17), .watchOS(.v10), .visionOS(.v1)],
  products: [
    .library(name: "EdgeTools", targets: ["EdgeTools"]),
    .library(name: "EdgeToolsLlama", targets: ["EdgeToolsLlama"]),
    .library(name: "EdgeToolsTokenizers", targets: ["EdgeToolsTokenizers"]),
    .library(name: "EdgeToolsXGrammar", targets: ["EdgeToolsXGrammar"])
  ],
  traits: [
    .trait(name: "FoundationEssentials", description: "Foundation Essentials conveniences."),
    .trait(
      name: "Foundation",
      description: "Full Foundation conveniences.",
      enabledTraits: ["FoundationEssentials"]
    ),
    .trait(name: "JS", description: "JavaScriptKit interoperability."),
    .trait(name: "XGrammar", description: "XGrammar-powered structured generation."),
    .trait(name: "Needle2", description: "Needle 2 engine support."),
    .trait(
      name: "ChatTemplates",
      description: "Jinja chat-template rendering support."
    ),
    .trait(
      name: "HuggingFaceTokenizers",
      description: "Hugging Face tokenizer and chat-template support.",
      enabledTraits: ["FoundationEssentials", "ChatTemplates"]
    ),
    .trait(
      name: "FoundationModels",
      description: "Apple FoundationModels interoperability.",
      enabledTraits: ["FoundationEssentials"]
    ),
    .trait(
      name: "MLX",
      description: "MLX model support.",
      enabledTraits: ["HuggingFaceTokenizers", "FoundationEssentials"]
    ),
    .trait(
      name: "Llama",
      description: "Vendored cactus-patched llama.cpp engine support.",
      enabledTraits: ["ChatTemplates"]
    ),
    .default(enabledTraits: ["Foundation"])
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.4"),
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
          condition: .when(traits: ["FoundationEssentials"])
        ),
        .target(
          name: "_EdgeToolsJavaScript",
          condition: .when(traits: ["JS"])
        ),
        .product(name: "yyjson", package: "yyjson"),
        .product(name: "HeapModule", package: "swift-collections"),
        .product(name: "OrderedCollections", package: "swift-collections"),
        .product(name: "Atomics", package: "swift-atomics"),
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
        .product(
          name: "MLX",
          package: "mlx-swift",
          condition: .when(platforms: [.macOS], traits: ["MLX"])
        ),
        .product(
          name: "MLXNN",
          package: "mlx-swift",
          condition: .when(platforms: [.macOS], traits: ["MLX"])
        ),
        .product(
          name: "MLXLLM",
          package: "mlx-swift-lm",
          condition: .when(platforms: [.macOS], traits: ["MLX"])
        ),
        .product(
          name: "MLXLMCommon",
          package: "mlx-swift-lm",
          condition: .when(platforms: [.macOS], traits: ["MLX"])
        ),
        .product(
          name: "MLXVLM",
          package: "mlx-swift-lm",
          condition: .when(platforms: [.macOS], traits: ["MLX"])
        ),
        .target(name: "EdgeToolsCore"),
        .target(
          name: "EdgeToolsLlama",
          condition: .when(
            platforms: [
              .macOS, .iOS, .tvOS, .watchOS, .visionOS, .linux, .android, .windows
            ],
            traits: ["Llama"]
          )
        ),
        .target(name: "EdgeToolsTokenizers"),
        .target(name: "EdgeToolsXGrammar", condition: .when(traits: ["XGrammar"])),
        .target(
          name: "CLlama",
          condition: .when(
            platforms: [
              .macOS, .iOS, .tvOS, .watchOS, .visionOS, .linux, .android, .windows
            ],
            traits: ["Llama"]
          )
        ),
        .target(
          name: "CNeedle2",
          condition: .when(
            platforms: [.macOS, .iOS, .tvOS, .watchOS, .linux, .windows, .android],
            traits: ["Needle2"]
          )
        )
      ],
      path: "swift/EdgeTools/Sources/EdgeTools",
      swiftSettings: edgeToolsSwiftSettings,
      linkerSettings: [
        .linkedLibrary(
          "c++",
          .when(
            platforms: [.macOS, .iOS, .tvOS, .watchOS],
            traits: ["Needle2"]
          )
        ),
        .linkedLibrary("c++", .when(platforms: [.linux], traits: ["Needle2"])),
        .linkedLibrary("c++_shared", .when(platforms: [.android], traits: ["Needle2"])),
        .linkedFramework(
          "Metal",
          .when(platforms: [.macOS, .iOS, .tvOS, .visionOS], traits: ["Llama"])
        ),
        .linkedFramework(
          "MetalKit",
          .when(platforms: [.macOS, .iOS, .tvOS, .visionOS], traits: ["Llama"])
        ),
        .linkedFramework(
          "Accelerate",
          .when(
            platforms: [.macOS, .iOS, .tvOS, .watchOS, .visionOS],
            traits: ["Llama"]
          )
        ),
        .linkedLibrary(
          "c++",
          .when(
            platforms: [.macOS, .iOS, .tvOS, .watchOS, .visionOS],
            traits: ["Llama"]
          )
        ),
        .linkedLibrary("stdc++", .when(platforms: [.linux], traits: ["Llama"])),
        .linkedLibrary("m", .when(platforms: [.linux], traits: ["Llama"])),
        .linkedLibrary("dl", .when(platforms: [.linux], traits: ["Llama"])),
        .linkedLibrary("pthread", .when(platforms: [.linux], traits: ["Llama"])),
        .linkedLibrary("c++_shared", .when(platforms: [.android], traits: ["Llama"])),
        .linkedLibrary("log", .when(platforms: [.android], traits: ["Llama"])),
        .linkedLibrary("dl", .when(platforms: [.android], traits: ["Llama"]))
      ]
    ),
    .target(
      name: "_EdgeToolsFoundation",
      path: "swift/EdgeTools/Sources/_EdgeToolsFoundation"
    ),
    .target(
      name: "EdgeToolsLlama",
      dependencies: [
        .target(
          name: "_EdgeToolsFoundation",
          condition: .when(traits: ["FoundationEssentials"])
        ),
        .target(
          name: "CLlama",
          condition: .when(
            platforms: [
              .macOS, .iOS, .tvOS, .watchOS, .visionOS, .linux, .android, .windows
            ]
          )
        )
      ],
      path: "swift/EdgeTools/Sources/EdgeToolsLlama",
      swiftSettings: edgeToolsSwiftSettings,
      linkerSettings: [
        .linkedFramework("Metal", .when(platforms: [.macOS, .iOS, .tvOS, .visionOS])),
        .linkedFramework("MetalKit", .when(platforms: [.macOS, .iOS, .tvOS, .visionOS])),
        .linkedFramework(
          "Accelerate",
          .when(platforms: [.macOS, .iOS, .tvOS, .watchOS, .visionOS])
        ),
        .linkedLibrary(
          "c++",
          .when(platforms: [.macOS, .iOS, .tvOS, .watchOS, .visionOS])
        ),
        .linkedLibrary("stdc++", .when(platforms: [.linux])),
        .linkedLibrary("m", .when(platforms: [.linux])),
        .linkedLibrary("dl", .when(platforms: [.linux])),
        .linkedLibrary("pthread", .when(platforms: [.linux])),
        .linkedLibrary("c++_shared", .when(platforms: [.android])),
        .linkedLibrary("log", .when(platforms: [.android])),
        .linkedLibrary("dl", .when(platforms: [.android]))
      ]
    ),
    .target(
      name: "EdgeToolsCore",
      dependencies: [
        .target(
          name: "_EdgeToolsFoundation",
          condition: .when(traits: ["FoundationEssentials"])
        ),
        .product(name: "yyjson", package: "yyjson"),
        .product(name: "OrderedCollections", package: "swift-collections")
      ],
      path: "swift/EdgeTools/Sources/EdgeToolsCore",
      swiftSettings: edgeToolsSwiftSettings
    ),
    .target(
      name: "EdgeToolsTokenizers",
      dependencies: [
        "EdgeToolsCore",
        .target(
          name: "_EdgeToolsFoundation",
          condition: .when(traits: ["FoundationEssentials"])
        ),
        .product(name: "OrderedCollections", package: "swift-collections"),
        .target(
          name: "EdgeToolsLlama",
          condition: .when(
            platforms: [
              .macOS, .iOS, .tvOS, .watchOS, .visionOS, .linux, .android, .windows
            ],
            traits: ["Llama"]
          )
        ),
        .target(name: "EdgeToolsXGrammar", condition: .when(traits: ["XGrammar"])),
        .target(
          name: "CTokenizers",
          condition: .when(
            platforms: [
              .macOS, .iOS, .tvOS, .watchOS, .visionOS, .linux, .android, .windows
            ],
            traits: ["HuggingFaceTokenizers"]
          )
        ),
        .target(
          name: "CMinja",
          condition: .when(
            platforms: [
              .macOS, .iOS, .tvOS, .watchOS, .visionOS, .linux, .android, .windows
            ],
            traits: ["ChatTemplates"]
          )
        ),
        .target(
          name: "CLlama",
          condition: .when(
            platforms: [
              .macOS, .iOS, .tvOS, .watchOS, .visionOS, .linux, .android, .windows
            ],
            traits: ["Llama"]
          )
        )
      ],
      path: "swift/EdgeTools/Sources/EdgeToolsTokenizers",
      swiftSettings: edgeToolsSwiftSettings,
      linkerSettings: [
        .linkedFramework(
          "Metal",
          .when(platforms: [.macOS, .iOS, .tvOS, .visionOS], traits: ["Llama"])
        ),
        .linkedFramework(
          "MetalKit",
          .when(platforms: [.macOS, .iOS, .tvOS, .visionOS], traits: ["Llama"])
        ),
        .linkedFramework(
          "Accelerate",
          .when(
            platforms: [.macOS, .iOS, .tvOS, .watchOS, .visionOS],
            traits: ["Llama"]
          )
        ),
        .linkedLibrary(
          "c++",
          .when(
            platforms: [.macOS, .iOS, .tvOS, .watchOS, .visionOS],
            traits: ["Llama"]
          )
        ),
        .linkedLibrary("stdc++", .when(platforms: [.linux], traits: ["Llama"])),
        .linkedLibrary("m", .when(platforms: [.linux], traits: ["Llama"])),
        .linkedLibrary("dl", .when(platforms: [.linux], traits: ["Llama"])),
        .linkedLibrary("pthread", .when(platforms: [.linux], traits: ["Llama"])),
        .linkedLibrary("c++_shared", .when(platforms: [.android], traits: ["Llama"])),
        .linkedLibrary("log", .when(platforms: [.android], traits: ["Llama"])),
        .linkedLibrary("dl", .when(platforms: [.android], traits: ["Llama"]))
      ]
    ),
    .target(
      name: "CMinja",
      path: "swift/EdgeTools/Sources/CMinja",
      exclude: ["minja/LICENSE", "minja/PIN", "nlohmann/PIN"],
      sources: ["bridging.cc"],
      publicHeadersPath: "include",
      cxxSettings: [
        .headerSearchPath(".")
      ]
    ),
    .target(
      name: "_EdgeToolsJavaScript",
      dependencies: [
        .product(name: "JavaScriptKit", package: "JavaScriptKit")
      ],
      path: "swift/EdgeTools/Sources/_EdgeToolsJavaScript",
      swiftSettings: edgeToolsJavaScriptSwiftSettings,
      plugins: [.plugin(name: "BridgeJS", package: "JavaScriptKit")]
    ),
    .target(
      name: "EdgeToolsXGrammar",
      dependencies: ["CXGrammar"],
      path: "swift/EdgeTools/Sources/EdgeToolsXGrammar"
    ),
    .binaryTarget(
      name: "CNeedle2",
      path: "bin/needle2-2.0.2.artifactbundle.zip"
    ),
    .binaryTarget(
      name: "CLlama",
      path: "bin/llama-b10076.artifactbundle.zip"
    ),
    .binaryTarget(
      name: "CTokenizers",
      path: "bin/tokenizers-0.1.0.artifactbundle.zip"
    ),
    .target(
      name: "CXGrammar",
      path: "swift/EdgeTools/Sources/CXGrammar",
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
    .plugin(
      name: "PatchPlugin",
      capability: .buildTool(),
      path: "swift/EdgeTools/Plugins/PatchPlugin"
    ),
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
      path: "swift/EdgeTools/Sources/EdgeToolsMacros"
    ),
    .testTarget(
      name: "EdgeToolsMacrosTests",
      dependencies: [
        "EdgeToolsMacros", .product(name: "MacroTesting", package: "swift-macro-testing")
      ],
      path: "swift/EdgeTools/Tests/EdgeToolsMacrosTests"
    ),
    .testTarget(
      name: "EdgeToolsLlamaTests",
      dependencies: [
        "EdgeToolsLlama",
        .product(name: "CustomDump", package: "swift-custom-dump")
      ],
      path: "swift/EdgeTools/Tests/EdgeToolsLlamaTests"
    ),
    .testTarget(
      name: "EdgeToolsTokenizersTests",
      dependencies: [
        "EdgeToolsTokenizers",
        .product(name: "CustomDump", package: "swift-custom-dump"),
        .target(name: "EdgeToolsXGrammar", condition: .when(traits: ["XGrammar"]))
      ],
      path: "swift/EdgeTools/Tests/EdgeToolsTokenizersTests",
      resources: [.process("Resources")]
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
          condition: .when(traits: ["HuggingFaceTokenizers"])
        )
      ],
      path: "swift/EdgeTools/Tests/EdgeToolsTests",
      exclude: [
        "GenerationSchema/__Snapshots__",
        "Models/Needle2/__Snapshots__",
        "Models/__Snapshots__",
        "Models/Gemma/__Snapshots__",
        "Models/LFM/__Snapshots__",
        "Models/Qwen/__Snapshots__",
        "Tokenization/__Snapshots__"
      ]
    )
  ],
  swiftLanguageModes: [.v6],
  cxxLanguageStandard: .cxx17
)
