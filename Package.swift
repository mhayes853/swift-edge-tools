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
    .library(name: "Needle", targets: ["Needle"])
  ],
  traits: [
    .trait(name: "SwiftNeedleXGrammar", description: "XGrammar-powered structured generation."),
    .trait(
      name: "SwiftNeedleTokenizers",
      description: "Hugging Face tokenizers support via swift-transformers."
    ),
    .trait(
      name: "SwiftNeedleSentencepiece",
      description: "Sentencepiece tokenizer implementation.",
      enabledTraits: ["SwiftNeedleTokenizers"]
    ),
    .trait(
      name: "SwiftNeedleMLX",
      description: "MLX Support.",
      enabledTraits: ["SwiftNeedleSentencepiece"]
    ),
    .default(
      enabledTraits: [
        "SwiftNeedleXGrammar",
        "SwiftNeedleMLX",
        "SwiftNeedleSentencepiece",
        "SwiftNeedleTokenizers"
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
    .package(url: "https://github.com/mlc-ai/xgrammar", from: "0.2.2")
  ],
  targets: [
    .target(
      name: "Needle",
      dependencies: [
        "NeedleMacros",
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
        .target(name: "CNeedleXGrammar", condition: .when(traits: ["SwiftNeedleXGrammar"])),
        .product(
          name: "Tokenizers",
          package: "swift-transformers",
          condition: .when(traits: ["SwiftNeedleTokenizers"])
        ),
        .product(
          name: "Hub",
          package: "swift-transformers",
          condition: .when(traits: ["SwiftNeedleTokenizers"])
        ),
        .target(
          name: "CNeedleSentencepiece",
          condition: .when(traits: ["SwiftNeedleSentencepiece"])
        )
      ],
      swiftSettings: [.enableExperimentalFeature("LifetimeDependence")]
    ),
    .target(
      name: "CNeedleXGrammar",
      dependencies: [.product(name: "XGrammar", package: "xgrammar")],
      publicHeadersPath: "include"
    ),
    .target(
      name: "CNeedleSentencepiece",
      path: "Sources/CNeedleSentencepiece",
      sources: [
        "bridging.cc",
        "sentencepiece/src/bpe_model.cc",
        "sentencepiece/src/char_model.cc",
        "sentencepiece/src/error.cc",
        "sentencepiece/src/filesystem.cc",
        "sentencepiece/src/model_factory.cc",
        "sentencepiece/src/model_interface.cc",
        "sentencepiece/src/normalizer.cc",
        "sentencepiece/src/sentencepiece_processor.cc",
        "sentencepiece/src/unigram_model.cc",
        "sentencepiece/src/util.cc",
        "sentencepiece/src/word_model.cc",
        "sentencepiece/src/builtin_pb/sentencepiece.pb.cc",
        "sentencepiece/src/builtin_pb/sentencepiece_model.pb.cc",
        "sentencepiece/src/builtin_pb/sentencepiece.pb.h",
        "sentencepiece/src/builtin_pb/sentencepiece_model.pb.h",
        "sentencepiece/third_party/absl/flags/flag.cc",
        "sentencepiece/third_party/protobuf-lite/arena.cc",
        "sentencepiece/third_party/protobuf-lite/arenastring.cc",
        "sentencepiece/third_party/protobuf-lite/bytestream.cc",
        "sentencepiece/third_party/protobuf-lite/coded_stream.cc",
        "sentencepiece/third_party/protobuf-lite/common.cc",
        "sentencepiece/third_party/protobuf-lite/extension_set.cc",
        "sentencepiece/third_party/protobuf-lite/generated_enum_util.cc",
        "sentencepiece/third_party/protobuf-lite/generated_message_table_driven_lite.cc",
        "sentencepiece/third_party/protobuf-lite/generated_message_util.cc",
        "sentencepiece/third_party/protobuf-lite/implicit_weak_message.cc",
        "sentencepiece/third_party/protobuf-lite/int128.cc",
        "sentencepiece/third_party/protobuf-lite/io_win32.cc",
        "sentencepiece/third_party/protobuf-lite/message_lite.cc",
        "sentencepiece/third_party/protobuf-lite/parse_context.cc",
        "sentencepiece/third_party/protobuf-lite/repeated_field.cc",
        "sentencepiece/third_party/protobuf-lite/status.cc",
        "sentencepiece/third_party/protobuf-lite/statusor.cc",
        "sentencepiece/third_party/protobuf-lite/stringpiece.cc",
        "sentencepiece/third_party/protobuf-lite/stringprintf.cc",
        "sentencepiece/third_party/protobuf-lite/structurally_valid.cc",
        "sentencepiece/third_party/protobuf-lite/strutil.cc",
        "sentencepiece/third_party/protobuf-lite/time.cc",
        "sentencepiece/third_party/protobuf-lite/wire_format_lite.cc",
        "sentencepiece/third_party/protobuf-lite/zero_copy_stream.cc",
        "sentencepiece/third_party/protobuf-lite/zero_copy_stream_impl.cc",
        "sentencepiece/third_party/protobuf-lite/zero_copy_stream_impl_lite.cc"
      ],
      publicHeadersPath: "include",
      cxxSettings: [
        .define("HAVE_PTHREAD=1"),
        .headerSearchPath("."),
        .headerSearchPath("sentencepiece"),
        .headerSearchPath("sentencepiece/src"),
        .headerSearchPath("sentencepiece/src/builtin_pb"),
        .headerSearchPath("sentencepiece/third_party/protobuf-lite")
      ]
    ),
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
        .product(name: "CustomDump", package: "swift-custom-dump"),
        .product(
          name: "Hub",
          package: "swift-transformers",
          condition: .when(traits: ["SwiftNeedleTokenizers"])
        )
      ],
      exclude: ["NeedleGenerationSchemaTests/__Snapshots__"],
      resources: [.process("Resources")]
    )
  ],
  swiftLanguageModes: [.v6],
  cxxLanguageStandard: .cxx20
)
