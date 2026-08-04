import EdgeTools
import Foundation

#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI
#endif

#if CoreML && canImport(CoreML)
  import CoreML
#endif

private enum NeedleTestModelExport {
  static func export(
    backend: String?,
    outputDirectoryName: String,
    arguments: [String],
    expectedFilesExist: (URL) -> Bool,
    errorMessage: (String) -> any Error
  ) async throws -> URL {
    let outputDirectory = URL.swiftEdgeToolsTestsDirectory.appending(path: outputDirectoryName)
    if expectedFilesExist(outputDirectory) {
      return outputDirectory
    }

    let backendName = backend ?? "CoreAI"
    print("=== Exporting Test \(backendName) Model ===")
    try FileManager.default.createDirectory(
      at: outputDirectory.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )

    let pythonDirectory = Self.pythonDirectory()
    let pythonExecutable = Self.pythonExecutable(in: pythonDirectory)

    let process = Process()
    process.executableURL = pythonExecutable
    process.currentDirectoryURL = pythonDirectory
    process.arguments = Self.processArguments(
      backend: backend,
      outputDirectory: outputDirectory,
      additionalArguments: arguments
    )

    let outputPipe = Pipe()
    process.standardOutput = outputPipe
    process.standardError = outputPipe
    try process.run()
    let output =
      String(
        decoding: outputPipe.fileHandleForReading.readDataToEndOfFile(),
        as: UTF8.self
      )
    process.waitUntilExit()

    guard process.terminationStatus == 0 else {
      throw errorMessage(output)
    }
    guard expectedFilesExist(outputDirectory) else {
      throw errorMessage("\(backendName) export did not produce the expected files.")
    }
    print("=== Exported Test \(backendName) Model Successfully ===")
    return outputDirectory
  }

  static func hasTokenizer(in directory: URL) -> Bool {
    ["tokenizer.model", "tokenizer.json"]
      .contains { FileManager.default.fileExists(atPath: directory.appending(path: $0).path()) }
  }

  private static func processArguments(
    backend: String?,
    outputDirectory: URL,
    additionalArguments: [String]
  ) -> [String] {
    var arguments = ["cli.py"]
    if let backend {
      arguments += ["--backend", backend]
    }
    arguments += ["--output", outputDirectory.path()]
    arguments += additionalArguments
    return arguments
  }

  private static func pythonDirectory() -> URL {
    let packageDirectory =
      URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
    return packageDirectory.appending(path: "python")
  }

  private static func pythonExecutable(in pythonDirectory: URL) -> URL {
    let venvExecutable = pythonDirectory.appending(path: ".venv/bin/python")
    if FileManager.default.isExecutableFile(atPath: venvExecutable.path()) {
      return venvExecutable
    }
    return URL(fileURLWithPath: "/usr/bin/python3")
  }
}

#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  @available(anyAppleOS 27.0, *)
  func exportNeedleCoreAI(
    outputDirectoryName: String = "coreai-export",
    arguments: [String] = []
  ) async throws -> URL {
    try await NeedleTestModelExport.export(
      backend: nil,
      outputDirectoryName: outputDirectoryName,
      arguments: arguments,
      expectedFilesExist: SelfCoreAIExport.filesExist(in:),
      errorMessage: { CoreAIModelExportError(message: $0) }
    )
  }

  @available(anyAppleOS 27.0, *)
  func makeNeedleCoreAIModelEngine(
    quantizerPreset: String? = nil,
    palettizerBits: Int? = nil,
    compilePlatforms: [String] = []
  ) async throws -> NeedleCoreAIModelEngine {
    var arguments = quantizerPreset.map { ["--quantizer-preset", $0] } ?? []
    arguments += palettizerBits.map { ["--palettizer-n-bits", String($0)] } ?? []
    arguments += compilePlatforms.flatMap { ["--compile-platform", $0] }
    let directory = try await exportNeedleCoreAI(
      outputDirectoryName: SelfCoreAIExport.outputDirectoryName(
        quantizerPreset: quantizerPreset,
        palettizerBits: palettizerBits,
        compilePlatforms: compilePlatforms
      ),
      arguments: arguments
    )
    return try await NeedleCoreAIModelEngine(modelDirectoryURL: directory)
  }

  private enum SelfCoreAIExport {
    static func filesExist(in directory: URL) -> Bool {
      guard NeedleTestModelExport.hasTokenizer(in: directory) else { return false }

      let fileManager = FileManager.default
      let contents = (try? fileManager.contentsOfDirectory(atPath: directory.path())) ?? []
      let hasEncoderModel =
        contents.contains("encoder.aimodel")
        || contents.contains { $0.hasPrefix("encoder.") && $0.hasSuffix(".aimodelc") }
      let hasDecoderModel =
        contents.contains("decoder.aimodel")
        || contents.contains { $0.hasPrefix("decoder.") && $0.hasSuffix(".aimodelc") }
      return hasEncoderModel
        && hasDecoderModel
        && fileManager.fileExists(atPath: directory.appending(path: "configuration.json").path())
    }

    static func outputDirectoryName(
      quantizerPreset: String?,
      palettizerBits: Int?,
      compilePlatforms: [String]
    ) -> String {
      let compileSuffix =
        compilePlatforms.isEmpty
        ? ""
        : "-aot-" + compilePlatforms.joined(separator: "-")
      let compressionSuffix = [
        quantizerPreset,
        palettizerBits.map { "palette\($0)" }
      ]
      .compactMap { $0 }
      .joined(separator: "-")
      let suffix = compressionSuffix.isEmpty ? "" : "-\(compressionSuffix)"
      return "coreai-export-v6\(suffix)\(compileSuffix)"
    }
  }

  struct CoreAIModelExportError: Hashable, Error {
    let message: String
  }
#endif

#if CoreML && canImport(CoreML)
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  func exportNeedleCoreML(
    outputDirectoryName: String = "coreml-export",
    arguments: [String] = [],
    compilePlatforms: [String] = []
  ) async throws -> URL {
    let compileArguments = compilePlatforms.flatMap { ["--compile-platform", $0] }
    return try await NeedleTestModelExport.export(
      backend: "CoreML",
      outputDirectoryName: outputDirectoryName,
      arguments: arguments + compileArguments,
      expectedFilesExist: {
        SelfCoreMLExport.filesExist(in: $0, compilePlatforms: compilePlatforms)
      },
      errorMessage: { CoreMLModelExportError(message: $0) }
    )
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  func makeNeedleCoreMLModelEngine(
    quantizerPreset: String? = nil,
    palettizerBits: Int? = nil,
    compilePlatforms: [String] = [],
    computeUnits: MLComputeUnits = .cpuAndGPU
  ) async throws -> NeedleCoreMLModelEngine {
    var arguments = quantizerPreset.map { ["--quantizer-preset", $0] } ?? []
    arguments += palettizerBits.map { ["--palettizer-n-bits", String($0)] } ?? []
    let directory = try await exportNeedleCoreML(
      outputDirectoryName: SelfCoreMLExport.outputDirectoryName(
        quantizerPreset: quantizerPreset,
        palettizerBits: palettizerBits,
        compilePlatforms: compilePlatforms
      ),
      arguments: arguments,
      compilePlatforms: compilePlatforms
    )
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    return try await NeedleCoreMLModelEngine(
      modelDirectoryURL: directory,
      modelConfiguration: configuration
    )
  }

  private enum SelfCoreMLExport {
    static func filesExist(in directory: URL, compilePlatforms: [String]) -> Bool {
      let fileManager = FileManager.default
      let hasModels: Bool
      if compilePlatforms.isEmpty {
        hasModels =
          fileManager.fileExists(atPath: directory.appending(path: "encoder.mlpackage").path())
          && fileManager.fileExists(atPath: directory.appending(path: "decoder.mlpackage").path())
      } else {
        hasModels = compilePlatforms.allSatisfy { platform in
          let compiledDirectory = directory.appending(path: "compiled").appending(path: platform)
          return fileManager.fileExists(
            atPath: compiledDirectory.appending(path: "encoder.mlmodelc").path()
          )
            && fileManager.fileExists(
              atPath: compiledDirectory.appending(path: "decoder.mlmodelc").path()
            )
        }
      }
      return NeedleTestModelExport.hasTokenizer(in: directory)
        && hasModels
        && fileManager.fileExists(atPath: directory.appending(path: "configuration.json").path())
    }

    static func outputDirectoryName(
      quantizerPreset: String?,
      palettizerBits: Int?,
      compilePlatforms: [String]
    ) -> String {
      let compileSuffix =
        compilePlatforms.isEmpty
        ? ""
        : "-aot-" + compilePlatforms.joined(separator: "-")
      let compressionSuffix = [
        quantizerPreset,
        palettizerBits.map { "palette\($0)" }
      ]
      .compactMap { $0 }
      .joined(separator: "-")
      let suffix = compressionSuffix.isEmpty ? "" : "-\(compressionSuffix)"
      return "coreml-export-v18\(suffix)\(compileSuffix)"
    }
  }

  struct CoreMLModelExportError: Hashable, Error {
    let message: String
  }
#endif

#if ONNX && canImport(COnnxRuntime)
  func exportNeedleONNX(
    outputDirectoryName: String = "onnx-export-v2-float32",
    quantization: String? = nil
  ) async throws -> URL {
    var arguments = ["--onnx-dtype", "float32"]
    arguments += quantization.map { ["--onnx-quantization", $0] } ?? []
    return try await NeedleTestModelExport.export(
      backend: "onnx",
      outputDirectoryName: outputDirectoryName,
      arguments: arguments,
      expectedFilesExist: SelfONNXExport.filesExist(in:),
      errorMessage: { ONNXModelExportError(message: $0) }
    )
  }

  func makeNeedleONNXModelEngine(
    quantization: String? = nil
  ) async throws -> NeedleCONNXModelEngine {
    let quantizationSuffix = quantization.map { "-\($0)" } ?? ""
    let directory = try await exportNeedleONNX(
      outputDirectoryName: "onnx-export-v2-float32\(quantizationSuffix)",
      quantization: quantization
    )
    return try await NeedleCONNXModelEngine(from: directory)
  }

  private enum SelfONNXExport {
    static func filesExist(in directory: URL) -> Bool {
      let fileManager = FileManager.default
      return NeedleTestModelExport.hasTokenizer(in: directory)
        && fileManager.fileExists(atPath: directory.appending(path: "configuration.json").path())
        && fileManager.fileExists(atPath: directory.appending(path: "encoder.onnx").path())
        && fileManager.fileExists(atPath: directory.appending(path: "decoder.onnx").path())
    }
  }

  struct ONNXModelExportError: Hashable, Error {
    let message: String
  }
#endif
