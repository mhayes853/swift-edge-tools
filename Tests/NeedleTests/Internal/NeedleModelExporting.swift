import Foundation
import Needle

private enum NeedleTestModelExport {
  static func export(
    backend: String?,
    outputDirectoryName: String,
    arguments: [String],
    expectedFilesExist: (URL) -> Bool,
    errorMessage: (String) -> any Error
  ) async throws -> URL {
    let outputDirectory = URL.swiftNeedleTestsDirectory.appending(path: outputDirectoryName)
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
    process.waitUntilExit()

    guard process.terminationStatus == 0 else {
      let output = String(
        decoding: outputPipe.fileHandleForReading.readDataToEndOfFile(),
        as: UTF8.self
      )
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

#if swift(>=6.4) && CoreAI && Sentencepiece && canImport(CoreAI)
  import CoreAI

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
  func makeNeedleCoreAIEngine(
    quantizerPreset: String? = nil,
    compilePlatforms: [String] = []
  ) async throws -> NeedleCoreAIEngine {
    var arguments = quantizerPreset.map { ["--quantizer-preset", $0] } ?? []
    arguments += compilePlatforms.flatMap { ["--compile-platform", $0] }
    let directory = try await exportNeedleCoreAI(
      outputDirectoryName: SelfCoreAIExport.outputDirectoryName(
        quantizerPreset: quantizerPreset,
        compilePlatforms: compilePlatforms
      ),
      arguments: arguments
    )
    return try await NeedleCoreAIEngine(modelDirectoryURL: directory)
  }

  private enum SelfCoreAIExport {
    static func filesExist(in directory: URL) -> Bool {
      guard NeedleTestModelExport.hasTokenizer(in: directory) else { return false }

      let fileManager = FileManager.default
      let contents = (try? fileManager.contentsOfDirectory(atPath: directory.path())) ?? []
      let hasEncoderModel = contents.contains("encoder.aimodel")
        || contents.contains { $0.hasPrefix("encoder.") && $0.hasSuffix(".aimodelc") }
      let hasDecoderModel = contents.contains("decoder.aimodel")
        || contents.contains { $0.hasPrefix("decoder.") && $0.hasSuffix(".aimodelc") }
      return hasEncoderModel
        && hasDecoderModel
        && fileManager.fileExists(atPath: directory.appending(path: "configuration.json").path())
    }

    static func outputDirectoryName(
      quantizerPreset: String?,
      compilePlatforms: [String]
    ) -> String {
      let compileSuffix = compilePlatforms.isEmpty
        ? ""
        : "-aot-" + compilePlatforms.joined(separator: "-")
      if let quantizerPreset {
        return "coreai-export-\(quantizerPreset)\(compileSuffix)"
      }
      return "coreai-export\(compileSuffix)"
    }
  }

  struct CoreAIModelExportError: Hashable, Error {
    let message: String
  }
#endif

#if swift(>=6.4) && CoreML && Sentencepiece && canImport(CoreML)
  import CoreML

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  func exportNeedleCoreML(
    outputDirectoryName: String = "coreml-export",
    arguments: [String] = []
  ) async throws -> URL {
    try await NeedleTestModelExport.export(
      backend: "CoreML",
      outputDirectoryName: outputDirectoryName,
      arguments: arguments,
      expectedFilesExist: SelfCoreMLExport.filesExist(in:),
      errorMessage: { CoreMLModelExportError(message: $0) }
    )
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  func makeNeedleCoreMLEngine(
    quantizerPreset: String? = nil
  ) async throws -> NeedleCoreMLEngine {
    let arguments = quantizerPreset.map { ["--quantizer-preset", $0] } ?? []
    let directory = try await exportNeedleCoreML(
      outputDirectoryName: SelfCoreMLExport.outputDirectoryName(quantizerPreset: quantizerPreset),
      arguments: arguments
    )
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuAndGPU
    return try await NeedleCoreMLEngine(
      modelDirectoryURL: directory,
      modelConfiguration: configuration
    )
  }

  private enum SelfCoreMLExport {
    static func filesExist(in directory: URL) -> Bool {
      let fileManager = FileManager.default
      return NeedleTestModelExport.hasTokenizer(in: directory)
        && fileManager.fileExists(atPath: directory.appending(path: "encoder.mlpackage").path())
        && fileManager.fileExists(atPath: directory.appending(path: "decoder.mlpackage").path())
        && fileManager.fileExists(atPath: directory.appending(path: "configuration.json").path())
    }

    static func outputDirectoryName(quantizerPreset: String?) -> String {
      if let quantizerPreset {
        return "coreml-export-v9-\(quantizerPreset)"
      }
      return "coreml-export-v9"
    }
  }

  struct CoreMLModelExportError: Hashable, Error {
    let message: String
  }
#endif
