#if swift(>=6.4) && CoreAI && Sentencepiece && canImport(CoreAI)
  import Foundation
  import Needle

  @available(anyAppleOS 27.0, *)
  func exportNeedleCoreAI(
    outputDirectoryName: String = "coreai-export",
    arguments: [String] = []
  ) async throws -> URL {
    let outputDirectory = URL.swiftNeedleTestsDirectory.appending(path: outputDirectoryName)
    if SelfCoreAIExport.filesExist(in: outputDirectory) {
      return outputDirectory
    }

    print("=== Exporting Test CoreAI Model ===")
    try FileManager.default.createDirectory(
      at: outputDirectory.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )

    let pythonDirectory = SelfCoreAIExport.pythonDirectory()
    let pythonExecutable = SelfCoreAIExport.pythonExecutable(in: pythonDirectory)

    let process = Process()
    process.executableURL = pythonExecutable
    process.currentDirectoryURL = pythonDirectory
    process.arguments = [
      "cli.py",
      "--output",
      outputDirectory.path()
    ] + arguments

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
      throw CoreAIModelExportError(message: output)
    }
    guard SelfCoreAIExport.filesExist(in: outputDirectory) else {
      throw CoreAIModelExportError(message: "CoreAI export did not produce the expected files.")
    }
    print("=== Exported Test CoreAI Model Successfully ===")
    return outputDirectory
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
    static func pythonDirectory() -> URL {
      let packageDirectory =
        URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
      return packageDirectory.appending(path: "python")
    }

    static func filesExist(in directory: URL) -> Bool {
      let fileManager = FileManager.default
      let hasTokenizer = ["tokenizer.model", "tokenizer.json"]
        .contains { fileManager.fileExists(atPath: directory.appending(path: $0).path()) }
      guard hasTokenizer else { return false }

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

    static func pythonExecutable(in pythonDirectory: URL) -> URL {
      let venvExecutable = pythonDirectory.appending(path: ".venv/bin/python")
      if FileManager.default.isExecutableFile(atPath: venvExecutable.path()) {
        return venvExecutable
      }
      return URL(fileURLWithPath: "/usr/bin/python3")
    }
  }

  struct CoreAIModelExportError: Hashable, Error {
    let message: String
  }
#endif
