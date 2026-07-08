#if swift(>=6.4) && CoreML && Sentencepiece && canImport(CoreML)
  import CoreML
  import Foundation
  import Needle

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  func exportNeedleCoreML(
    outputDirectoryName: String = "coreml-export",
    arguments: [String] = []
  ) async throws -> URL {
    let outputDirectory = URL.swiftNeedleTestsDirectory.appending(path: outputDirectoryName)
    if SelfCoreMLExport.filesExist(in: outputDirectory) {
      return outputDirectory
    }

    print("=== Exporting Test CoreML Model ===")
    try FileManager.default.createDirectory(
      at: outputDirectory.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )

    let pythonDirectory = SelfCoreMLExport.pythonDirectory()
    let pythonExecutable = SelfCoreMLExport.pythonExecutable(in: pythonDirectory)

    let process = Process()
    process.executableURL = pythonExecutable
    process.currentDirectoryURL = pythonDirectory
    process.arguments = [
      "cli.py",
      "--backend",
      "CoreML",
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
      throw CoreMLModelExportError(message: output)
    }
    guard SelfCoreMLExport.filesExist(in: outputDirectory) else {
      throw CoreMLModelExportError(message: "CoreML export did not produce the expected files.")
    }
    print("=== Exported Test CoreML Model Successfully ===")
    return outputDirectory
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
    return try await NeedleCoreMLEngine(modelDirectoryURL: directory)
  }

  private enum SelfCoreMLExport {
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
      return hasTokenizer
        && fileManager.fileExists(atPath: directory.appending(path: "encoder.mlpackage").path())
        && fileManager.fileExists(atPath: directory.appending(path: "decoder.mlpackage").path())
        && fileManager.fileExists(atPath: directory.appending(path: "configuration.json").path())
    }

    static func outputDirectoryName(quantizerPreset: String?) -> String {
      if let quantizerPreset {
        return "coreml-export-v3-\(quantizerPreset)"
      }
      return "coreml-export-v3"
    }

    static func pythonExecutable(in pythonDirectory: URL) -> URL {
      let venvExecutable = pythonDirectory.appending(path: ".venv/bin/python")
      if FileManager.default.isExecutableFile(atPath: venvExecutable.path()) {
        return venvExecutable
      }
      return URL(fileURLWithPath: "/usr/bin/python3")
    }
  }

  struct CoreMLModelExportError: Hashable, Error {
    let message: String
  }
#endif
