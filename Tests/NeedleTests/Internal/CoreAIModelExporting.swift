#if swift(>=6.4) && CoreAI && Sentencepiece && canImport(CoreAI)
  import Foundation
  import Needle

  @available(anyAppleOS 27.0, *)
  func exportNeedleCoreAI() async throws -> URL {
    let outputDirectory = URL.swiftNeedleTestsDirectory.appending(path: "coreai-export")
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
    ]

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
  func makeNeedleCoreAIEngine() async throws -> NeedleCoreAIEngine {
    let directory = try await exportNeedleCoreAI()
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
      return [
        "encoder.aimodel",
        "decoder.aimodel",
        "configuration.json",
        "tokenizer.model"
      ]
      .allSatisfy { fileManager.fileExists(atPath: directory.appending(path: $0).path()) }
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
