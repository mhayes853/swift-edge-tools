import EdgeTools
import Foundation

#if ONNX && canImport(COnnxRuntime)
  func exportNeedleONNX(
    outputDirectoryName: String = "onnx-export-v3-float32",
    quantization: String? = nil
  ) async throws -> URL {
    let outputDirectory = URL.swiftEdgeToolsTestsDirectory.appending(path: outputDirectoryName)
    if SelfONNXExport.filesExist(in: outputDirectory) {
      return outputDirectory
    }

    print("=== Exporting Test ONNX Model ===")
    try FileManager.default.createDirectory(
      at: outputDirectory.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    let pythonDirectory = SelfONNXExport.pythonDirectory()
    let process = Process()
    process.executableURL = SelfONNXExport.pythonExecutable(in: pythonDirectory)
    process.currentDirectoryURL = pythonDirectory
    process.arguments =
      ["cli.py", "--output", outputDirectory.path(), "--dtype", "float32"]
      + (quantization.map { ["--quantization", $0] } ?? [])

    let outputPipe = Pipe()
    process.standardOutput = outputPipe
    process.standardError = outputPipe
    try process.run()
    let output = String(
      decoding: outputPipe.fileHandleForReading.readDataToEndOfFile(),
      as: UTF8.self
    )
    process.waitUntilExit()

    guard process.terminationStatus == 0 else {
      throw ONNXModelExportError(message: output)
    }
    guard SelfONNXExport.filesExist(in: outputDirectory) else {
      throw ONNXModelExportError(message: "ONNX export did not produce the expected files.")
    }
    print("=== Exported Test ONNX Model Successfully ===")
    return outputDirectory
  }

  func makeNeedleONNXModelEngine(
    quantization: String? = nil
  ) async throws -> NeedleCONNXModelEngine {
    let quantizationSuffix = quantization.map { "-\($0)" } ?? ""
    let directory = try await exportNeedleONNX(
      outputDirectoryName: "onnx-export-v3-float32\(quantizationSuffix)",
      quantization: quantization
    )
    return try await NeedleCONNXModelEngine(from: directory)
  }

  private enum SelfONNXExport {
    static func filesExist(in directory: URL) -> Bool {
      let fileManager = FileManager.default
      let hasTokenizer = ["tokenizer.model", "tokenizer.json"]
        .contains {
          fileManager.fileExists(atPath: directory.appending(path: $0).path())
        }
      return hasTokenizer
        && fileManager.fileExists(atPath: directory.appending(path: "configuration.json").path())
        && fileManager.fileExists(atPath: directory.appending(path: "encoder.onnx").path())
        && fileManager.fileExists(atPath: directory.appending(path: "decoder.onnx").path())
    }

    static func pythonDirectory() -> URL {
      let packageDirectory =
        URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
      return packageDirectory.appending(path: "python")
    }

    static func pythonExecutable(in pythonDirectory: URL) -> URL {
      let venvExecutable = pythonDirectory.appending(path: ".venv/bin/python")
      if FileManager.default.isExecutableFile(atPath: venvExecutable.path()) {
        return venvExecutable
      }
      return URL(fileURLWithPath: "/usr/bin/python3")
    }
  }

  struct ONNXModelExportError: Hashable, Error {
    let message: String
  }
#endif
