import CustomDump
import EdgeToolsCLI
import Foundation
import Testing

@Suite
struct `EngineRunner tests` {
  @Test
  func `Public Initializer Rejects Unsupported Model Engine Pairs`() async {
    let detection = ModelDetection(
      directory: URL(fileURLWithPath: "/models/qwen"),
      model: .qwen3,
      engines: [.onnx],
      files: ["model.onnx"]
    )

    await #expect(throws: EdgeCLIError.self) {
      try await EngineRunner(detection: detection, requestedEngine: .onnx)
    }
  }

  @Test
  func `Initializer Selects And Publishes The Validated Engine`() async throws {
    let detection = ModelDetection(
      directory: URL(fileURLWithPath: "/models/needle"),
      model: .needle,
      engines: [.mlx, .onnx],
      files: ["model.safetensors", "model.onnx"]
    )
    let runner = try await EngineRunner(
      detection: detection,
      requestedEngine: .onnx,
      hardwareUnit: .gpu,
      loader: { _, _, _ in .stub() }
    )

    expectNoDifference(runner.engine, .onnx)
  }

}
