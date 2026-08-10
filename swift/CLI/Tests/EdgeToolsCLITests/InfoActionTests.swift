import CustomDump
import EdgeToolsCLI
import Foundation
import Testing

@Suite
struct `InfoAction tests` {
  @Test
  func `Reports The Detected Model And Engines`() async throws {
    let report = try await inspectModel(
      context: .stub(
        directory: URL(fileURLWithPath: "/models/needle"),
        engines: [.mlx, .onnx],
        files: ["config.json", "encoder.onnx", "model.safetensors"]
      ),
      source: .test()
    )

    expectNoDifference(report.model, "Needle")
    expectNoDifference(report.engines, ["mlx", "onnx"])
    expectNoDifference(report.defaultEngine, "mlx")
    expectNoDifference(report.unavailableEngines, ["coreml", "coreai"])
  }

  @Test
  func `Marks Experimental Engines And Leaves No Default`() async throws {
    let report = try await inspectModel(
      context: .stub(engines: [.coreai], files: ["configuration.json", "encoder.aimodel"]),
      source: .test()
    )

    expectNoDifference(report.experimentalEngines, ["coreai"])
    expectNoDifference(report.defaultEngine, nil)
    expectNoDifference(report.displayText().contains("coreai (experimental)"), true)
  }

  @Test
  func `Propagates Detection Failures`() async {
    var context = EdgeContext.stub()
    context.detectModel = { _ in throw EdgeCLIError("no config") }
    await #expect(throws: EdgeCLIError("no config")) {
      try await inspectModel(context: context, source: .test())
    }
  }

  @Test
  func `Resolves The Source Described By The Model Options`() async throws {
    let sources = LockedBox<ModelSource?>(nil)
    _ = try await inspectModel(
      context: .stub(onResolve: { sources.value = $0 }),
      source: .test(repo: "org/model")
    )

    expectNoDifference(
      try #require(sources.value).location,
      .huggingFace(repo: "org/model", revision: "main")
    )
  }
}
