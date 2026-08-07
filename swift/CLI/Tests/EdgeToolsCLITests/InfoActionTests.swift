import CustomDump
import EdgeToolsCLI
import Foundation
import Testing

@Suite
struct `InfoAction tests` {
  @Test
  func `Reports The Detected Model And Engines`() async throws {
    let action = InfoAction(
      context: .stub(
        directory: URL(fileURLWithPath: "/models/needle"),
        engines: [.mlx, .onnx],
        files: ["config.json", "encoder.onnx", "model.safetensors"]
      ),
      source: .test(),
      quiet: true
    )
    let report = try await action()

    expectNoDifference(report.model, "Needle")
    expectNoDifference(report.engines, ["mlx", "onnx"])
    expectNoDifference(report.defaultEngine, "mlx")
    expectNoDifference(report.unavailableEngines, ["coreml", "coreai"])
  }

  @Test
  func `Marks Experimental Engines And Leaves No Default`() async throws {
    let action = InfoAction(
      context: .stub(engines: [.coreai], files: ["configuration.json", "encoder.aimodel"]),
      source: .test(),
      quiet: true
    )
    let report = try await action()

    expectNoDifference(report.experimentalEngines, ["coreai"])
    expectNoDifference(report.defaultEngine, nil)
    expectNoDifference(report.displayText().contains("coreai (experimental)"), true)
  }

  @Test
  func `Propagates Detection Failures`() async {
    var context = EdgeContext.stub()
    context.detectModel = { _ in throw EdgeCLIError("no config") }
    let action = InfoAction(context: context, source: .test(), quiet: true)

    await #expect(throws: EdgeCLIError("no config")) {
      try await action()
    }
  }

  @Test
  func `Resolves The Source Described By The Model Options`() async throws {
    let sources = LockedBox<ModelSource?>(nil)
    let action = InfoAction(
      context: .stub(onResolve: { sources.value = $0 }),
      source: .test(repo: "org/model"),
      quiet: true
    )
    _ = try await action()

    expectNoDifference(
      try #require(sources.value).location,
      .huggingFace(repo: "org/model", revision: "main")
    )
  }
}
