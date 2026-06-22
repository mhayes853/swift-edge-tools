import CustomDump
import Needle
import Observation
import Testing

@Suite
struct `NeedleSession tests` {
  @Test
  func `System Prompt Is Observable`() {
    let session = NeedleSession(engine: MockEngine())

    let didChange = Lock(false)
    withObservationTracking {
      _ = session.systemPrompt
    } onChange: {
      didChange.withLock { $0 = true }
    }

    session.systemPrompt = "new prompt"
    didChange.withLock { expectNoDifference($0, true) }
  }
}

// MARK: - MockEngine

private final class MockEngine: NeedleEngine, Sendable {
  struct GenerateParameters: NeedleEngineGenerateParameters {
    static let `default` = GenerateParameters()
  }

  let stopper = NeedleEngineStopper {}

  func generate(
    prompt: NeedlePrompt,
    parameters: GenerateParameters,
    onToken: (NeedleToken) -> Void
  ) throws -> NeedleEngineGeneration {
    NeedleEngineGeneration(
      prefillMetrics: NeedlePrefillMetrics(tokens: 0, duration: .zero),
      decodeMetrics: NeedleDecodeMetrics(
        tokens: 0,
        duration: .zero,
        durationToFirstToken: .zero
      ),
      wasStopped: false,
      tokens: []
    )
  }

  func reset() {}
}
