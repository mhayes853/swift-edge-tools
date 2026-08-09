import EdgeTools
import Testing

#if !os(WASI)
  import SnapshotTesting
#endif

@Suite
struct `Gemma4 tests` {
  #if MLX && XGrammar && canImport(MLX) && !os(WASI)
    @Suite(.serialized, .enabledIfXcode())
    struct `Gemma4MLXModelEngine tests` {
      @Test
      func `Completes Text Tool Turn Snapshot`() async throws {
        let engine = try await Gemma4MLXModelEngine(from: downloadGemma4E2B())
        let transcript = try await completeWeatherTurn(using: engine)

        withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
      }

      @Test
      func `Generates Reasoning Snapshot`() async throws {
        let engine = try await Gemma4MLXModelEngine(from: downloadGemma4E2B())
        let generation = try await generateReasoning(using: engine)

        withKnownIssue { assertSnapshot(of: generation, as: .dump, record: .all) }
      }

      @Test
      func `Describes Image Snapshot`() async throws {
        let engine = try await Gemma4MLXModelEngine(from: downloadGemma4E2B())
        let response = try await describeRedImage(using: engine)

        withKnownIssue { assertSnapshot(of: response, as: .lines, record: .all) }
      }

      @Test
      func `Completes Image Conditioned Tool Turn Snapshot`() async throws {
        let engine = try await Gemma4MLXModelEngine(from: downloadGemma4E2B())
        let result = try await completeImageColorTurn(using: engine)

        withKnownIssue { assertSnapshot(of: result, as: .dump, record: .all) }
      }
    }
  #endif
}
