import EdgeTools
import Testing

#if !os(WASI)
  import SnapshotTesting
#endif

@Suite
struct `Qwen3 tests` {
  #if MLX && XGrammar && canImport(MLX) && !os(WASI)
    @Suite(.serialized, .enabledIfMLXTests())
    struct `Qwen3MLXModelEngine tests` {
      @Test
      func `Completes Tool Turn Snapshot`() async throws {
        let engine = try await Qwen3MLXModelEngine(from: downloadQwen3())
        let transcript = try await completeWeatherTurn(using: engine)

        withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
      }

      @Test
      func `Generates Reasoning Snapshot`() async throws {
        let engine = try await Qwen3MLXModelEngine(from: downloadQwen3())
        let generation = try await generateReasoning(using: engine)

        withKnownIssue { assertSnapshot(of: generation, as: .dump, record: .all) }
      }
    }
  #endif
}
