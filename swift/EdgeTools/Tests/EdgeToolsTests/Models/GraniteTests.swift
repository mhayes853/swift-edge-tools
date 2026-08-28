import EdgeTools
import Testing

#if !os(WASI)
  import SnapshotTesting
#endif

extension `Model tests` {
  @Suite
  struct `Granite tests` {
    #if MLX && canImport(MLX) && !os(WASI)
      @Suite(.enabledIfMLXTests())
      struct `GraniteMoeHybridMLXModelEngine tests` {
        @Test
        func `Completes Tool Turn Snapshot`() async throws {
          let engine = try await GraniteMoeHybridMLXModelEngine(from: downloadGraniteMoeHybrid())
          let transcript = try await completeWeatherTurn(using: engine)

          withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
        }
      }
    #endif

    #if HuggingFaceTokenizers && Llama && canImport(CLlama) && !os(WASI)
      @Suite
      struct `GraniteLlamaModelEngine tests` {
        @Test
        func `Llama Completes Tool Turn Snapshot`() async throws {
          let engine = try GraniteLlamaModelEngine(
            modelPath: (try await downloadGGUFModel(id: .graniteMoeHybrid)).path()
          )
          let transcript = try await completeWeatherTurn(using: engine)

          withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
        }
      }
    #endif
  }
}
