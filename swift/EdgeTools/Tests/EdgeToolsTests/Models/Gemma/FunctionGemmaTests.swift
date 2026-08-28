import CustomDump
import EdgeTools
import Testing

#if !os(WASI)
  import SnapshotTesting
#endif

@Suite(.serialized)
struct `Model tests` {}

extension `Model tests` {
  @Suite
  struct `FunctionGemma tests` {
    #if MLX && canImport(MLX) && !os(WASI)
      @Suite(.enabledIfMLXTests())
      struct `FunctionGemmaMLXModelEngine tests` {
        @Test
        func `Completes Tool Turn Snapshot`() async throws {
          let engine = try await FunctionGemmaMLXModelEngine(from: downloadFunctionGemma())
          let transcript = try await completeWeatherTurn(using: engine)

          withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
        }
      }
    #endif

    #if HuggingFaceTokenizers && Llama && canImport(CLlama) && !os(WASI)
      @Suite
      struct `FunctionGemmaLlamaModelEngine tests` {
        @Test
        func `Llama Completes Tool Turn Snapshot`() async throws {
          let engine = try FunctionGemmaLlamaModelEngine(
            modelPath: (try await downloadGGUFModel(id: .functionGemma)).path(),
            modelParameters: llamaTestModelParameters()
          )
          let transcript = try await completeWeatherTurn(using: engine)

          withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
        }
      }
    #endif
  }
}
