import CustomDump
import EdgeTools
import Testing

#if !os(WASI)
  import SnapshotTesting
#endif

@Suite
struct `FunctionGemma tests` {
  #if MLX && XGrammar && canImport(MLX) && !os(WASI)
    @Suite(.serialized, .enabledIfXcode())
    struct `FunctionGemma MLX engine tests` {
      @Test
      func `Completes Tool Turn Snapshot`() async throws {
        let engine = try await FunctionGemmaMLXEngine(from: downloadFunctionGemma())
        let transcript = try await completeWeatherTurn(using: engine)

        assertSnapshot(of: transcript, as: .dump)
      }
    }
  #endif
}
