import CustomDump
import EdgeTools
import SnapshotTesting
import Testing

@Suite
struct Qwen3P5ConsolidatedTests {
  #if MLX && XGrammar && canImport(MLX)
    @Suite(.serialized, .enabledIfXcode())
    struct `Qwen3P5 MLX engine tests` {
      @Test
      func `Completes Tool Turn Snapshot`() async throws {
        let engine = try await Qwen3P5MLXEngine(from: downloadQwen3P5())
        let transcript = try await completeWeatherTurn(using: engine)

        assertSnapshot(of: transcript, as: .dump)
      }
    }
  #endif
}
