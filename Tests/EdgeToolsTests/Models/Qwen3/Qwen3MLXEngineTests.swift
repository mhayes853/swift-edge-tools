#if MLX && XGrammar && canImport(MLX)
  import CustomDump
  import EdgeTools
  import SnapshotTesting
  import Testing

  @Suite(.serialized, .enabledIfXcode())
  struct `Qwen3 MLX engine tests` {
    @Test
    func `Completes Tool Turn Snapshot`() async throws {
      let engine = try await Qwen3MLXEngine(from: downloadQwen3())
      let transcript = try await completeWeatherTurn(using: engine)

      assertSnapshot(of: transcript, as: .dump)
    }
  }
#endif
