#if MLX && XGrammar && canImport(MLX)
  import CustomDump
  import EdgeTools
  import SnapshotTesting
  import Testing

  @Suite(.serialized, .enabledIfXcode())
  struct `LFM2 MLX engine tests` {
    @Test
    func `Completes Tool Turn Snapshot`() async throws {
      let engine = try await LFM2MLXEngine(from: downloadLFM2())
      let transcript = try await completeWeatherTurn(using: engine)

      assertSnapshot(of: transcript, as: .dump)
    }
  }
#endif
