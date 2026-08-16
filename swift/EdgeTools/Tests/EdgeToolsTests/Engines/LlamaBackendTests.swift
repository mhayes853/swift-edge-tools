#if Llama && canImport(CLlama)
  import CustomDump
  import EdgeTools
  import Testing

  @Suite
  struct `LlamaBackend tests` {
    @Test
    func `Vendored Build Links And Reports Backend Features`() {
      LlamaBackend.initialize()
      defer { LlamaBackend.shutdown() }

      expectNoDifference(LlamaBackend.systemInfo.isEmpty, false)
      expectNoDifference(LlamaAPI.vendored.modelHasProbe != nil, true)
      expectNoDifference(LlamaAPI.vendored.probeConfidence != nil, true)
    }
  }
#endif
