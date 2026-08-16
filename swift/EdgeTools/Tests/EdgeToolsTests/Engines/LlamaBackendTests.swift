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
      expectNoDifference(LlamaApi.vendored.model.hasProbe != nil, true)
      expectNoDifference(LlamaApi.vendored.context.probeConfidence != nil, true)
    }
  }
#endif
