#if canImport(CLlama)
  import CustomDump
  import EdgeToolsLlama
  import Testing

  @Suite
  struct `LlamaSystemInfo tests` {
    @Test
    func `Vendored Build Links And Reports Backend Features`() {
      expectNoDifference(llamaSystemInfo().isEmpty, false)
    }
  }
#endif
