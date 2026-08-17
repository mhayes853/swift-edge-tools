#if Llama && canImport(CLlama)
  import CustomDump
  import EdgeTools
  import Testing

  @Suite
  struct `LlamaSystemInfo tests` {
    @Test
    func `Vendored Build Links And Reports Backend Features`() {
      expectNoDifference(llamaSystemInfo().isEmpty, false)
    }
  }
#endif
