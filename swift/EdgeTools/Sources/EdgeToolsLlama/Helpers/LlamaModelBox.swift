#if canImport(CLlama)
  package final class LlamaModelBox: Sendable {
    package let model: LlamaModel

    package init(model: consuming LlamaModel) {
      self.model = consume model
    }
  }
#endif
