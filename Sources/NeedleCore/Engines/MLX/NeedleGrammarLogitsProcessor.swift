#if SwiftNeedleMLX
  import MLX
  import MLXLMCommon

  // MARK: - NeedleGrammarLogitsProcessor

  public struct NeedleGrammarLogitsProcessor<Matcher: NeedleGrammarMatcher>: LogitProcessor {
    private var matcher: Matcher
    private let bitmaskTable: MLXArray

    public init(matcher: Matcher) {
      self.matcher = matcher
      self.bitmaskTable = NeedleCore.bitmaskTable()
    }

    public mutating func prompt(_ prompt: MLXArray) {
    }

    public mutating func didSample(token: MLXArray) {
      self.matcher.accept(tokenId: token.item(Int.self))
    }

    public func process(logits: MLXArray) -> MLXArray {
      let vocabSize = logits.dim(1)
      let table = self.bitmaskTable.asType(logits.dtype)
      let mask = self.matcher.bitmask().storage.withUnsafeBytes { MLXArray($0)[.newAxis, 0...] }
      return logits[0..., 0..<vocabSize] + table[mask].flattened(start: -2)[0..., 0..<vocabSize]
    }
  }

  // MARK: - Table

  private func bitmaskTable() -> MLXArray {
    var values = [Float]()
    values.reserveCapacity(256 * 8)
    for byte in 0..<256 {
      for bit in 0..<8 {
        values.append(((byte >> bit) & 1) != 0 ? 0 : -.infinity)
      }
    }
    return MLXArray(values, [256, 8])
  }
#endif
