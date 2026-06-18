#if SwiftNeedleMLX
  import MLX
  import MLXLMCommon

  // MARK: - NeedleXGrammarLogitsProcessor

  public struct NeedleXGrammarLogitsProcessor: LogitProcessor {
    private var matcher: NeedleXGrammarEngine.Matcher

    public init(matcher: NeedleXGrammarEngine.Matcher) {
      self.matcher = matcher
    }

    public mutating func prompt(_ prompt: MLXArray) {
    }

    public mutating func didSample(token: MLXArray) {
      self.matcher.accept(tokenId: token.item(Int.self))
    }

    public func process(logits: MLXArray) -> MLXArray {
      applyBitmaskMLX(logits: logits, mask: self.matcher.bitmask())
    }
  }

  // MARK: - Bitmask

  public func applyBitmaskMLX(logits: MLXArray, mask: NeedleGrammarBitmask) -> MLXArray {
    let vocabSize = logits.dim(1)
    let table = MLXArray(bitmaskTable, [256, 8]).asType(logits.dtype)
    let mask = mask.storage.withUnsafeBytes { MLXArray($0)[.newAxis, 0...] }
    return logits[0..., 0..<vocabSize] + table[mask].flattened(start: -2)[0..., 0..<vocabSize]
  }

  // MARK: - Table

  private let bitmaskTable = {
    var values = [Float]()
    values.reserveCapacity(256 * 8)
    for byte in 0..<256 {
      for bit in 0..<8 {
        values.append(((byte >> bit) & 1) != 0 ? 0 : -.infinity)
      }
    }
    return values
  }()
#endif
