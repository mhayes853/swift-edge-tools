#if MLX && canImport(MLX)
  import MLX
  import MLXLMCommon

  // MARK: - Logit Processor

  public final class NeedleApplyBitmaskProcessorMLX: LogitProcessor {
    public private(set) var matcher: XGrammarMatcher
    
    public init(matcher: XGrammarMatcher) {
      self.matcher = matcher
    }
    
    public func prompt(_ prompt: MLXArray) {
    }
    
    public func didSample(token: MLXArray) {
      self.matcher.accept(tokenId: token.item(NeedleToken.ID.self))
    }
    
    public func process(logits: MLXArray) -> MLXArray {
      applyBitmaskMLX(logits: logits, mask: self.matcher.bitmask())
    }
  }

  // MARK: - Apply Bitmask

  public func applyBitmaskMLX(logits: MLXArray, mask: GrammarBitmask) -> MLXArray {
    let vocabSize = logits.dim(1)
    let table = MLXArray(Needle.bitmaskTable, [256, 8]).asType(logits.dtype)
    let mask = mask.storage.withUnsafeBytes { MLXArray($0)[.newAxis, 0...] }
    return logits[0..., 0..<vocabSize] + table[mask].flattened(start: -2)[0..., 0..<vocabSize]
  }
#endif
