#if MLX && canImport(MLX)
  import MLX

  public func applyBitmaskMLX(logits: MLXArray, mask: NeedleGrammarBitmask) -> MLXArray {
    let vocabSize = logits.dim(1)
    let table = MLXArray(Needle.bitmaskTable, [256, 8]).asType(logits.dtype)
    let mask = mask.storage.withUnsafeBytes { MLXArray($0)[.newAxis, 0...] }
    return logits[0..., 0..<vocabSize] + table[mask].flattened(start: -2)[0..., 0..<vocabSize]
  }
#endif
