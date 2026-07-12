#if MLX && canImport(MLX)
  import MLX
  import MLXLMCommon

  // MARK: - MLX Logit Processor

  public final class EdgeToolsApplyBitmaskProcessorMLX: LogitProcessor {
    public private(set) var matcher: XGrammarMatcher

    public init(matcher: consuming XGrammarMatcher) {
      self.matcher = consume matcher
    }

    public func prompt(_ prompt: MLXArray) {}

    public func didSample(token: MLXArray) {
      self.matcher.accept(tokenId: token.item(EdgeToolsToken.ID.self))
    }

    public func process(logits: MLXArray) -> MLXArray {
      applyBitmaskMLX(logits: logits, mask: self.matcher.bitmask())
    }
  }

  // MARK: - MLX

  public func applyBitmaskMLX(logits: MLXArray, mask: GrammarBitmask) -> MLXArray {
    let vocabularySize = logits.dim(1)
    let table = MLXArray(EdgeTools.bitmaskTable, [256, 8]).asType(logits.dtype)
    let mask = mask.storage.withUnsafeBytes { MLXArray($0)[.newAxis, 0...] }
    return logits[0..., 0..<vocabularySize]
      + table[mask].flattened(start: -2)[0..., 0..<vocabularySize]
  }
#endif

#if CoreML && canImport(CoreML)
  import CoreML

  // MARK: - Core ML

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  public func applyBitmaskCoreML(logits: MLTensor, mask: GrammarBitmask) -> MLTensor {
    let maskTensor = mask.storage.withUnsafeBytes { bytes in
      let scalars = bytes.bindMemory(to: UInt8.self)
        .flatMap { EdgeTools.bitmaskTable[(Int($0) * 8)..<(Int($0) * 8 + 8)] }
      return MLTensor(shape: [1, logits.shape[1]], scalars: Array(scalars.prefix(logits.shape[1])))
    }
    return logits + maskTensor
  }
#endif

#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI

  // MARK: - Core AI

  @discardableResult
  @available(anyAppleOS 27.0, *)
  public func applyBitmaskCoreAI(logits: inout NDArray, mask: GrammarBitmask) -> NDArray {
    var logitsView = logits.mutableView(as: Float.self)
    mask.storage.withUnsafeBytes { bytes in
      let buffer = bytes.bindMemory(to: UInt8.self)
      for rowIndex in 0..<logitsView.shape[0] {
        for columnIndex in 0..<logitsView.shape[1] {
          let tableIndex = Int(buffer[columnIndex >> 3]) * 8 + (columnIndex & 7)
          logitsView[scalarAt: [rowIndex, columnIndex]] += EdgeTools.bitmaskTable[tableIndex]
        }
      }
    }
    return logits
  }
#endif
