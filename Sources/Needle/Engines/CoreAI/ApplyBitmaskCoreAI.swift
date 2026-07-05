#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI

  @discardableResult
  @available(anyAppleOS 27.0, *)
  public func applyBitmaskCoreAI(logits: inout NDArray, mask: NeedleGrammarBitmask) -> NDArray {
    var logitsView = logits.mutableView(as: Float.self)
    mask.storage.withUnsafeBytes { ptr in
      let buffer = ptr.bindMemory(to: UInt8.self)
      for i in 0..<logitsView.shape[0] {
        for j in 0..<logitsView.shape[1] {
          let tableIndex = Int(buffer[j >> 3]) * 8 + (j & 7)
          logitsView[scalarAt: [i, j]] += Needle.bitmaskTable[tableIndex]
        }
      }
    }
    return logits
  }
#endif
