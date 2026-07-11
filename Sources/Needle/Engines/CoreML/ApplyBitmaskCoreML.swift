#if CoreML && canImport(CoreML)
  import CoreML

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  public func applyBitmaskCoreML(logits: MLTensor, mask: GrammarBitmask) -> MLTensor {
    let maskTensor = mask.storage.withUnsafeBytes { bytes in
      let scalars = bytes.bindMemory(to: UInt8.self)
        .flatMap { Needle.bitmaskTable[(Int($0) * 8)..<(Int($0) * 8 + 8)] }
      return MLTensor(shape: [1, logits.shape[1]], scalars: Array(scalars.prefix(logits.shape[1])))
    }
    return logits + maskTensor
  }
#endif
