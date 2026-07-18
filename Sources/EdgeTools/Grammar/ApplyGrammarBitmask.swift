// MARK: - MLX

#if MLX && canImport(MLX)
  import MLX
  import MLXLMCommon

  public final class EdgeToolsApplyBitmaskProcessorMLX: LogitProcessor {
    public private(set) var matcher: XGRMatcher

    public init(matcher: consuming XGRMatcher) {
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

  public func applyBitmaskMLX(logits: MLXArray, mask: GrammarBitmask) -> MLXArray {
    let vocabularySize = logits.dim(1)
    let table = MLXArray(_bitmaskTable, [256, 8]).asType(logits.dtype)
    let mask = mask.storage.withUnsafeBytes { MLXArray($0)[.newAxis, 0...] }
    return logits[0..., 0..<vocabularySize]
      + table[mask].flattened(start: -2)[0..., 0..<vocabularySize]
  }
#endif

// MARK: - Core ML

#if CoreML && canImport(CoreML)
  import CoreML

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  public func applyBitmaskCoreML(logits: MLTensor, mask: GrammarBitmask) -> MLTensor {
    let maskTensor = mask.storage.withUnsafeBytes { bytes in
      let scalars = bytes.bindMemory(to: UInt8.self)
        .flatMap { _bitmaskTable[(Int($0) * 8)..<(Int($0) * 8 + 8)] }
      return MLTensor(shape: [1, logits.shape[1]], scalars: Array(scalars.prefix(logits.shape[1])))
    }
    return logits + maskTensor
  }
#endif

// MARK: - Core AI

#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI

  @discardableResult
  @available(anyAppleOS 27.0, *)
  public func applyBitmaskCoreAI(logits: inout NDArray, mask: GrammarBitmask) -> NDArray {
    var logitsView = logits.mutableView(as: Float.self)
    let rowCount = logitsView.shape[0]
    let vocabularySize = logitsView.shape[1]
    mask.storage.withUnsafeBytes { bytes in
      let maskBytes = bytes.bindMemory(to: UInt8.self)
      if logitsView.isContiguous {
        logitsView.withUnsafeMutablePointer { logitsPointer, _, _ in
          for rowIndex in 0..<rowCount {
            _applyBitmaskSIMDRow(
              logits: logitsPointer.advanced(by: rowIndex * vocabularySize),
              vocabularySize: vocabularySize,
              maskBytes: maskBytes
            )
          }
        }
      } else {
        for rowIndex in 0..<rowCount {
          for columnIndex in 0..<vocabularySize {
            let value = _bitmaskValue(maskBytes: maskBytes, index: columnIndex)
            logitsView[scalarAt: [rowIndex, columnIndex]] += value
          }
        }
      }
    }
    return logits
  }

  @inline(always)
  func _applyBitmaskSIMDRow(
    logits: UnsafeMutablePointer<Float>,
    vocabularySize: Int,
    maskBytes: UnsafeBufferPointer<UInt8>
  ) {
    var index = 0
    while index + SIMD8<Float>.scalarCount <= vocabularySize {
      let pointer = UnsafeMutableRawPointer(logits.advanced(by: index))
      let values = UnsafeRawPointer(pointer).loadUnaligned(as: SIMD8<Float>.self)
      let mask = _bitmaskSIMDTable[Int(maskBytes[index >> 3])]
      pointer.storeBytes(of: values + mask, as: SIMD8<Float>.self)
      index += SIMD8<Float>.scalarCount
    }
    while index < vocabularySize {
      logits[index] += _bitmaskValue(maskBytes: maskBytes, index: index)
      index += 1
    }
  }

  @inline(always)
  func _bitmaskValue(maskBytes: UnsafeBufferPointer<UInt8>, index: Int) -> Float {
    let tableIndex = Int(maskBytes[index >> 3]) * 8 + (index & 7)
    return _bitmaskTable[tableIndex]
  }
#endif

// MARK: - Lookup Tables

@usableFromInline
let _bitmaskTable = (0..<256)
  .flatMap { byte in
    (0..<8).map { bit in ((byte >> bit) & 1) != 0 ? 0 : -Float.infinity }
  }

@usableFromInline
let _bitmaskSIMDTable = _bitmaskTable.withUnsafeBufferPointer { table in
  (0..<256)
    .map { byte in
      UnsafeRawPointer(table.baseAddress!.advanced(by: byte * 8))
        .loadUnaligned(as: SIMD8<Float>.self)
    }
}
