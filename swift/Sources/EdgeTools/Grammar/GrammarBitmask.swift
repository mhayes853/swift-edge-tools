#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && canImport(MLX)
  import MLX
  import MLXLMCommon
#endif

#if CoreML && canImport(CoreML)
  import CoreML
#endif

#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI
#endif

// MARK: - GrammarBitmask

public struct GrammarBitmask: Hashable, Sendable {
  public var storage: [UInt8]

  @inlinable
  @inline(always)
  public init(storage: [UInt8]) {
    self.storage = storage
  }

  @inlinable
  @inline(always)
  public init(bitCount: Int, repeating value: Bool = false) {
    precondition(bitCount.isMultiple(of: 8), "Grammar bit count must be a multiple of 8.")
    self.storage = [UInt8](repeating: value ? .max : .zero, count: bitCount / 8)
  }
}

// MARK: - Collection

extension GrammarBitmask: MutableCollection {
  public typealias Index = Int
  public typealias Element = Bool

  public func index(after index: Int) -> Int {
    index + 1
  }
  public var startIndex: Int { self.storage.startIndex }
  public var endIndex: Int { self.storage.endIndex * 8 }

  @inlinable
  @inline(always)
  public subscript(position: Index) -> Bool {
    get { (self.storage[position / 8] & (1 << (position % 8))).boolValue }
    set {
      let index = position / 8
      let mask = UInt8(1) &<< UInt8(position % 8)
      self.storage[index] = (self.storage[index] & ~mask) | (newValue ? mask : 0)
    }
  }
}

extension GrammarBitmask: RandomAccessCollection {}

// MARK: - MLX

#if MLX && canImport(MLX)
  public final class MLXBitmaskProcessor<Matcher: EdgeToolsGrammarMatcher>: LogitProcessor {
    public private(set) var matcher: Matcher

    public init(matcher: Matcher) {
      self.matcher = matcher
    }

    public func prompt(_ prompt: MLXArray) {}

    public func didSample(token: MLXArray) {
      guard !self.matcher.isTerminated else { return }
      self.matcher.accept(tokenId: token.item(EdgeToolsToken.ID.self))
    }

    public func process(logits: MLXArray) -> MLXArray {
      guard !self.matcher.isTerminated else { return logits }
      return applyBitmaskMLX(logits: logits, mask: self.matcher.grammarBitmask())
    }
  }
#endif

#if MLX && canImport(MLX)
  public func applyBitmaskMLX(logits: MLXArray, mask: GrammarBitmask) -> MLXArray {
    let vocabularySize = logits.dim(1)
    validateBitmaskCoverage(mask: mask, vocabularySize: vocabularySize)
    let mask = mask.storage.withUnsafeBytes { MLXArray($0)[.newAxis, 0...] }
    return compiledApplyBitmaskMLX(logits, mask)
  }

  private let compiledApplyBitmaskMLX = compile { logits, mask in
    let vocabularySize = logits.dim(1)
    let table = MLXArray(bitmaskTable, [256, 8]).asType(logits.dtype)
    return logits[0..., 0..<vocabularySize]
      + table[mask].flattened(start: -2)[0..., 0..<vocabularySize]
  }
#endif

// MARK: - Core ML

#if CoreML && canImport(CoreML)
  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  public func applyBitmaskCoreML(logits: MLTensor, mask: GrammarBitmask) -> MLTensor {
    let vocabularySize = logits.shape[1]
    validateBitmaskCoverage(mask: mask, vocabularySize: vocabularySize)
    let maskTensor = mask.storage.withUnsafeBytes { bytes in
      let scalars = bytes.bindMemory(to: UInt8.self)
        .flatMap { bitmaskTable[(Int($0) * 8)..<(Int($0) * 8 + 8)] }
      return MLTensor(shape: [1, vocabularySize], scalars: Array(scalars.prefix(vocabularySize)))
    }
    return logits + maskTensor
  }
#endif

// MARK: - Core AI

#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  @discardableResult
  @available(anyAppleOS 27.0, *)
  public func applyBitmask(logits: inout NDArray, mask: GrammarBitmask) -> NDArray {
    var logitsView = logits.mutableView(as: Float.self)
    let rowCount = logitsView.shape[0]
    let vocabularySize = logitsView.shape[1]
    validateBitmaskCoverage(mask: mask, vocabularySize: vocabularySize)
    if logitsView.isContiguous {
      logitsView.withUnsafeMutablePointer { logitsPointer, _, _ in
        for rowIndex in 0..<rowCount {
          var logits = MutableSpan(
            _unsafeStart: logitsPointer.advanced(by: rowIndex * vocabularySize),
            count: vocabularySize
          )
          applyBitmask(logits: &logits, mask: mask)
        }
      }
    } else {
      mask.storage.withUnsafeBytes { bytes in
        let maskBytes = bytes.bindMemory(to: UInt8.self)
        for rowIndex in 0..<rowCount {
          for columnIndex in 0..<vocabularySize {
            let value = bitmaskValue(maskBytes: maskBytes, index: columnIndex)
            logitsView[scalarAt: [rowIndex, columnIndex]] += value
          }
        }
      }
    }
    return logits
  }

#endif

// MARK: - Contiguous Storage

public func applyBitmask(logits: inout MutableSpan<Float>, mask: GrammarBitmask) {
  validateBitmaskCoverage(mask: mask, vocabularySize: logits.count)
  mask.storage.withUnsafeBufferPointer { maskBytes in
    logits.withUnsafeMutableBufferPointer { buffer in
      guard let baseAddress = buffer.baseAddress else { return }
      applyBitmaskSIMDRow(
        logits: baseAddress,
        vocabularySize: buffer.count,
        maskBytes: maskBytes
      )
    }
  }
}

@inline(always)
private func applyBitmaskSIMDRow(
  logits: UnsafeMutablePointer<Float>,
  vocabularySize: Int,
  maskBytes: UnsafeBufferPointer<UInt8>
) {
  var index = 0
  while index + SIMD8<Float>.scalarCount <= vocabularySize {
    let pointer = UnsafeMutableRawPointer(logits.advanced(by: index))
    let values = UnsafeRawPointer(pointer).loadUnaligned(as: SIMD8<Float>.self)
    let mask = bitmaskSIMDTable[Int(maskBytes[index >> 3])]
    pointer.storeBytes(of: values + mask, as: SIMD8<Float>.self)
    index += SIMD8<Float>.scalarCount
  }
  while index < vocabularySize {
    logits[index] += bitmaskValue(maskBytes: maskBytes, index: index)
    index += 1
  }
}

@inline(always)
private func bitmaskValue(maskBytes: UnsafeBufferPointer<UInt8>, index: Int) -> Float {
  let tableIndex = Int(maskBytes[index >> 3]) * 8 + (index & 7)
  return bitmaskTable[tableIndex]
}

// MARK: - Lookup Tables

@usableFromInline
let bitmaskTable = (0..<256)
  .flatMap { byte in
    (0..<8).map { bit in ((byte >> bit) & 1) != 0 ? 0 : -Float.infinity }
  }

@usableFromInline
let bitmaskSIMDTable = bitmaskTable.withUnsafeBufferPointer { table in
  (0..<256)
    .map { byte in
      UnsafeRawPointer(table.baseAddress!.advanced(by: byte * 8))
        .loadUnaligned(as: SIMD8<Float>.self)
    }
}

// MARK: - Validation

@_transparent
private func validateBitmaskCoverage(mask: GrammarBitmask, vocabularySize: Int) {
  precondition(
    mask.count >= vocabularySize,
    "Grammar bitmask (\(mask.count) tokens) does not cover the model vocabulary (\(vocabularySize) tokens)."
  )
}
