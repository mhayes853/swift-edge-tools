import EdgeToolsCore

#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && canImport(MLX)
  import MLX
  import MLXLMCommon
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

  /// The number of bits needed to store an XGrammar vocabulary mask.
  ///
  /// XGrammar packs masks into 32-bit words, so the result may be larger than the vocabulary.
  @inlinable
  @inline(always)
  public static func bitCount(forVocabularySize vocabularySize: Int) -> Int {
    precondition(vocabularySize >= 0, "Vocabulary size must not be negative.")
    return ((vocabularySize + 31) / 32) * 32
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

    public init(matcher: consuming Matcher) {
      self.matcher = consume matcher
    }

    public func prompt(_ prompt: MLXArray) {}

    public func didSample(token: MLXArray) {
      guard !self.matcher.isTerminated else { return }
      self.matcher.accept(tokenId: token.item(EdgeToolsToken.ID.self))
    }

    public func process(logits: MLXArray) -> MLXArray {
      guard !self.matcher.isTerminated else { return logits }
      guard let bitmask = self.matcher.grammarBitmask() else { return logits }
      return applyBitmaskMLX(logits: logits, mask: bitmask)
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

// MARK: - CPU

public func applyBitmaskCPU(logits: inout MutableSpan<Float>, mask: GrammarBitmask) {
  logits.withUnsafeMutableBufferPointer { applyBitmaskCPU(logits: $0, mask: mask) }
}

public func applyBitmaskCPU(logits: UnsafeMutableBufferPointer<Float>, mask: GrammarBitmask) {
  validateBitmaskCoverage(mask: mask, vocabularySize: logits.count)
  guard let base = logits.baseAddress else { return }
  var index = 0
  for byte in mask.storage {
    guard index < logits.count else { break }
    if byte == .max {
      index += 8
      continue
    }
    if index + 8 <= logits.count {
      let pointer = UnsafeMutableRawPointer(base.advanced(by: index))
      let block = pointer.loadUnaligned(as: SIMD8<Float>.self)
      pointer.storeBytes(of: block + bitmaskSIMDTable[Int(byte)], as: SIMD8<Float>.self)
    } else {
      for bit in 0..<8 where index + bit < logits.count {
        if byte & (1 &<< UInt8(bit)) == 0 {
          logits[index + bit] = -.infinity
        }
      }
    }
    index += 8
  }
}

// MARK: - Lookup Tables

let bitmaskSIMDTable = (0..<256)
  .map { byte in
    SIMD8<Float>((0..<8).map { bit in ((byte >> bit) & 1) != 0 ? 0 : -Float.infinity })
  }

@usableFromInline
let bitmaskTable = (0..<256)
  .flatMap { byte in
    (0..<8).map { bit in ((byte >> bit) & 1) != 0 ? 0 : -Float.infinity }
  }

// MARK: - Validation

@_transparent
func validateBitmaskCoverage(mask: GrammarBitmask, vocabularySize: Int) {
  precondition(
    mask.count >= vocabularySize,
    "Grammar bitmask (\(mask.count) tokens) does not cover the model vocabulary (\(vocabularySize) tokens)."
  )
}
