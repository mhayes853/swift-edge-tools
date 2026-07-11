// MARK: - GrammarBitmask

public struct GrammarBitmask: Hashable, Sendable {
  public var storage: [Int32]

  @inlinable
  @inline(__always)
  public init(storage: [Int32]) {
    self.storage = storage
  }

  @inlinable
  @inline(__always)
  public init(wordCount: Int) {
    self.storage = [Int32](repeating: 0, count: wordCount)
  }

  @inlinable
  @inline(__always)
  public init() {
    self.init(wordCount: 256)
  }
}

// MARK: - Collection

extension GrammarBitmask: MutableCollection {
  public typealias Index = Int
  public typealias Element = Bool

  public func index(after index: Int) -> Int { index + 1 }
  public var startIndex: Int { self.storage.startIndex }
  public var endIndex: Int { self.storage.endIndex * 32 }

  @inlinable
  @inline(__always)
  public subscript(position: Index) -> Bool {
    get { (self.storage[position / 32] & (1 << (position % 32))).boolValue }
    set {
      let index = position / 32
      let mask = 1 &<< Int32(position % 32)
      self.storage[index] = (self.storage[index] & ~mask) | (newValue ? mask : 0)
    }
  }
}

extension GrammarBitmask: RandomAccessCollection {}
