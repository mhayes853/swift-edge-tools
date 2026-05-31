// MARK: - NeedleGrammarBitmask

public struct NeedleGrammarBitmask: Hashable, Sendable {
  public var storage: [Int32]

  @inlinable
  @inline(__always)
  public init(storage: [Int32]) {
    self.storage = storage
  }

  @inlinable
  @inline(__always)
  public init() {
    self.storage = [Int32](repeating: 0, count: 256)
  }
}

// MARK: - Collection

extension NeedleGrammarBitmask: MutableCollection {
  public typealias Index = Int
  public typealias Element = Bool

  public func index(after i: Int) -> Int { i + 1 }
  public var startIndex: Int { self.storage.startIndex }
  public var endIndex: Int { self.storage.endIndex * 32 }

  @inlinable
  @inline(__always)
  public subscript(position: Index) -> Bool {
    get { Int((self.storage[position / 32] & (1 << (position % 32)))) != 0 }
    set {
      let index = position / 32
      let mask = 1 &<< Int32(position % 32)
      self.storage[index] = (self.storage[index] & ~mask) | (newValue ? mask : 0)
    }
  }
}

extension NeedleGrammarBitmask: RandomAccessCollection {}
