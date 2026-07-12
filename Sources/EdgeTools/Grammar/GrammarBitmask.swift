// MARK: - GrammarBitmask

public struct GrammarBitmask: Hashable, Sendable {
  public var storage: [UInt8]

  @inlinable
  @inline(__always)
  public init(storage: [UInt8]) {
    self.storage = storage
  }

  @inlinable
  @inline(__always)
  public init(bitCount: Int, repeating value: Bool = false) {
    precondition(bitCount.isMultiple(of: 8), "Grammar bit count must be a multiple of 8.")
    self.storage = [UInt8](repeating: value ? .max : .zero, count: bitCount / 8)
  }
}

// MARK: - Collection

extension GrammarBitmask: MutableCollection {
  public typealias Index = Int
  public typealias Element = Bool

  public func index(after index: Int) -> Int { index + 1 }
  public var startIndex: Int { self.storage.startIndex }
  public var endIndex: Int { self.storage.endIndex * 8 }

  @inlinable
  @inline(__always)
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
