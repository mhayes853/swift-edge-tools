// MARK: - NeedleGrammarBitmask

public struct NeedleGrammarBitmask: Hashable, Sendable {
  private var elements: [Int]

  public init() {
    self.elements = []
  }
}

// MARK: - Collection

extension NeedleGrammarBitmask: MutableCollection {
  public typealias Index = Int
  public typealias Element = Int

  public func index(after i: Int) -> Int { i + 1 }
  public var startIndex: Int { 0 }
  public var endIndex: Int { fatalError() }

  public subscript(position: Index) -> Element {
    get { fatalError() }
    set {}
  }
}

extension NeedleGrammarBitmask: RangeReplaceableCollection {
  public func replaceSubrange<C>(_ subrange: Range<Int>, with newElements: C)
  where C: Collection, Int == C.Element {
  }
}

extension NeedleGrammarBitmask: RandomAccessCollection {}
