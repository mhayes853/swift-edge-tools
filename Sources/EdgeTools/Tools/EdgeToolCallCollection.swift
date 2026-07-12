// MARK: - EdgeToolCalls

public struct EdgeToolCallCollection: Sendable {
  private var elements: [Element]
}

// MARK: - Collection

extension EdgeToolCallCollection: Collection {
  public typealias Element = AnyEdgeToolCall
  public typealias Index = Int

  public func index(after i: Int) -> Int { i + 1 }
  public var startIndex: Int { self.elements.startIndex }
  public var endIndex: Int { self.elements.endIndex }

  public subscript(position: Index) -> Element {
    _read { yield self.elements[position] }
  }
}

extension EdgeToolCallCollection: RandomAccessCollection {}
extension EdgeToolCallCollection: RangeReplaceableCollection {
  public init() {
    self.elements = []
  }

  public mutating func replaceSubrange(
    _ subrange: Range<Int>,
    with newElements: some Collection<Element>
  ) {
    self.elements.replaceSubrange(subrange, with: newElements)
  }
}

// MARK: - Strongly Typed Mutators

extension EdgeToolCallCollection {
  public mutating func append<Tool: EdgeTool>(_ toolCall: EdgeToolCall<Tool>) {
    self.append(AnyEdgeToolCall(toolCall))
  }

  public mutating func append<Tool: EdgeTool>(
    contentsOf toolCalls: some Collection<EdgeToolCall<Tool>>
  ) {
    self.append(contentsOf: toolCalls.lazy.map(AnyEdgeToolCall.init))
  }

  public mutating func insert<Tool: EdgeTool>(
    _ toolCall: EdgeToolCall<Tool>,
    at index: Int
  ) {
    self.insert(AnyEdgeToolCall(toolCall), at: index)
  }

  public mutating func insert<Tool: EdgeTool>(
    contentsOf toolCalls: some Collection<EdgeToolCall<Tool>>,
    at index: Int
  ) {
    self.insert(contentsOf: toolCalls.lazy.map(AnyEdgeToolCall.init), at: index)
  }

  public mutating func replaceSubrange<Tool: EdgeTool>(
    _ subrange: Range<Int>,
    with toolCalls: some Collection<EdgeToolCall<Tool>>
  ) {
    self.replaceSubrange(subrange, with: toolCalls.lazy.map(AnyEdgeToolCall.init))
  }
}

// MARK: - Strongly Typed Accessor

extension EdgeToolCallCollection {
  public subscript<Tool: EdgeTool>(index: Int, as type: Tool.Type) -> EdgeToolCall<Tool>? {
    self[index].as(type)
  }
}
