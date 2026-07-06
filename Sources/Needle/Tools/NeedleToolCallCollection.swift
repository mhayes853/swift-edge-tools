// MARK: - NeedleToolCalls

public struct NeedleToolCallCollection: Sendable {
  private var elements: [Element]
}

// MARK: - Collection

extension NeedleToolCallCollection: Collection {
  public typealias Element = AnyNeedleToolCall
  public typealias Index = Int

  public func index(after i: Int) -> Int { i + 1 }
  public var startIndex: Int { self.elements.startIndex }
  public var endIndex: Int { self.elements.endIndex }

  public subscript(position: Index) -> Element {
    _read { yield self.elements[position] }
  }
}

extension NeedleToolCallCollection: RandomAccessCollection {}
extension NeedleToolCallCollection: RangeReplaceableCollection {
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

extension NeedleToolCallCollection {
  public mutating func append<Tool: NeedleTool>(_ toolCall: NeedleToolCall<Tool>) {
    self.append(AnyNeedleToolCall(toolCall))
  }

  public mutating func append<Tool: NeedleTool>(
    contentsOf toolCalls: some Collection<NeedleToolCall<Tool>>
  ) {
    self.append(contentsOf: toolCalls.lazy.map(AnyNeedleToolCall.init))
  }

  public mutating func insert<Tool: NeedleTool>(
    _ toolCall: NeedleToolCall<Tool>,
    at index: Int
  ) {
    self.insert(AnyNeedleToolCall(toolCall), at: index)
  }

  public mutating func insert<Tool: NeedleTool>(
    contentsOf toolCalls: some Collection<NeedleToolCall<Tool>>,
    at index: Int
  ) {
    self.insert(contentsOf: toolCalls.lazy.map(AnyNeedleToolCall.init), at: index)
  }

  public mutating func replaceSubrange<Tool: NeedleTool>(
    _ subrange: Range<Int>,
    with toolCalls: some Collection<NeedleToolCall<Tool>>
  ) {
    self.replaceSubrange(subrange, with: toolCalls.lazy.map(AnyNeedleToolCall.init))
  }
}

// MARK: - Strongly Typed Accessor

extension NeedleToolCallCollection {
  public subscript<Tool: NeedleTool>(index: Int, as type: Tool.Type) -> NeedleToolCall<Tool>? {
    self[index].as(type)
  }
}
