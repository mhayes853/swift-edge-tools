// MARK: - NeedleToolCalls

public struct NeedleToolCallCollection: Sendable {
  private let elements: [Element]
}

// MARK: - Collection

extension NeedleToolCallCollection: Collection {
  public typealias Index = Int

  public func index(after i: Int) -> Int { i + 1 }
  public var startIndex: Int { self.elements.startIndex }
  public var endIndex: Int { self.elements.endIndex }

  public subscript(position: Index) -> Element {
    _read { yield self.elements[position] }
  }
}

extension NeedleToolCallCollection: RandomAccessCollection {}

// MARK: - Element

extension NeedleToolCallCollection {
  public struct Element: Sendable {
    public func call<Tool: NeedleTool>(for tool: Tool.Type) -> NeedleToolCall<Tool>? {
      nil
    }
  }
}
