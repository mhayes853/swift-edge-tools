// MARK: - NeedleStaticToolCalls

public struct NeedleStaticToolCalls<Collection: NeedleStaticToolsCollection>: Sendable {
  private let elements: [Element]
}

// MARK: - Collection

extension NeedleStaticToolCalls: Swift.Collection {
  public typealias Index = Int

  public func index(after i: Int) -> Int { i + 1 }
  public var startIndex: Int { self.elements.startIndex }
  public var endIndex: Int { self.elements.endIndex }

  public subscript(position: Index) -> Element {
    _read { yield self.elements[position] }
  }
}

extension NeedleStaticToolCalls: RandomAccessCollection {}

// MARK: - Element

extension NeedleStaticToolCalls {
  public struct Element: Sendable {
    public var status: NeedleToolCallStatus<Collection.Output> {
      fatalError()
    }

    public func result<Tool: NeedleTool>(
      of tool: Tool.Type
    ) -> Result<NeedleToolCall<Tool>, any Error>? {
      nil
    }
  }
}
