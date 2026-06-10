// MARK: - NeedleDynamicToolInvocations

public struct NeedleDynamicToolCalls: Sendable {
  private let elements: [Element]
}

// MARK: - Collection

extension NeedleDynamicToolCalls: Collection {
  public typealias Index = Int

  public func index(after i: Int) -> Int { i + 1 }
  public var startIndex: Int { self.elements.startIndex }
  public var endIndex: Int { self.elements.endIndex }

  public subscript(position: Index) -> Element {
    _read { yield self.elements[position] }
  }
}

extension NeedleDynamicToolCalls: RandomAccessCollection {}

// MARK: - Element

extension NeedleDynamicToolCalls {
  public struct Element: Sendable {

    public func result<Tool: NeedleTool>(
      of tool: Tool.Type
    ) -> Result<NeedleToolCallOf<Tool>, any Error>? {
      nil
    }
  }
}
