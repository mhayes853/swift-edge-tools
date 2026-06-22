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

// MARK: - Invoke

extension NeedleToolCallCollection {
  public struct InvocationError: Error, Sendable {
    public let failures: [Failure]

    public init(failures: [Failure]) {
      self.failures = failures
    }

    public struct Failure: Sendable {
      public let toolCall: AnyNeedleToolCall
      public let error: any Error

      public init(toolCall: AnyNeedleToolCall, error: any Error) {
        self.toolCall = toolCall
        self.error = error
      }
    }
  }

  public func invokeAll() async throws {
    try await self.invokeAll(where: { _ in true })
  }

  public func invokeAll(where predicate: (Element) -> Bool) async throws {
    let failures = await withTaskGroup(of: InvocationError.Failure?.self) { group in
      for call in self.filter(predicate) {
        group.addTask {
          do {
            _ = try await call.invoke()
            return nil
          } catch {
            return InvocationError.Failure(toolCall: call, error: error)
          }
        }
      }
      return await group.reduce(into: [InvocationError.Failure?]()) { $0.append($1) }
        .compactMap { $0 }
    }
    if !failures.isEmpty {
      throw InvocationError(failures: failures)
    }
  }
}
