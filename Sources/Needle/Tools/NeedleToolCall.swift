// MARK: - NeedleToolCall

public final class NeedleToolCall<Tool: NeedleTool>: Sendable {
  private struct State {
    var status = NeedleToolCallStatus<Tool.Output>.idle
    var input: Tool.Input
  }

  private let state: Lock<State>

  public var input: Tool.Input {
    get { self.state.withLock { $0.input } }
    set { self.state.withLock { $0.input = newValue } }
  }

  public let tool: Tool

  public var status: NeedleToolCallStatus<Tool.Output> {
    self.state.withLock { $0.status }
  }

  public init(tool: Tool, input: Tool.Input) {
    self.tool = tool
    self.state = Lock(State(input: input))
  }

  public func invoke() async throws -> sending Tool.Output {
    fatalError()
  }
}

// MARK: - AnyNeedleToolCall

public final class AnyNeedleToolCall: Sendable {
  public var tool: any NeedleTool {
    self.base._tool
  }

  public var input: any ConvertibleFromNeedleValue & Sendable {
    get { self.base._input }
    set { self.base._input = newValue }
  }

  public var status: NeedleToolCallStatus<Any> {
    self.base._status
  }

  private let base: _AnyNeedleToolCall

  public init(_ toolCall: NeedleToolCall<some NeedleTool>) {
    self.base = toolCall
  }

  public func `as`<Tool: NeedleTool>(_: Tool.Type) -> NeedleToolCall<Tool>? {
    self.base as? NeedleToolCall<Tool>
  }

  public func invoke() async throws -> sending Any {
    try await self.base._invoke()
  }
}

private protocol _AnyNeedleToolCall: Sendable {
  var _tool: any NeedleTool { get }
  var _input: any ConvertibleFromNeedleValue & Sendable { get nonmutating set }
  var _status: NeedleToolCallStatus<Any> { get }
  func _invoke() async throws -> sending Any
}

extension NeedleToolCall: _AnyNeedleToolCall {
  var _tool: any NeedleTool { self.tool }

  var _input: any ConvertibleFromNeedleValue & Sendable {
    get { self.input }
    set {
      guard let input = newValue as? Tool.Input else {
        fatalError("New input value must have the same type as the existing input value.")
      }
      self.input = input
    }
  }

  var _status: NeedleToolCallStatus<Any> {
    switch self.status {
    case .idle: .idle
    case .running: .running
    case .finished(let result): .finished(result.map { $0 })
    }
  }

  func _invoke() async throws -> sending Any {
    try await self.invoke()
  }
}

// MARK: - NeedleToolCallStatus

public enum NeedleToolCallStatus<Output> {
  case idle
  case running
  case finished(Result<Output, any Error>)
}

extension NeedleToolCallStatus: Sendable where Output: Sendable {}
