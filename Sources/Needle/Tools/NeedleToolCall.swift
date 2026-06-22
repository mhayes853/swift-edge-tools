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

// MARK: - NeedleToolCallStatus

public enum NeedleToolCallStatus<Output> {
  case idle
  case running
  case finished(Result<Output, any Error>)
}

extension NeedleToolCallStatus: Sendable where Output: Sendable {}
