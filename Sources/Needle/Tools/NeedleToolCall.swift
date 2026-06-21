// MARK: - NeedleToolCall

public final class NeedleToolCall<Tool: NeedleTool>: Sendable {
  private struct State {
    let input: Tool.Input
    let tool: Tool
    var status = NeedleToolCallStatus<Tool.Output>.idle
  }

  private let state: Lock<State>

  public var input: Tool.Input {
    self.state.withLock { $0.input }
  }

  public var status: NeedleToolCallStatus<Tool.Output> {
    self.state.withLock { $0.status }
  }

  public init(tool: sending Tool, input: sending Tool.Input) {
    self.state = Lock(State(input: input, tool: tool))
  }

  public func invoke() async throws -> Tool.Output {
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
