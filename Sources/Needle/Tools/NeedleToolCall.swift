import Foundation
import Observation

// MARK: - NeedleToolCall

public final class NeedleToolCall<Tool: NeedleTool>: Sendable, Observable {
  private enum Status {
    case idle
    case running(Task<Tool.Output?, any Error>)
    case finished(Result<Tool.Output, any Error>)
  }

  private enum InvokeAction {
    case awaitTask(Task<Tool.Output?, any Error>)
    case returnResult(Result<Tool.Output, any Error>)
  }

  private struct State {
    var status = Status.idle
    var input: Tool.Input
    var task: Task<Void, any Error>?
  }

  private let state: Lock<State>
  private let registrar = ObservationRegistrar()

  public var input: Tool.Input {
    get {
      self.registrar.access(self, keyPath: \.input)
      return self.state.withLock { $0.input }
    }
    set {
      self.registrar.withMutation(of: self, keyPath: \.input) {
        self.state.withLock { $0.input = newValue }
      }
    }
  }

  public let tool: Tool

  public var status: NeedleToolCallStatus<Tool.Output> {
    self.registrar.access(self, keyPath: \.status)
    return self.state.withLock {
      switch $0.status {
      case .idle: .idle
      case .running: .running
      case .finished(let result): .finished(result)
      }
    }
  }

  public init(tool: Tool, input: Tool.Input) {
    self.tool = tool
    self.state = Lock(State(input: input))
  }

  deinit {
    self.state.withLock {
      switch $0.status {
      case .running(let task): task.cancel()
      default: break
      }
    }
  }

  public func invoke() async throws -> Tool.Output {
    let action: InvokeAction = self.state.withLock { state in
      switch state.status {
      case .idle:
        let task = Task { [weak self] in try await self?.run() }
        self.registrar.withMutation(of: self, keyPath: \.status) {
          state.status = .running(task)
        }
        return .awaitTask(task)
      case .running(let task):
        return .awaitTask(task)
      case .finished(let result):
        return .returnResult(result)
      }
    }

    switch action {
    case .awaitTask(let task):
      let output = try await withTaskCancellationHandler {
        try await task.value
      } onCancel: {
        task.cancel()
      }
      guard let output else { throw CancellationError() }
      return output
    case .returnResult(let result):
      return try result.get()
    }
  }

  private func run() async throws -> Tool.Output {
    do {
      let output = try await self.tool.invoke(input: self.input)
      self.registrar.withMutation(of: self, keyPath: \.status) {
        self.state.withLock { $0.status = .finished(.success(output)) }
      }
      return output
    } catch {
      self.registrar.withMutation(of: self, keyPath: \.status) {
        self.state.withLock { $0.status = .finished(.failure(error)) }
      }
      throw error
    }
  }
}

// MARK: - AnyNeedleToolCall

public final class AnyNeedleToolCall: Sendable, Observable {
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
