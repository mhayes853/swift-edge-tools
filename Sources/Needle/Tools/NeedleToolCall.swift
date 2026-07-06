import Foundation
import Observation

// MARK: - NeedleToolCallID

public struct NeedleToolCallID:
  Hashable, Sendable, RawRepresentable, Codable, ExpressibleByStringLiteral
{
  public var rawValue: String

  public init(rawValue: String) {
    self.rawValue = rawValue
  }

  public init() {
    self.init(rawValue: UUID().uuidString)
  }

  public init(stringLiteral value: StringLiteralType) {
    self.init(rawValue: value)
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    self.rawValue = try container.decode(String.self)
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    try container.encode(self.rawValue)
  }
}

// MARK: - NeedleToolCall

public final class NeedleToolCall<Tool: NeedleTool>: Sendable, Observable, Identifiable {
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

  public let id: NeedleToolCallID
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

  public var output: Tool.Output {
    get async throws {
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
        guard let output = try await task.cancellableValue else { throw CancellationError() }
        return output
      case .returnResult(let result):
        return try result.get()
      }
    }
  }

  public init(id: NeedleToolCallID, tool: Tool, input: Tool.Input) {
    self.id = id
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

public final class AnyNeedleToolCall: Sendable, Observable, Identifiable {
  public var tool: any NeedleTool {
    self.base._tool
  }

  public var input: any ConvertibleFromNeedleValue & Sendable {
    get { self.base._input }
    set { self.base._input = newValue }
  }

  public var status: NeedleToolCallStatus<any Sendable> {
    self.base._status
  }

  public var id: NeedleToolCallID {
    self.base._id
  }

  public var output: any Sendable {
    get async throws { try await self.base._output }
  }

  private let base: _AnyNeedleToolCall

  public init(_ toolCall: NeedleToolCall<some NeedleTool>) {
    self.base = toolCall
  }

  public func `as`<Tool: NeedleTool>(_: Tool.Type) -> NeedleToolCall<Tool>? {
    self.base as? NeedleToolCall<Tool>
  }
}

private protocol _AnyNeedleToolCall: Sendable {
  var _id: NeedleToolCallID { get }
  var _tool: any NeedleTool { get }
  var _input: any ConvertibleFromNeedleValue & Sendable { get nonmutating set }
  var _status: NeedleToolCallStatus<any Sendable> { get }
  var _output: any Sendable { get async throws }
}

extension NeedleToolCall: _AnyNeedleToolCall {
  var _id: NeedleToolCallID { self.id }
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

  var _status: NeedleToolCallStatus<any Sendable> {
    self.status.map { $0 }
  }

  var _output: any Sendable {
    get async throws { try await self.output }
  }
}

// MARK: - NeedleToolCallStatus

public enum NeedleToolCallStatus<Output> {
  case idle
  case running
  case finished(Result<Output, any Error>)

  public func map<T, E: Error>(
    _ body: (Output) throws(E) -> T
  ) throws(E) -> NeedleToolCallStatus<T> {
    switch self {
    case .idle: .idle
    case .running: .running
    case .finished(.success(let output)): .finished(.success(try body(output)))
    case .finished(.failure(let error)): .finished(.failure(error))
    }
  }
}

extension NeedleToolCallStatus: Sendable where Output: Sendable {}
