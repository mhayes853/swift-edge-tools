import Foundation
import Observation

// MARK: - EdgeToolCallID

public struct EdgeToolCallID:
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

// MARK: - EdgeToolCall

public final class EdgeToolCall<Tool: EdgeTool>: Sendable, Observable, Identifiable {
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
    var task: Task<Void, any Error>?
  }

  private let state: Lock<State>
  private let registrar = ObservationRegistrar()

  public let rawValue: EdgeRawToolCall
  public let input: Tool.Input
  public let id: EdgeToolCallID
  public let tool: Tool

  public var status: EdgeToolCallStatus<Tool.Output> {
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

  public init(
    id: EdgeToolCallID,
    tool: Tool,
    rawInput: EdgeToolsValue
  ) throws(Tool.Input.EdgeToolsConversionFailure) {
    self.id = id
    self.tool = tool
    self.rawValue = EdgeRawToolCall(name: tool.name, arguments: rawInput)
    self.input = try Tool.Input(edgeToolsValue: rawInput)
    self.state = Lock(State())
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

// MARK: - AnyEdgeToolCall

public final class AnyEdgeToolCall: Sendable, Observable, Identifiable {
  public var tool: any EdgeTool {
    self.base.erasedTool
  }

  public var rawValue: EdgeRawToolCall {
    self.base.erasedRawValue
  }

  public var input: any ConvertibleFromEdgeToolsValue & Sendable {
    self.base.erasedInput
  }

  public var status: EdgeToolCallStatus<any Sendable> {
    self.base.erasedStatus
  }

  public var id: EdgeToolCallID {
    self.base.erasedID
  }

  public var output: any Sendable {
    get async throws { try await self.base.erasedOutput }
  }

  private let base: _AnyEdgeToolCall

  public init(_ toolCall: EdgeToolCall<some EdgeTool>) {
    self.base = toolCall
  }

  public func `as`<Tool: EdgeTool>(_: Tool.Type) -> EdgeToolCall<Tool>? {
    self.base as? EdgeToolCall<Tool>
  }
}

private protocol _AnyEdgeToolCall: Sendable {
  var erasedID: EdgeToolCallID { get }
  var erasedTool: any EdgeTool { get }
  var erasedRawValue: EdgeRawToolCall { get }
  var erasedInput: any ConvertibleFromEdgeToolsValue & Sendable { get }
  var erasedStatus: EdgeToolCallStatus<any Sendable> { get }
  var erasedOutput: any Sendable { get async throws }
}

extension EdgeToolCall: _AnyEdgeToolCall {
  var erasedID: EdgeToolCallID { self.id }
  var erasedTool: any EdgeTool { self.tool }
  var erasedRawValue: EdgeRawToolCall { self.rawValue }

  var erasedInput: any ConvertibleFromEdgeToolsValue & Sendable {
    self.input
  }

  var erasedStatus: EdgeToolCallStatus<any Sendable> {
    self.status.map { $0 }
  }

  var erasedOutput: any Sendable {
    get async throws { try await self.output }
  }
}

// MARK: - EdgeToolCallStatus

public enum EdgeToolCallStatus<Output> {
  case idle
  case running
  case finished(Result<Output, any Error>)

  @inlinable
  public func map<T, E: Error>(
    _ body: (Output) throws(E) -> T
  ) throws(E) -> EdgeToolCallStatus<T> {
    switch self {
    case .idle: .idle
    case .running: .running
    case .finished(.success(let output)): .finished(.success(try body(output)))
    case .finished(.failure(let error)): .finished(.failure(error))
    }
  }

  @inlinable
  public func flatMap<T, E: Error>(
    _ body: (Output) throws(E) -> EdgeToolCallStatus<T>
  ) throws(E) -> EdgeToolCallStatus<T> {
    switch self {
    case .idle: .idle
    case .running: .running
    case .finished(.success(let output)): try body(output)
    case .finished(.failure(let error)): .finished(.failure(error))
    }
  }
}

extension EdgeToolCallStatus: Sendable where Output: Sendable {}
