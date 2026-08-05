import OrderedCollections
import _Concurrency

// MARK: - EdgeRawToolCall

public struct EdgeRawToolCall: Hashable, Sendable {
  public let name: String
  public let arguments: EdgeToolsValue

  public init(name: String, arguments: EdgeToolsValue) {
    self.name = name
    self.arguments = arguments
  }

  package init?(jsonValue: EdgeToolsValue) {
    guard
      case .object(let object) = jsonValue,
      case .string(let name) = object["name"],
      let arguments = object["arguments"]
    else { return nil }
    self.init(name: name, arguments: arguments)
  }
}

#if !$Embedded
  import Observation
#endif

// MARK: - EdgeTool

public protocol EdgeTool<Input, Output>: Sendable {
  associatedtype Input: ConvertibleFromEdgeToolsValue & Sendable
  associatedtype Output: Sendable
  associatedtype Failure: Error & Sendable = any Error & Sendable

  var name: String { get }
  var description: String { get }
  var arguments: EdgeToolsGenerationSchema { get }
  var includesSchemaInInstructions: Bool { get }

  func invoke(input: Input) async throws(Failure) -> Output
}

extension EdgeTool where Input: EdgeToolsGenerable {
  public var arguments: EdgeToolsGenerationSchema {
    Input.edgeToolsGenerationSchema
  }
}

extension EdgeTool {
  public var includesSchemaInInstructions: Bool { true }

  public var definition: EdgeToolDefinition {
    EdgeToolDefinition(
      name: self.name,
      description: self.description,
      arguments: self.arguments,
      includesSchemaInInstructions: self.includesSchemaInInstructions
    )
  }
}

// MARK: - EdgeToolDefinition

public struct EdgeToolDefinition: Hashable, Sendable {
  public var name: String
  public var description: String
  public var arguments: EdgeToolsGenerationSchema
  public var includesSchemaInInstructions: Bool

  public init(
    name: String,
    description: String,
    arguments: EdgeToolsGenerationSchema,
    includesSchemaInInstructions: Bool = true
  ) {
    self.name = name
    self.description = description
    self.arguments = arguments
    self.includesSchemaInInstructions = includesSchemaInInstructions
  }
}

// MARK: - EdgeToolCallID

public struct EdgeToolCallID: Hashable, Sendable, RawRepresentable, ExpressibleByStringLiteral {
  public var rawValue: String

  public init(rawValue: String) {
    self.rawValue = rawValue
  }

  public init(
    randomNumberGenerator: some RandomNumberGenerator = SystemRandomNumberGenerator()
  ) {
    var randomNumberGenerator = randomNumberGenerator
    self.init(rawValue: makeEdgeToolCallID(using: &randomNumberGenerator))
  }

  public init(stringLiteral value: String) {
    self.init(rawValue: value)
  }
}

#if !$Embedded
  extension EdgeRawToolCall: Codable {}
  extension EdgeToolDefinition: Codable {}

  extension EdgeToolCallID: Codable {
    public init(from decoder: any Decoder) throws {
      let container = try decoder.singleValueContainer()
      self.rawValue = try container.decode(String.self)
    }

    public func encode(to encoder: any Encoder) throws {
      var container = encoder.singleValueContainer()
      try container.encode(self.rawValue)
    }
  }
#endif

private func makeEdgeToolCallID(using generator: inout some RandomNumberGenerator) -> String {
  var bytes = (0..<16).map { _ in UInt8.random(in: .min ... .max, using: &generator) }
  bytes[6] = (bytes[6] & 0x0F) | 0x40
  bytes[8] = (bytes[8] & 0x3F) | 0x80

  let hexDigits = Array("0123456789ABCDEF".utf8)
  var encoded = [UInt8]()
  encoded.reserveCapacity(36)
  for index in bytes.indices {
    if [4, 6, 8, 10].contains(index) { encoded.append(UInt8(ascii: "-")) }
    encoded.append(hexDigits[Int(bytes[index] >> 4)])
    encoded.append(hexDigits[Int(bytes[index] & 0x0F)])
  }
  return String(decoding: encoded, as: UTF8.self)
}

// MARK: - EdgeToolCall

public final class EdgeToolCall<Tool: EdgeTool>: Sendable, Identifiable {
  private enum Status {
    case idle
    case running(Task<Result<Tool.Output, Tool.Failure>, Never>)
    case finished(Result<Tool.Output, Tool.Failure>)
  }

  private enum InvokeAction {
    case awaitTask(Task<Result<Tool.Output, Tool.Failure>, Never>)
    case returnResult(Result<Tool.Output, Tool.Failure>)
  }

  private struct State {
    var status = Status.idle
  }

  private let state: Lock<State>
  private let registrar = _ObservationRegistrar()

  public let rawValue: EdgeRawToolCall
  public let input: Tool.Input
  public let id: EdgeToolCallID
  public let tool: Tool

  public var status: EdgeToolCallStatus<Tool.Output, Tool.Failure> {
    self.accessStatus()
    return self.state.withLock {
      switch $0.status {
      case .idle: .idle
      case .running: .running
      case .finished(let result): .finished(result)
      }
    }
  }

  public var output: Tool.Output {
    get async throws(Tool.Failure) {
      let action: InvokeAction = self.state.withLock { state in
        switch state.status {
        case .idle:
          let task = Task { await self.runResult() }
          self.withStatusMutation {
            state.status = .running(task)
          }
          return .awaitTask(task)
        case .running(let task):
          return .awaitTask(task)
        case .finished(let result):
          return .returnResult(result)
        }
      }

      switch await self.result(for: action) {
      case .success(let output): return output
      case .failure(let failure): throw failure
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

  private func result(for action: InvokeAction) async -> Result<Tool.Output, Tool.Failure> {
    switch action {
    case .awaitTask(let task):
      await withTaskCancellationHandler {
        await task.value
      } onCancel: {
        task.cancel()
      }
    case .returnResult(let result): result
    }
  }

  private func runResult() async -> Result<Tool.Output, Tool.Failure> {
    do {
      return .success(try await self.run())
    } catch {
      return .failure(error)
    }
  }

  private func run() async throws(Tool.Failure) -> Tool.Output {
    do {
      let output = try await self.tool.invoke(input: self.input)
      self.withStatusMutation {
        self.state.withLock { $0.status = .finished(.success(output)) }
      }
      return output
    } catch {
      self.withStatusMutation {
        self.state.withLock { $0.status = .finished(.failure(error)) }
      }
      throw error
    }
  }
}

// MARK: - AnyEdgeToolCall

public final class AnyEdgeToolCall: Sendable, Identifiable {
  public var tool: any EdgeTool {
    self.base.erasedTool
  }

  public var rawValue: EdgeRawToolCall {
    self.base.erasedRawValue
  }

  public var input: any ConvertibleFromEdgeToolsValue & Sendable {
    self.base.erasedInput
  }

  public var status: EdgeToolCallStatus<any Sendable, any Error> {
    self.base.erasedStatus
  }

  public var id: EdgeToolCallID {
    self.base.erasedID
  }

  public var output: any Sendable {
    get async throws { try await self.base.erasedOutput }
  }

  private let base: any _AnyEdgeToolCall

  private init(base: any _AnyEdgeToolCall) {
    self.base = base
  }

  public static func erasing<Tool: EdgeTool>(_ toolCall: EdgeToolCall<Tool>) -> AnyEdgeToolCall {
    AnyEdgeToolCall(base: toolCall)
  }

  public func `as`<Tool: EdgeTool>(_: Tool.Type) -> EdgeToolCall<Tool>? {
    self.base as? EdgeToolCall<Tool>
  }
}

#if !$Embedded
  extension AnyEdgeToolCall {
    public convenience init(_ toolCall: EdgeToolCall<some EdgeTool>) {
      self.init(base: toolCall)
    }
  }
#endif

private protocol _AnyEdgeToolCall: AnyObject, Sendable {
  var erasedID: EdgeToolCallID { get }
  var erasedTool: any EdgeTool { get }
  var erasedRawValue: EdgeRawToolCall { get }
  var erasedInput: any ConvertibleFromEdgeToolsValue & Sendable { get }
  var erasedStatus: EdgeToolCallStatus<any Sendable, any Error> { get }
  var erasedOutput: any Sendable { get async throws }
}

extension EdgeToolCall: _AnyEdgeToolCall {
  var erasedID: EdgeToolCallID { self.id }
  var erasedTool: any EdgeTool { self.tool }
  var erasedRawValue: EdgeRawToolCall { self.rawValue }

  var erasedInput: any ConvertibleFromEdgeToolsValue & Sendable {
    self.input
  }

  var erasedStatus: EdgeToolCallStatus<any Sendable, any Error> {
    self.status.erasingOutputAndFailure()
  }

  var erasedOutput: any Sendable {
    get async throws { try await self.output }
  }
}

// MARK: - Observation

extension EdgeToolCall {
  fileprivate func accessStatus() {
    #if !$Embedded
      self.registrar.access(self, keyPath: \.status)
    #endif
  }

  fileprivate func withStatusMutation<Result>(_ body: () -> Result) -> Result {
    #if !$Embedded
      self.registrar.withMutation(of: self, keyPath: \.status, body)
    #else
      body()
    #endif
  }
}

#if !$Embedded
  extension EdgeToolCall: Observable {}
  extension AnyEdgeToolCall: Observable {}
#endif

// MARK: - EdgeToolCallStatus

public enum EdgeToolCallStatus<Output, Failure: Error> {
  case idle
  case running
  case finished(Result<Output, Failure>)

  @inlinable
  public func map<T>(
    _ body: (Output) throws(Failure) -> T
  ) throws(Failure) -> EdgeToolCallStatus<T, Failure> {
    switch self {
    case .idle: .idle
    case .running: .running
    case .finished(.success(let output)): .finished(.success(try body(output)))
    case .finished(.failure(let error)): .finished(.failure(error))
    }
  }

  @inlinable
  public func flatMap<T>(
    _ body: (Output) throws(Failure) -> EdgeToolCallStatus<T, Failure>
  ) throws(Failure) -> EdgeToolCallStatus<T, Failure> {
    switch self {
    case .idle: .idle
    case .running: .running
    case .finished(.success(let output)): try body(output)
    case .finished(.failure(let error)): .finished(.failure(error))
    }
  }
}

extension EdgeToolCallStatus where Output: Sendable {
  fileprivate func erasingOutputAndFailure() -> EdgeToolCallStatus<any Sendable, any Error> {
    switch self {
    case .idle: .idle
    case .running: .running
    case .finished(.success(let output)): .finished(.success(output))
    case .finished(.failure(let error)): .finished(.failure(error))
    }
  }
}

extension EdgeToolCallStatus: Sendable where Output: Sendable, Failure: Sendable {}

// MARK: - EdgeToolCalls

public struct EdgeToolCallCollection: Sendable, RangeReplaceableCollection {
  public typealias Element = AnyEdgeToolCall
  public typealias Index = Int

  private var elements: [Element]

  public var startIndex: Int { self.elements.startIndex }
  public var endIndex: Int { self.elements.endIndex }

  public init() {
    self.elements = []
  }

  public subscript(position: Index) -> Element {
    _read { yield self.elements[position] }
  }

  public subscript<Tool: EdgeTool>(index: Int, as type: Tool.Type) -> EdgeToolCall<Tool>? {
    self[index].as(type)
  }

  public func index(after index: Int) -> Int {
    index + 1
  }

  public mutating func replaceSubrange(
    _ subrange: Range<Int>,
    with newElements: some Collection<Element>
  ) {
    self.elements.replaceSubrange(subrange, with: newElements)
  }

  public mutating func append<Tool: EdgeTool>(_ toolCall: EdgeToolCall<Tool>) {
    self.append(.erasing(toolCall))
  }

  public mutating func append<Tool: EdgeTool>(
    contentsOf toolCalls: some Collection<EdgeToolCall<Tool>>
  ) {
    self.append(contentsOf: toolCalls.map { .erasing($0) })
  }

  public mutating func insert<Tool: EdgeTool>(_ toolCall: EdgeToolCall<Tool>, at index: Int) {
    self.insert(.erasing(toolCall), at: index)
  }

  public mutating func insert<Tool: EdgeTool>(
    contentsOf toolCalls: some Collection<EdgeToolCall<Tool>>,
    at index: Int
  ) {
    self.insert(contentsOf: toolCalls.map { .erasing($0) }, at: index)
  }

  public mutating func replaceSubrange<Tool: EdgeTool>(
    _ subrange: Range<Int>,
    with toolCalls: some Collection<EdgeToolCall<Tool>>
  ) {
    self.replaceSubrange(subrange, with: toolCalls.map { .erasing($0) })
  }
}

extension EdgeToolCallCollection: RandomAccessCollection {}
