import Foundation
import Observation

// MARK: - NeedleSession

public final class NeedleSession<Engine: NeedleEngine>: Sendable, Observable {
  private let engine: RecursiveLock<Engine>
  private let _systemPrompt: Lock<String>
  private let _activeStreams: Lock<[NeedleSessionStream]>
  private let observationRegistrar = ObservationRegistrar()

  public var systemPrompt: String {
    get {
      self.observationRegistrar.access(self, keyPath: \.systemPrompt)
      return self._systemPrompt.withLock { $0 }
    }
    set {
      self.observationRegistrar.withMutation(of: self, keyPath: \.systemPrompt) {
        self._systemPrompt.withLock { $0 = newValue }
      }
    }
  }

  public var isResponding: Bool {
    !self.activeStreams.isEmpty
  }

  public var activeStreams: [NeedleSessionStream] {
    self.observationRegistrar.access(self, keyPath: \.activeStreams)
    return self._activeStreams.withLock { $0 }
  }

  public init(engine: sending Engine, systemPrompt: String = "") {
    self.engine = RecursiveLock(engine)
    self._systemPrompt = Lock(systemPrompt)
    self._activeStreams = Lock([])
  }

  public func withEngine<T, E: Error>(
    perform body: (Engine) throws(E) -> sending T
  ) throws(E) -> sending T {
    try self.engine.withLock { (engine: inout sending Engine) throws(E) -> sending T in
      try body(engine)
    }
  }

  public func reset() {
    self.withEngine { $0.reset() }
  }

  fileprivate func removeActiveStream(_ stream: NeedleSessionStream) {
    self._activeStreams.withLock { activeStreams in
      self.observationRegistrar.withMutation(of: self, keyPath: \.activeStreams) {
        activeStreams.removeAll { $0 === stream }
      }
    }
  }
}

// MARK: - Generate

public struct NeedleSessionGeneration: Sendable {
  public let engineGeneration: NeedleEngineGeneration
  public let toolCalls: NeedleToolCallCollection
}

extension NeedleSession {
  @concurrent
  public func generate(
    tools: [any NeedleTool],
    with prompt: String,
    systemPromptOverride: String? = nil,
    parameters: sending Engine.GenerateParameters = .default,
    shouldInvokeTools: Bool = true
  ) async throws -> NeedleSessionGeneration {
    let stream = self.stream(
      tools: tools,
      with: prompt,
      systemPromptOverride: systemPromptOverride,
      parameters: parameters,
      shouldInvokeTools: shouldInvokeTools
    )
    return try await stream.finalGeneration
  }
}

// MARK: - Stream

public final class NeedleSessionStream: Sendable, Observable, Identifiable {
  public enum Status: Sendable {
    case awaitingExecution
    case generating
    case finished(Result<NeedleSessionGeneration, any Error>)
  }

  public var finalGeneration: NeedleSessionGeneration {
    get async throws {
      let task = self.state.withLock { $0.task! }  // NB: Task is set in init, so this is fine.
      return try await withTaskCancellationHandler {
        let generation = try await task.value
        try Task.checkCancellation()
        return generation
      } onCancel: {
        self.stop()
        task.cancel()
      }
    }
  }

  public var isGenerating: Bool {
    if case .generating = self.status { return true }
    return false
  }

  public var status: Status {
    self.registrar.access(self, keyPath: \.status)
    return self.state.withLock { $0.status }
  }

  public var toolCalls: NeedleToolCallCollection {
    self.registrar.access(self, keyPath: \.toolCalls)
    return self.state.withLock { $0.toolCalls }
  }

  private let state = RecursiveLock(State())
  private let registrar = ObservationRegistrar()

  deinit {
    self.state.withLock { state in
      state.task?.cancel()
    }
  }

  fileprivate init<Engine: NeedleEngine>(
    session: NeedleSession<Engine>,
    tools: [any NeedleTool],
    with prompt: String,
    systemPrompt: String,
    parameters: sending Engine.GenerateParameters,
    shouldInvokeTools: Bool
  ) {
    let task = Task<NeedleSessionGeneration, any Error> {
      do {
        let prompt = NeedlePrompt(
          system: systemPrompt,
          user: prompt,
          tools: tools.map(\.definition)
        )
        let generation: NeedleEngineGeneration = try session.withEngine { engine in
          engine.reset()
          let shouldGenerate = self.state.withLock { state in
            if state.wasStoppedBeforeGeneration {
              self.registrar.withMutation(of: self, keyPath: \.status) {
                state.beginGenerating(stopper: nil)
              }
              return false
            }
            self.registrar.withMutation(of: self, keyPath: \.status) {
              state.beginGenerating(stopper: engine.stopper)
            }
            return true
          }
          guard shouldGenerate else { return .empty }

          var parseState = ToolCallParseState(tools: tools)
          return try engine.generate(prompt: prompt, parameters: parameters) { token in
            self.state.withLock { state in
              state.emitToken(token: token)
              if let call = parseState.accept(token: token) {
                self.registrar.withMutation(of: self, keyPath: \.toolCalls) {
                  state.emitToolCall(call, shouldInvoke: shouldInvokeTools)
                }
              }
            }
          }
        }
        let sessionGeneration = self.state.withLock { state in
          let sessionGeneration = NeedleSessionGeneration(
            engineGeneration: generation,
            toolCalls: state.toolCalls
          )
          self.registrar.withMutation(of: self, keyPath: \.status) {
            state.finish(with: .success(sessionGeneration))
          }
          return sessionGeneration
        }
        session.removeActiveStream(self)
        return sessionGeneration
      } catch {
        self.state.withLock { state in
          self.registrar.withMutation(of: self, keyPath: \.status) {
            state.finish(with: .failure(error))
          }
        }
        session.removeActiveStream(self)
        throw error
      }
    }
    self.state.withLock { $0.task = task }
  }

  public func stop() {
    let stopper = self.state.withLock { $0.stop() }
    stopper?()
  }
}

extension NeedleSessionStream: AsyncSequence {
  public struct AsyncIterator: AsyncIteratorProtocol {
    fileprivate var base: AsyncThrowingStream<AnyNeedleToolCall, any Error>.AsyncIterator

    public mutating func next() async throws -> NeedleToolCallCollection.Element? {
      try await self.base.next()
    }

    @available(iOS 18.0, macOS 15.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public mutating func next(
      isolation actor: isolated (any Actor)?
    ) async throws -> NeedleToolCallCollection.Element? {
      try await self.base.next(isolation: actor)
    }
  }

  public func makeAsyncIterator() -> AsyncIterator {
    let (stream, continuation, id) = self.state.withLock { $0.takeToolCallStream() }
    continuation.onTermination = { [weak self] _ in
      self?.state.withLock { $0.removeTokenContinuation(id: id) }
    }
    return AsyncIterator(base: stream.makeAsyncIterator())
  }
}

extension NeedleSessionStream {
  public struct Tokens: AsyncSequence {
    fileprivate let stream: NeedleSessionStream

    public struct AsyncIterator: AsyncIteratorProtocol {
      fileprivate var base: AsyncThrowingStream<NeedleToken, any Error>.AsyncIterator

      public mutating func next() async throws -> NeedleToken? {
        try await self.base.next()
      }

      @available(iOS 18.0, macOS 15.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
      public mutating func next(
        isolation actor: isolated (any Actor)?
      ) async throws -> NeedleToken? {
        try await self.base.next(isolation: actor)
      }
    }

    public func makeAsyncIterator() -> AsyncIterator {
      let (tokenStream, continuation, id) = self.stream.state.withLock { $0.takeTokenStream() }
      continuation.onTermination = { [weak stream] _ in
        stream?.state.withLock { $0.removeToolContinuation(id: id) }
      }
      return AsyncIterator(base: tokenStream.makeAsyncIterator())
    }
  }

  public var tokens: Tokens {
    Tokens(stream: self)
  }
}

extension NeedleSessionStream {
  private struct State {
    private(set) var status: NeedleSessionStream.Status = .awaitingExecution
    private(set) var stopper: NeedleEngineStopper?
    private(set) var wasStoppedBeforeGeneration = false
    private var tokenContinuations =
      [Int: AsyncThrowingStream<NeedleToken, any Error>.Continuation]()
    private var toolsContinuations =
      [Int: AsyncThrowingStream<AnyNeedleToolCall, any Error>.Continuation]()
    private var nextTokenContinuationID = 0
    private var nextToolContinuationID = 0
    private(set) var toolCalls = NeedleToolCallCollection()
    var task: Task<NeedleSessionGeneration, any Error>?

    mutating func beginGenerating(stopper: NeedleEngineStopper?) {
      if case .awaitingExecution = self.status {
        self.stopper = stopper
        self.status = .generating
      }
    }

    mutating func stop() -> NeedleEngineStopper? {
      switch self.status {
      case .awaitingExecution:
        self.wasStoppedBeforeGeneration = true
        return nil
      case .generating:
        return self.stopper
      case .finished:
        return nil
      }
    }

    mutating func takeTokenStream() -> (
      AsyncThrowingStream<NeedleToken, any Error>,
      AsyncThrowingStream<NeedleToken, any Error>.Continuation,
      Int
    ) {
      let (stream, continuation) = AsyncThrowingStream<NeedleToken, any Error>.makeStream()
      let id = self.nextTokenContinuationID
      self.nextTokenContinuationID += 1
      self.tokenContinuations[id] = continuation
      return (stream, continuation, id)
    }

    mutating func takeToolCallStream() -> (
      AsyncThrowingStream<AnyNeedleToolCall, any Error>,
      AsyncThrowingStream<AnyNeedleToolCall, any Error>.Continuation,
      Int
    ) {
      let (stream, continuation) = AsyncThrowingStream<AnyNeedleToolCall, any Error>.makeStream()
      let id = self.nextToolContinuationID
      self.nextToolContinuationID += 1
      self.toolsContinuations[id] = continuation
      for call in self.toolCalls {
        continuation.yield(call)
      }
      return (stream, continuation, id)
    }

    mutating func removeTokenContinuation(id: Int) {
      self.tokenContinuations.removeValue(forKey: id)
    }

    mutating func removeToolContinuation(id: Int) {
      self.toolsContinuations.removeValue(forKey: id)
    }

    func emitToken(token: NeedleToken) {
      for continuation in self.tokenContinuations.values {
        continuation.yield(token)
      }
    }

    mutating func emitToolCall(_ toolCall: AnyNeedleToolCall, shouldInvoke: Bool) {
      self.toolCalls.append(toolCall)
      if shouldInvoke {
        Task { _ = try await toolCall.invoke() }
      }
      for continuation in self.toolsContinuations.values {
        continuation.yield(toolCall)
      }
    }

    mutating func finish(with result: Result<NeedleSessionGeneration, any Error>) {
      self.stopper = nil
      self.status = .finished(result)
      switch result {
      case .success:
        for continuation in self.tokenContinuations.values {
          continuation.finish()
        }
        for continuation in self.toolsContinuations.values {
          continuation.finish()
        }
      case .failure(let error):
        for continuation in self.tokenContinuations.values {
          continuation.finish(throwing: error)
        }
        for continuation in self.toolsContinuations.values {
          continuation.finish(throwing: error)
        }
      }
      self.tokenContinuations.removeAll()
      self.toolsContinuations.removeAll()
    }
  }
}

extension NeedleSession {
  public func stream(
    tools: [any NeedleTool],
    with prompt: String,
    systemPromptOverride: String? = nil,
    parameters: sending Engine.GenerateParameters = .default,
    shouldInvokeTools: Bool = true
  ) -> NeedleSessionStream {
    let stream = NeedleSessionStream(
      session: self,
      tools: tools,
      with: prompt,
      systemPrompt: systemPromptOverride ?? self.systemPrompt,
      parameters: parameters,
      shouldInvokeTools: shouldInvokeTools
    )
    self.observationRegistrar.withMutation(of: self, keyPath: \.activeStreams) {
      self._activeStreams.withLock { $0.append(stream) }
    }
    return stream
  }
}

// MARK: - Parse State

private struct ToolCallParseState {
  private enum Phase {
    case outsideBlock
    case insideArray
  }

  private static let opener = "<tool_call>"

  private let toolsByName: [String: any NeedleTool]
  private var phase = Phase.outsideBlock
  private var buffer = ""
  private var hasSeenArrayOpen = false
  private var braceDepth = 0
  private var currentObjectStart: String.Index?
  private var scanIndex: String.Index?

  init(tools: [any NeedleTool]) {
    var byName = [String: any NeedleTool]()
    for tool in tools {
      byName[tool.name.snakeCased()] = tool
    }
    self.toolsByName = byName
  }

  mutating func accept(token: NeedleToken) -> AnyNeedleToolCall? {
    self.buffer.append(token.stringValue)
    switch self.phase {
    case .outsideBlock:
      return self.handleOutsideBlock()
    case .insideArray:
      return self.handleInsideArray()
    }
  }

  private mutating func handleOutsideBlock() -> AnyNeedleToolCall? {
    guard let range = self.buffer.range(of: Self.opener) else { return nil }
    self.buffer.removeSubrange(..<range.upperBound)
    self.phase = .insideArray
    return self.handleInsideArray()
  }

  private mutating func handleInsideArray() -> AnyNeedleToolCall? {
    guard self.consumeArrayOpenBracketIfNeeded() else { return nil }

    var scanIndex = self.scanIndex ?? self.buffer.startIndex
    while scanIndex < self.buffer.endIndex {
      guard
        let boundaryIndex = self.buffer[scanIndex...].firstIndex(where: \.isNeedleBoundary)
      else {
        self.scanIndex = self.buffer.endIndex
        return nil
      }
      let nextIndex = self.buffer.index(after: boundaryIndex)

      switch self.buffer[boundaryIndex] {
      case "{":
        self.openBrace(at: boundaryIndex)
        scanIndex = nextIndex

      case "}":
        if let call = self.closeBrace(at: nextIndex) {
          return call
        }
        scanIndex = nextIndex

      case "]":
        if self.braceDepth == 0 {
          self.buffer.removeSubrange(..<nextIndex)
          return nil
        }
        scanIndex = nextIndex

      default:
        break
      }
    }

    self.scanIndex = scanIndex
    return nil
  }

  private mutating func consumeArrayOpenBracketIfNeeded() -> Bool {
    guard !self.hasSeenArrayOpen else { return true }
    guard let bracketIndex = self.buffer.firstIndex(of: "[") else { return false }
    self.buffer.removeSubrange(..<bracketIndex)
    self.hasSeenArrayOpen = true
    self.scanIndex = self.buffer.startIndex
    return true
  }

  private mutating func openBrace(at index: String.Index) {
    if self.braceDepth == 0 {
      self.currentObjectStart = index
    }
    self.braceDepth += 1
  }

  private mutating func closeBrace(at nextIndex: String.Index) -> AnyNeedleToolCall? {
    guard self.braceDepth > 0 else { return nil }
    self.braceDepth -= 1
    guard self.braceDepth == 0, let start = self.currentObjectStart else { return nil }

    let objectString = String(self.buffer[start..<nextIndex])
    self.buffer.removeSubrange(..<nextIndex)
    self.currentObjectStart = nil
    self.scanIndex = self.buffer.startIndex
    return self.parseAndBuild(objectString: objectString)
  }

  private func parseAndBuild(objectString: String) -> AnyNeedleToolCall? {
    guard let data = objectString.data(using: .utf8) else { return nil }
    guard let value = try? JSONDecoder().decode(ParsedToolCall.self, from: data) else {
      return nil
    }
    guard let tool = self.toolsByName[value.name] else { return nil }

    func open<Tool: NeedleTool>(_ concrete: Tool) -> AnyNeedleToolCall? {
      guard let typed = try? Tool.Input(needleValue: value.arguments) else { return nil }
      return AnyNeedleToolCall(NeedleToolCall(id: NeedleToolCallID(), tool: concrete, input: typed))
    }
    return open(tool)
  }
}

extension Character {
  fileprivate var isNeedleBoundary: Bool {
    ["{", "}", "]"].contains(self)
  }
}

private struct ParsedToolCall: Codable {
  let name: String
  let arguments: NeedleValue
}
