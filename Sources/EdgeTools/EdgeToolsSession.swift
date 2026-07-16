import Foundation
import Observation

// MARK: - EdgeToolsSession

public final class EdgeToolsSession<Engine: EdgeToolsEngine>: Sendable, Observable {
  private struct ActiveStreams: Sendable {
    var rawToolCallStreams = [EdgeToolsRawToolCallsStream]()
    var toolCallStreams = [EdgeToolsSessionStream]()
  }

  public let engine: Engine
  private let _activeStreams = Lock(ActiveStreams())
  private let observationRegistrar = ObservationRegistrar()

  public var isResponding: Bool {
    !self.activeStreams.isEmpty
  }

  public var activeStreams: EdgeToolsSessionActiveStreams {
    self.observationRegistrar.access(self, keyPath: \.activeStreams)
    return self._activeStreams.withLock {
      EdgeToolsSessionActiveStreams(
        rawToolCallStreams: $0.rawToolCallStreams,
        toolCallStreams: $0.toolCallStreams
      )
    }
  }

  public init(engine: sending Engine) {
    self.engine = engine
  }

  fileprivate func registerActiveStream(_ stream: EdgeToolsRawToolCallsStream) {
    self._activeStreams.withLock { activeStreams in
      self.observationRegistrar.withMutation(of: self, keyPath: \.activeStreams) {
        activeStreams.rawToolCallStreams.append(stream)
      }
    }
  }

  fileprivate func registerActiveStream(_ stream: EdgeToolsSessionStream) {
    self._activeStreams.withLock { activeStreams in
      self.observationRegistrar.withMutation(of: self, keyPath: \.activeStreams) {
        activeStreams.toolCallStreams.append(stream)
      }
    }
  }

  fileprivate func removeActiveStream(_ stream: EdgeToolsRawToolCallsStream) {
    self._activeStreams.withLock { activeStreams in
      self.observationRegistrar.withMutation(of: self, keyPath: \.activeStreams) {
        activeStreams.rawToolCallStreams.removeAll { $0 === stream }
      }
    }
  }

  fileprivate func removeActiveStream(_ stream: EdgeToolsSessionStream) {
    self._activeStreams.withLock { activeStreams in
      self.observationRegistrar.withMutation(of: self, keyPath: \.activeStreams) {
        activeStreams.toolCallStreams.removeAll { $0 === stream }
      }
    }
  }
}

extension EdgeToolsSession {
  public func tokenize(
    prompt: Engine.Prompt,
    tools: [EdgeToolDefinition]
  ) async throws -> [EdgeToolsToken] {
    try await self.engine.tokenize(prompt: prompt, tools: tools)
  }
}

extension EdgeToolsSession where Engine: EdgeToolsPrefillableEngine {
  public func prefill(
    promptPrefix: Engine.Prompt,
    tools: [EdgeToolDefinition]
  ) async throws -> EdgeToolsEnginePrefill {
    try await self.engine.prefill(promptPrefix: promptPrefix, tools: tools)
  }
}

public struct EdgeToolsSessionGeneration: Sendable {
  public let engineGeneration: EdgeToolsEngineGeneration
  public let toolCalls: EdgeToolCallCollection
}

public struct EdgeToolsRawToolCallsGeneration: Sendable {
  public let engineGeneration: EdgeToolsEngineGeneration
  public let toolCalls: [EdgeRawToolCall]
}

public struct EdgeToolsSessionActiveStreams: Sendable {
  public let rawToolCallStreams: [EdgeToolsRawToolCallsStream]
  public let toolCallStreams: [EdgeToolsSessionStream]

  public var isEmpty: Bool {
    self.rawToolCallStreams.isEmpty
  }
}

// MARK: - Tokens

public struct EdgeToolsSessionTokens: AsyncSequence, Sendable {
  public typealias Element = EdgeToolsToken

  public struct AsyncIterator: AsyncIteratorProtocol {
    fileprivate var base: AsyncThrowingStream<EdgeToolsToken, any Error>.AsyncIterator

    public mutating func next() async throws -> EdgeToolsToken? {
      try await self.base.next()
    }
  }

  fileprivate let makeIterator: @Sendable () -> AsyncIterator

  public func makeAsyncIterator() -> AsyncIterator { self.makeIterator() }
}

// MARK: - Raw Tool Calls

public final class EdgeToolsRawToolCallsStream: Sendable, Observable, AsyncSequence {
  public typealias Element = EdgeRawToolCall

  public struct AsyncIterator: AsyncIteratorProtocol {
    fileprivate var base: AsyncThrowingStream<EdgeRawToolCall, any Error>.AsyncIterator
    public mutating func next() async throws -> EdgeRawToolCall? { try await self.base.next() }
  }

  private struct State {
    var task: Task<EdgeToolsRawToolCallsGeneration, any Error>?
    var hasStarted = false
    var result: Result<EdgeToolsRawToolCallsGeneration, any Error>?
    var wasStoppedBeforeGeneration = false
    var stop: (@Sendable () -> Void)?
    var calls = [EdgeRawToolCall]()
    var tokens = [EdgeToolsToken]()
    var callContinuations = [Int: AsyncThrowingStream<EdgeRawToolCall, any Error>.Continuation]()
    var tokenContinuations = [Int: AsyncThrowingStream<EdgeToolsToken, any Error>.Continuation]()
    var onToolCall: (@Sendable (EdgeRawToolCall, Int) -> Void)?
    var onResult: (@Sendable (Result<EdgeToolsRawToolCallsGeneration, any Error>) -> Void)?
    var onFinish: (@Sendable () -> Void)?
    var nextID = 0
  }

  private let state = Lock(State())
  private let registrar = ObservationRegistrar()
  private let startGeneration:
    @Sendable (
      EdgeToolsRawToolCallsStream
    ) -> Task<EdgeToolsRawToolCallsGeneration, any Error>

  public var rawToolCalls: [EdgeRawToolCall] {
    self.registrar.access(self, keyPath: \.rawToolCalls)
    return self.state.withLock { $0.calls }
  }

  public var result: Result<EdgeToolsRawToolCallsGeneration, any Error>? {
    self.registrar.access(self, keyPath: \.result)
    return self.state.withLock { $0.result }
  }

  public var tokens: EdgeToolsSessionTokens {
    EdgeToolsSessionTokens(makeIterator: { [weak self] in self!.makeTokenIterator() })
  }

  public var finalGeneration: EdgeToolsRawToolCallsGeneration {
    get async throws {
      let task = self.state.withLock { $0.task! }
      return try await withTaskCancellationHandler {
        try await task.value
      } onCancel: {
        self.stop()
        task.cancel()
      }
    }
  }

  fileprivate init<Engine: EdgeToolsEngine>(
    session: EdgeToolsSession<Engine>,
    prompt: Engine.Prompt,
    tools: [EdgeToolDefinition],
    parameters: Engine.GenerateParameters,
    onToolCall: (@Sendable (EdgeRawToolCall, Int) -> Void)? = nil,
    onResult: (@Sendable (Result<EdgeToolsRawToolCallsGeneration, any Error>) -> Void)? = nil
  ) {
    self.startGeneration = { stream in
      Task {
        try await stream.runGeneration(
          session: session,
          prompt: prompt,
          tools: tools,
          parameters: parameters
        )
      }
    }
    self.state.withLock {
      $0.onToolCall = onToolCall
      $0.onResult = onResult
    }
  }

  fileprivate func start() {
    let shouldStart = self.state.withLock { state in
      guard !state.hasStarted else { return false }
      state.hasStarted = true
      return true
    }
    guard shouldStart else { return }
    let task = self.startGeneration(self)
    self.state.withLock { $0.task = task }
  }

  fileprivate func setOnFinish(_ onFinish: @escaping @Sendable () -> Void) {
    self.state.withLock { $0.onFinish = onFinish }
  }

  public func makeAsyncIterator() -> AsyncIterator {
    let (stream, continuation) = AsyncThrowingStream<EdgeRawToolCall, any Error>.makeStream()
    let (id, calls, result) = self.state.withLock { state in
      let id = state.nextID
      state.nextID += 1
      guard state.result == nil else { return (id, state.calls, state.result) }
      state.callContinuations[id] = continuation
      return (id, state.calls, nil)
    }
    calls.forEach { continuation.yield($0) }
    if let result {
      switch result {
      case .success: continuation.finish()
      case .failure(let error): continuation.finish(throwing: error)
      }
    } else {
      continuation.onTermination = { [weak self] reason in
        guard case .cancelled = reason else { return }
        self?.state.withLock { _ = $0.callContinuations.removeValue(forKey: id) }
      }
    }
    return AsyncIterator(base: stream.makeAsyncIterator())
  }

  public func stop() {
    let stop: (@Sendable () -> Void)? = self.state.withLock { state in
      guard state.result == nil else { return nil }
      guard let stop = state.stop else {
        state.wasStoppedBeforeGeneration = true
        return nil
      }
      return stop
    }
    stop?()
  }

  private func makeTokenIterator() -> EdgeToolsSessionTokens.AsyncIterator {
    let (stream, continuation) = AsyncThrowingStream<EdgeToolsToken, any Error>.makeStream()
    let (id, tokens, result) = self.state.withLock { state in
      let id = state.nextID
      state.nextID += 1
      guard state.result == nil else { return (id, state.tokens, state.result) }
      state.tokenContinuations[id] = continuation
      return (id, state.tokens, nil)
    }
    tokens.forEach { continuation.yield($0) }
    if let result {
      switch result {
      case .success: continuation.finish()
      case .failure(let error): continuation.finish(throwing: error)
      }
    } else {
      continuation.onTermination = { [weak self] reason in
        guard case .cancelled = reason else { return }
        self?.state.withLock { _ = $0.tokenContinuations.removeValue(forKey: id) }
      }
    }
    return EdgeToolsSessionTokens.AsyncIterator(base: stream.makeAsyncIterator())
  }

  private func runGeneration<Engine: EdgeToolsEngine>(
    session: EdgeToolsSession<Engine>,
    prompt: Engine.Prompt,
    tools: [EdgeToolDefinition],
    parameters: Engine.GenerateParameters
  ) async throws -> EdgeToolsRawToolCallsGeneration {
    let wasStopped = self.state.withLock { $0.wasStoppedBeforeGeneration }
    if wasStopped {
      let generation = EdgeToolsRawToolCallsGeneration(
        engineGeneration: .empty,
        toolCalls: []
      )
      self.finish(with: .success(generation))
      return generation
    }

    let channel = EdgeToolsGenerationChannel(
      onToken: { [weak self] token in self?.emit(token: token) },
      onToolCall: { [weak self] call in self?.emit(call: call) }
    )
    do {
      let generationTask = try session.engine.generate(
        prompt: prompt,
        tools: tools,
        parameters: parameters,
        channel: channel
      )
      let shouldStop = self.state.withLock { state in
        state.stop = { generationTask.stop() }
        return state.wasStoppedBeforeGeneration
      }
      if shouldStop { generationTask.stop() }
      let engineGeneration = try await generationTask.value
      try Task.checkCancellation()
      let generation = EdgeToolsRawToolCallsGeneration(
        engineGeneration: engineGeneration,
        toolCalls: self.rawToolCalls
      )
      self.finish(with: .success(generation))
      return generation
    } catch {
      self.finish(with: .failure(error))
      throw error
    }
  }

  private func emit(token: EdgeToolsToken) {
    let continuations = self.state.withLock { state in
      state.tokens.append(token)
      return Array(state.tokenContinuations.values)
    }
    continuations.forEach { $0.yield(token) }
  }

  private func emit(call: EdgeRawToolCall) {
    let emission = self.registrar.withMutation(of: self, keyPath: \.rawToolCalls) {
      self.state.withLock { state in
        let index = state.calls.count
        state.calls.append(call)
        return (index, state.onToolCall, Array(state.callContinuations.values))
      }
    }
    emission.1?(call, emission.0)
    emission.2.forEach { $0.yield(call) }
  }

  private func finish(with result: Result<EdgeToolsRawToolCallsGeneration, any Error>) {
    let completion = self.registrar.withMutation(of: self, keyPath: \.result) {
      self.state.withLock { state in
        state.result = result
        let completion = (
          state.onResult,
          state.onFinish,
          Array(state.callContinuations.values),
          Array(state.tokenContinuations.values)
        )
        state.callContinuations.removeAll()
        state.tokenContinuations.removeAll()
        state.stop = nil
        return completion
      }
    }
    completion.0?(result)
    completion.1?()
    switch result {
    case .success:
      completion.2.forEach { $0.finish() }
      completion.3.forEach { $0.finish() }
    case .failure(let error):
      completion.2.forEach { $0.finish(throwing: error) }
      completion.3.forEach { $0.finish(throwing: error) }
    }
  }
}

// MARK: - Typed Tool Calls

public final class EdgeToolsSessionStream: Sendable, Observable, Identifiable, AsyncSequence {
  public typealias Element = EdgeToolCallCollection.Element

  public struct AsyncIterator: AsyncIteratorProtocol {
    fileprivate var base: EdgeToolsRawToolCallsStream.AsyncIterator
    fileprivate let stream: EdgeToolsSessionStream
    private var rawIndex = 0

    fileprivate init(
      base: EdgeToolsRawToolCallsStream.AsyncIterator,
      stream: EdgeToolsSessionStream
    ) {
      self.base = base
      self.stream = stream
    }

    public mutating func next() async throws -> Element? {
      while try await self.base.next() != nil {
        defer { self.rawIndex += 1 }
        if let call = self.stream.call(at: self.rawIndex) {
          return call
        }
      }
      return nil
    }
  }

  fileprivate let rawStream: EdgeToolsRawToolCallsStream
  private let storage: Storage

  public var isGenerating: Bool {
    self.result == nil
  }

  public var isFinished: Bool {
    self.result != nil
  }

  public var tokens: EdgeToolsSessionTokens {
    self.rawStream.tokens
  }

  public var toolCalls: EdgeToolCallCollection {
    self.storage.toolCalls
  }

  public var result: Result<EdgeToolsSessionGeneration, any Error>? {
    self.storage.result
  }

  public var finalGeneration: EdgeToolsSessionGeneration {
    get async throws {
      _ = try await self.rawStream.finalGeneration
      return try self.result!.get()
    }
  }

  fileprivate init<Engine: EdgeToolsEngine>(
    session: EdgeToolsSession<Engine>,
    prompt: Engine.Prompt,
    tools: [any EdgeTool],
    parameters: Engine.GenerateParameters,
    shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool
  ) {
    let storage = Storage()
    let toolsByName = Dictionary(uniqueKeysWithValues: tools.map { ($0.name.snakeCased(), $0) })
    self.storage = storage
    self.rawStream = EdgeToolsRawToolCallsStream(
      session: session,
      prompt: prompt,
      tools: tools.map(\.definition),
      parameters: parameters,
      onToolCall: { call, index in
        storage.resolve(
          call,
          at: index,
          toolsByName: toolsByName,
          shouldInvokeTools: shouldInvokeTools
        )
      },
      onResult: { result in storage.finish(result) }
    )
  }

  fileprivate func setOnFinish(_ onFinish: @escaping @Sendable () -> Void) {
    self.storage.setOnFinish(onFinish)
  }

  fileprivate func start() {
    self.rawStream.start()
  }

  public func makeAsyncIterator() -> AsyncIterator {
    AsyncIterator(base: self.rawStream.makeAsyncIterator(), stream: self)
  }

  public func stop() {
    self.rawStream.stop()
  }

  private func call(at index: Int) -> AnyEdgeToolCall? {
    self.storage.call(at: index)
  }

  private final class Storage: Sendable, Observable {
    private struct State {
      var callsByRawIndex = [AnyEdgeToolCall?]()
      var toolCalls = EdgeToolCallCollection()
      var result: Result<EdgeToolsSessionGeneration, any Error>?
    }

    private let state = Lock(State())
    private let registrar = ObservationRegistrar()
    private let onFinish = Lock<(@Sendable () -> Void)?>(nil)

    var toolCalls: EdgeToolCallCollection {
      self.registrar.access(self, keyPath: \.toolCalls)
      return self.state.withLock { $0.toolCalls }
    }

    var result: Result<EdgeToolsSessionGeneration, any Error>? {
      self.registrar.access(self, keyPath: \.result)
      return self.state.withLock { $0.result }
    }

    func call(at index: Int) -> AnyEdgeToolCall? {
      self.state.withLock { $0.callsByRawIndex[index] }
    }

    func resolve(
      _ rawCall: EdgeRawToolCall,
      at index: Int,
      toolsByName: [String: any EdgeTool],
      shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool
    ) {
      let call = Self.resolve(rawCall, toolsByName: toolsByName)
      self.state.withLock { state in
        guard state.callsByRawIndex.count == index else { return }
        state.callsByRawIndex.append(call)
        guard let call else { return }
        self.registrar.withMutation(of: self, keyPath: \.toolCalls) {
          state.toolCalls.append(call)
        }
        if shouldInvokeTools(call) {
          _ = Task { _ = try await call.output }
        }
      }
    }

    func finish(_ rawResult: Result<EdgeToolsRawToolCallsGeneration, any Error>) {
      let result = rawResult.map {
        EdgeToolsSessionGeneration(engineGeneration: $0.engineGeneration, toolCalls: self.toolCalls)
      }
      let shouldFinish = self.state.withLock { state in
        guard state.result == nil else { return false }
        self.registrar.withMutation(of: self, keyPath: \.result) { state.result = result }
        return true
      }
      if shouldFinish { self.onFinish.withLock { $0 }?() }
    }

    func setOnFinish(_ onFinish: @escaping @Sendable () -> Void) {
      self.onFinish.withLock { $0 = onFinish }
    }

    private static func resolve(
      _ rawCall: EdgeRawToolCall,
      toolsByName: [String: any EdgeTool]
    ) -> AnyEdgeToolCall? {
      guard let tool = toolsByName[rawCall.name.snakeCased()] else { return nil }
      func open<Tool: EdgeTool>(_ tool: Tool) -> AnyEdgeToolCall? {
        guard
          let call = try? EdgeToolCall(
            id: EdgeToolCallID(),
            tool: tool,
            rawInput: rawCall.arguments
          )
        else { return nil }
        return AnyEdgeToolCall(call)
      }
      return open(tool)
    }
  }
}

extension EdgeToolsSession {
  public func streamRawToolCalls(
    prompt: Engine.Prompt,
    tools: [EdgeToolDefinition],
    parameters: Engine.GenerateParameters = .default
  ) -> EdgeToolsRawToolCallsStream {
    if let message = duplicateToolNameError(tools.map(\.name)) {
      assertionFailure(message)
    }
    let stream = EdgeToolsRawToolCallsStream(
      session: self,
      prompt: prompt,
      tools: tools,
      parameters: parameters
    )
    self.registerActiveStream(stream)
    stream.setOnFinish { [weak self, weak stream] in
      guard let self, let stream else { return }
      self.removeActiveStream(stream)
    }
    stream.start()
    return stream
  }

  public func generateRawToolCalls(
    prompt: Engine.Prompt,
    tools: [EdgeToolDefinition],
    parameters: Engine.GenerateParameters = .default
  ) async throws -> EdgeToolsRawToolCallsGeneration {
    try await self.streamRawToolCalls(
      prompt: prompt,
      tools: tools,
      parameters: parameters
    )
    .finalGeneration
  }

  public func stream(
    prompt: Engine.Prompt,
    tools: [any EdgeTool],
    parameters: Engine.GenerateParameters = .default,
    shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool = { _ in true }
  ) -> EdgeToolsSessionStream {
    if let message = duplicateToolNameError(tools.map(\.name)) {
      assertionFailure(message)
    }
    let stream = EdgeToolsSessionStream(
      session: self,
      prompt: prompt,
      tools: tools,
      parameters: parameters,
      shouldInvokeTools: shouldInvokeTools
    )
    self.registerActiveStream(stream.rawStream)
    self.registerActiveStream(stream)
    stream.rawStream.setOnFinish { [weak self, weak rawStream = stream.rawStream] in
      guard let self, let rawStream else { return }
      self.removeActiveStream(rawStream)
    }
    stream.setOnFinish { [weak self, weak stream] in
      guard let self, let stream else { return }
      self.removeActiveStream(stream)
    }
    stream.start()
    return stream
  }

  @concurrent
  public func generate(
    prompt: Engine.Prompt,
    tools: [any EdgeTool],
    parameters: Engine.GenerateParameters = .default,
    shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool = { _ in true }
  ) async throws -> EdgeToolsSessionGeneration {
    try await self.stream(
      prompt: prompt,
      tools: tools,
      parameters: parameters,
      shouldInvokeTools: shouldInvokeTools
    )
    .finalGeneration
  }
}

// MARK: - Duplicate Tool Name Error

package func duplicateToolNameError(_ names: some Sequence<String>) -> String? {
  var grouped = [String: [String]]()
  for name in names { grouped[name.snakeCased(), default: []].append(name) }
  let duplicates = grouped.filter { $0.value.count > 1 }
  guard !duplicates.isEmpty else { return nil }
  return duplicates.sorted { $0.key < $1.key }
    .map { normalized, originals in
      "The names \(originals.sorted().map { "'\($0)'" }.joined(separator: " and ")) all normalize to '\(normalized)'."
    }
    .joined(separator: "\n")
}
