import Observation

// MARK: - EdgeToolsSession

public final class EdgeToolsSession<Engine: EdgeToolsEngine>: Sendable, Observable {
  public let engine: Engine
  private let _tools: Lock<[any EdgeTool]>
  private let _activeStreams = Lock([EdgeToolsSessionStream]())
  private let observationRegistrar = ObservationRegistrar()

  public var tools: [any EdgeTool] {
    get {
      self.observationRegistrar.access(self, keyPath: \.tools)
      return self._tools.withLock { $0 }
    }
    set {
      self._tools.withLock { tools in
        self.observationRegistrar.withMutation(of: self, keyPath: \.tools) {
          tools = newValue
        }
      }
    }
  }

  public var isResponding: Bool {
    !self.activeStreams.isEmpty
  }

  public var activeStreams: [EdgeToolsSessionStream] {
    self.observationRegistrar.access(self, keyPath: \.activeStreams)
    return self._activeStreams.withLock { $0 }
  }

  public init(engine: sending Engine, tools: [any EdgeTool] = []) {
    self.engine = engine
    self._tools = Lock(tools)
  }

  fileprivate func registerActiveStream(_ stream: EdgeToolsSessionStream) {
    self._activeStreams.withLock { streams in
      self.observationRegistrar.withMutation(of: self, keyPath: \.activeStreams) {
        streams.append(stream)
      }
    }
  }

  fileprivate func removeActiveStream(_ stream: EdgeToolsSessionStream) {
    self._activeStreams.withLock { streams in
      self.observationRegistrar.withMutation(of: self, keyPath: \.activeStreams) {
        streams.removeAll { $0 === stream }
      }
    }
  }
}

extension EdgeToolsSession {
  public func tokenize(prompt: Engine.Prompt) async throws -> [EdgeToolsToken] {
    let toolDefinitions = self.tools.map(\.definition)
    return try await self.engine.tokenize(prompt: prompt, tools: toolDefinitions)
  }
}

extension EdgeToolsSession where Engine: EdgeToolsPrefillableEngine {
  public func prefill(promptPrefix: Engine.Prompt) async throws -> EdgeToolsEnginePrefill {
    let toolDefinitions = self.tools.map(\.definition)
    return try await self.engine.prefill(promptPrefix: promptPrefix, tools: toolDefinitions)
  }
}

// MARK: - Generation

public struct EdgeToolsSessionGeneration: Sendable {
  public let engineGeneration: EdgeToolsEngineGeneration
  public let toolCalls: EdgeToolCallCollection

  public var response: String {
    self.engineGeneration.response
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

  public func makeAsyncIterator() -> AsyncIterator {
    self.makeIterator()
  }
}

// MARK: - Stream

public final class EdgeToolsSessionStream: Sendable, Observable, Identifiable, AsyncSequence {
  public typealias Element = EdgeToolCallCollection.Element

  public struct AsyncIterator: AsyncIteratorProtocol {
    fileprivate var base: AsyncThrowingStream<Element, any Error>.AsyncIterator

    public mutating func next() async throws -> Element? {
      try await self.base.next()
    }
  }

  private struct State {
    var task: Task<EdgeToolsSessionGeneration, any Error>?
    var hasStarted = false
    var result: Result<EdgeToolsSessionGeneration, any Error>?
    var wasStoppedBeforeGeneration = false
    var stop: (@Sendable () -> Void)?
    var toolCalls = EdgeToolCallCollection()
    var tokens = [EdgeToolsToken]()
    var callContinuations = [Int: AsyncThrowingStream<Element, any Error>.Continuation]()
    var tokenContinuations = [Int: AsyncThrowingStream<EdgeToolsToken, any Error>.Continuation]()
    var onFinish: (@Sendable () -> Void)?
    var nextID = 0
  }

  private let state = Lock(State())
  private let registrar = ObservationRegistrar()
  private let toolsByName: [String: any EdgeTool]
  private let shouldInvokeTools: @Sendable (AnyEdgeToolCall) -> Bool
  private let startGeneration:
    @Sendable (EdgeToolsSessionStream) -> Task<EdgeToolsSessionGeneration, any Error>

  public var isGenerating: Bool {
    self.result == nil
  }

  public var isFinished: Bool {
    self.result != nil
  }

  public var tokens: EdgeToolsSessionTokens {
    EdgeToolsSessionTokens(makeIterator: { [weak self] in self!.makeTokenIterator() })
  }

  public var toolCalls: EdgeToolCallCollection {
    self.registrar.access(self, keyPath: \.toolCalls)
    return self.state.withLock { $0.toolCalls }
  }

  public var result: Result<EdgeToolsSessionGeneration, any Error>? {
    self.registrar.access(self, keyPath: \.result)
    return self.state.withLock { $0.result }
  }

  public var response: Result<String, any Error>? {
    self.result.map { $0.map(\.response) }
  }

  public var finalGeneration: EdgeToolsSessionGeneration {
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
    tools: [any EdgeTool],
    parameters: Engine.GenerateParameters,
    shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool
  ) {
    let toolDefinitions = tools.map(\.definition)
    self.toolsByName = Dictionary(uniqueKeysWithValues: tools.map { ($0.name.snakeCased(), $0) })
    self.shouldInvokeTools = shouldInvokeTools
    self.startGeneration = { stream in
      Task {
        try await stream.runGeneration(
          session: session,
          prompt: prompt,
          toolDefinitions: toolDefinitions,
          parameters: parameters
        )
      }
    }
  }
}

extension EdgeToolsSessionStream {
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
    let (stream, continuation) = AsyncThrowingStream<Element, any Error>.makeStream()
    let (id, calls, result) = self.state.withLock { state in
      let id = state.nextID
      state.nextID += 1
      guard state.result == nil else { return (id, state.toolCalls, state.result) }
      state.callContinuations[id] = continuation
      return (id, state.toolCalls, nil)
    }
    for call in calls { continuation.yield(call) }
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
}

extension EdgeToolsSessionStream {
  private func makeTokenIterator() -> EdgeToolsSessionTokens.AsyncIterator {
    let (stream, continuation) = AsyncThrowingStream<EdgeToolsToken, any Error>.makeStream()
    let (id, tokens, result) = self.state.withLock { state in
      let id = state.nextID
      state.nextID += 1
      guard state.result == nil else { return (id, state.tokens, state.result) }
      state.tokenContinuations[id] = continuation
      return (id, state.tokens, nil)
    }
    for token in tokens { continuation.yield(token) }
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
    toolDefinitions: [EdgeToolDefinition],
    parameters: Engine.GenerateParameters
  ) async throws -> EdgeToolsSessionGeneration {
    let wasStopped = self.state.withLock { $0.wasStoppedBeforeGeneration }
    if wasStopped {
      let generation = EdgeToolsSessionGeneration(
        engineGeneration: .empty,
        toolCalls: EdgeToolCallCollection()
      )
      self.finish(with: .success(generation))
      return generation
    }

    let channel = EdgeToolsGenerationChannel(
      onToken: { [weak self] token in self?.emit(token: token) },
      onToolCall: { [weak self] call in self?.emit(rawCall: call) }
    )
    do {
      let generationTask = try session.engine.generate(
        prompt: prompt,
        tools: toolDefinitions,
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
      let generation = EdgeToolsSessionGeneration(
        engineGeneration: engineGeneration,
        toolCalls: self.toolCalls
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
    for continuation in continuations {
      continuation.yield(token)
    }
  }

  private func emit(rawCall: EdgeRawToolCall) {
    guard let call = self.resolve(rawCall) else { return }
    let continuations = self.registrar.withMutation(of: self, keyPath: \.toolCalls) {
      self.state.withLock { state in
        state.toolCalls.append(call)
        return Array(state.callContinuations.values)
      }
    }
    if self.shouldInvokeTools(call) {
      _ = Task { _ = try await call.output }
    }
    for continuation in continuations {
      continuation.yield(call)
    }
  }

  private func finish(with result: Result<EdgeToolsSessionGeneration, any Error>) {
    let completion = self.registrar.withMutation(of: self, keyPath: \.result) {
      self.state.withLock { state in
        state.result = result
        let completion = (
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
    completion.0?()
    switch result {
    case .success:
      for continuation in completion.1 { continuation.finish() }
      for continuation in completion.2 { continuation.finish() }
    case .failure(let error):
      for continuation in completion.1 { continuation.finish(throwing: error) }
      for continuation in completion.2 { continuation.finish(throwing: error) }
    }
  }

  private func resolve(_ rawCall: EdgeRawToolCall) -> AnyEdgeToolCall? {
    guard let tool = self.toolsByName[rawCall.name.snakeCased()] else { return nil }
    func open<Tool: EdgeTool>(_ tool: Tool) -> AnyEdgeToolCall? {
      let call = try? EdgeToolCall(id: EdgeToolCallID(), tool: tool, rawInput: rawCall.arguments)
      guard let call else { return nil }
      return AnyEdgeToolCall(call)
    }
    return open(tool)
  }
}

extension EdgeToolsSession {
  public func stream(
    prompt: Engine.Prompt,
    parameters: Engine.GenerateParameters = .default,
    shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool = { _ in true }
  ) -> EdgeToolsSessionStream {
    let tools = self.tools
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
    self.registerActiveStream(stream)
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
    parameters: Engine.GenerateParameters = .default,
    shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool = { _ in true }
  ) async throws -> EdgeToolsSessionGeneration {
    try await self.stream(
      prompt: prompt,
      parameters: parameters,
      shouldInvokeTools: shouldInvokeTools
    )
    .finalGeneration
  }
}

// MARK: - Duplicate Tool Name Error

package func duplicateToolNameError(_ names: some Sequence<String>) -> String? {
  let grouped = Dictionary(grouping: names) { $0.snakeCased() }
  let duplicates = grouped.filter { $0.value.count > 1 }
  guard !duplicates.isEmpty else { return nil }
  return duplicates.sorted { $0.key < $1.key }
    .map { normalized, originals in
      "The names \(originals.sorted().map { "'\($0)'" }.joined(separator: " and ")) all normalize to '\(normalized)'."
    }
    .joined(separator: "\n")
}
