import Foundation
import Observation

// MARK: - EdgeToolsSession

public final class EdgeToolsSession<Engine: EdgeToolEngine>: Sendable, Observable {
  fileprivate let engine: Engine
  private let _systemPrompt: Lock<String>
  private let _activeStreams = Lock([EdgeToolsSessionStream]())
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

  public var activeStreams: [EdgeToolsSessionStream] {
    self.observationRegistrar.access(self, keyPath: \.activeStreams)
    return self._activeStreams.withLock { $0 }
  }

  public init(engine: sending Engine, systemPrompt: String = "") {
    self.engine = engine
    self._systemPrompt = Lock(systemPrompt)
  }

  fileprivate func removeActiveStream(_ stream: EdgeToolsSessionStream) {
    self._activeStreams.withLock { activeStreams in
      self.observationRegistrar.withMutation(of: self, keyPath: \.activeStreams) {
        activeStreams.removeAll { $0 === stream }
      }
    }
  }
}

// MARK: - Tokenize

extension EdgeToolsSession {
  public func tokenize(
    tools: [any EdgeTool],
    with prompt: String,
    systemPromptOverride: String?
  ) async throws -> [EdgeToolsToken] {
    try await self.tokenize(
      tools: tools.map(\.definition),
      with: prompt,
      systemPromptOverride: systemPromptOverride
    )
  }

  public func tokenize(
    tools: [EdgeToolDefinition],
    with prompt: String,
    systemPromptOverride: String?
  ) async throws -> [EdgeToolsToken] {
    let prompt = EdgeToolsPrompt(
      system: systemPromptOverride ?? self.systemPrompt,
      user: prompt,
      tools: tools
    )
    return try await self.tokenize(prompt: prompt)
  }

  public func tokenize(prompt: EdgeToolsPrompt) async throws -> [EdgeToolsToken] {
    try await self.engine.tokenize(prompt: prompt)
  }
}

// MARK: - Generate

public struct EdgeToolsSessionGeneration: Sendable {
  public let engineGeneration: EdgeToolEngineGeneration
  public let toolCalls: EdgeToolCallCollection
}

extension EdgeToolsSession {
  @concurrent
  public func generate(
    tools: [any EdgeTool],
    with prompt: String,
    systemPromptOverride: String? = nil,
    parameters: Engine.GenerateParameters = .default,
    shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool = { _ in true }
  ) async throws -> EdgeToolsSessionGeneration {
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

public final class EdgeToolsSessionStream: Sendable, Observable, Identifiable {
  public var isGenerating: Bool {
    self.registrar.access(self, keyPath: \.isGenerating)
    return self.state.withLock { $0.result == nil }
  }

  public var isFinished: Bool {
    self.registrar.access(self, keyPath: \.isFinished)
    return self.state.withLock { $0.result != nil }
  }

  public var finalGeneration: EdgeToolsSessionGeneration {
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

  public var toolCalls: EdgeToolCallCollection {
    self.registrar.access(self, keyPath: \.toolCalls)
    return self.state.withLock { $0.toolCalls }
  }

  public var result: Result<EdgeToolsSessionGeneration, any Error>? {
    self.registrar.access(self, keyPath: \.result)
    return self.state.withLock { $0.result }
  }

  private let state = Lock(State())
  private let registrar = ObservationRegistrar()

  deinit {
    self.state.withLock { state in
      state.task?.cancel()
      if state.result == nil {
        state.generationTaskStop?()
      }
    }
  }

  fileprivate init<Engine: EdgeToolEngine>(
    session: EdgeToolsSession<Engine>,
    tools: [any EdgeTool],
    with prompt: String,
    systemPrompt: String,
    parameters: Engine.GenerateParameters,
    shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool
  ) {
    let task = Task {
      return try await self.runGeneration(
        session: session,
        tools: tools,
        prompt: prompt,
        systemPrompt: systemPrompt,
        parameters: parameters,
        shouldInvokeTools: shouldInvokeTools
      )
    }
    self.state.withLock { $0.task = task }
  }

  private func runGeneration<Engine: EdgeToolEngine>(
    session: EdgeToolsSession<Engine>,
    tools: [any EdgeTool],
    prompt: String,
    systemPrompt: String,
    parameters: Engine.GenerateParameters,
    shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool
  ) async throws -> EdgeToolsSessionGeneration {
    let stoppedBeforeGeneration: EdgeToolsSessionGeneration? = self.state.withLock { state in
      guard state.wasStoppedBeforeGeneration else { return nil }
      let sessionGeneration = EdgeToolsSessionGeneration(
        engineGeneration: .empty,
        toolCalls: state.toolCalls
      )
      self.registrar.withMutation(of: self, keyPath: \.result) {
        state.finish(with: .success(sessionGeneration))
      }
      return sessionGeneration
    }
    if let stoppedBeforeGeneration {
      session.removeActiveStream(self)
      return stoppedBeforeGeneration
    }

    let edgeToolsPrompt = EdgeToolsPrompt(
      system: systemPrompt,
      user: prompt,
      tools: tools.map(\.definition)
    )
    let toolsByName = Dictionary(
      uniqueKeysWithValues: tools.map { ($0.name.snakeCased(), $0) }
    )
    let shouldInvokeToolsBox = Lock(shouldInvokeTools)

    let generationTask: Engine.GenerationTask
    do {
      generationTask = try session.engine.generate(
        prompt: edgeToolsPrompt,
        parameters: parameters
      ) { [weak self] token, rawToolCall in
        guard let self else { return }
        let call = rawToolCall.flatMap { self.resolve($0, toolsByName: toolsByName) }
        let shouldInvoke = shouldInvokeToolsBox.withLock { $0 }
        self.state.withLock { state in
          state.emitToken(token: token)
          if let call {
            self.registrar.withMutation(of: self, keyPath: \.toolCalls) {
              state.emitToolCall(call, shouldInvoke: shouldInvoke)
            }
          }
        }
      }
    } catch {
      self.state.withLock { state in
        self.registrar.withMutation(of: self, keyPath: \.result) {
          state.finish(with: .failure(error))
        }
      }
      session.removeActiveStream(self)
      throw error
    }

    let shouldStopGeneration = self.state.withLock { state in
      state.generationTaskStop = { generationTask.stop() }
      if state.wasStoppedBeforeGeneration {
        return true
      }
      return false
    }
    if shouldStopGeneration {
      generationTask.stop()
    }

    let generation: EdgeToolEngineGeneration
    do {
      generation = try await generationTask.value
      try Task.checkCancellation()
    } catch {
      self.state.withLock { state in
        self.registrar.withMutation(of: self, keyPath: \.result) {
          state.finish(with: .failure(error))
        }
      }
      session.removeActiveStream(self)
      throw error
    }
    let sessionGeneration = self.state.withLock { state in
      let sessionGeneration = EdgeToolsSessionGeneration(
        engineGeneration: generation,
        toolCalls: state.toolCalls
      )
      self.registrar.withMutation(of: self, keyPath: \.result) {
        state.finish(with: .success(sessionGeneration))
      }
      return sessionGeneration
    }
    session.removeActiveStream(self)
    return sessionGeneration
  }

  public func stop() {
    let generationTaskStop: (@Sendable () -> Void)? = self.state.withLock { state in
      guard state.result == nil else { return nil }
      guard let generationTaskStop = state.generationTaskStop else {
        state.markStoppedBeforeGeneration()
        return nil
      }
      return generationTaskStop
    }
    generationTaskStop?()
  }

  private func resolve(
    _ rawToolCall: EdgeRawToolCall,
    toolsByName: [String: any EdgeTool]
  ) -> AnyEdgeToolCall? {
    guard let tool = toolsByName[rawToolCall.name.snakeCased()] else { return nil }

    func open<Tool: EdgeTool>(_ concrete: Tool) -> AnyEdgeToolCall? {
      guard let input = try? Tool.Input(edgeToolsValue: rawToolCall.arguments) else { return nil }
      return AnyEdgeToolCall(EdgeToolCall(id: EdgeToolCallID(), tool: concrete, input: input))
    }
    return open(tool)
  }
}

extension EdgeToolsSessionStream: AsyncSequence {
  public struct AsyncIterator: AsyncIteratorProtocol {
    fileprivate var base: AsyncThrowingStream<AnyEdgeToolCall, any Error>.AsyncIterator

    public mutating func next() async throws -> EdgeToolCallCollection.Element? {
      try await self.base.next()
    }

    @available(iOS 18.0, macOS 15.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public mutating func next(
      isolation actor: isolated (any Actor)?
    ) async throws -> EdgeToolCallCollection.Element? {
      try await self.base.next(isolation: actor)
    }
  }

  public func makeAsyncIterator() -> AsyncIterator {
    let (stream, continuation, id) = self.state.withLock { $0.takeToolCallStream() }
    continuation.onTermination = { [weak self] reason in
      switch reason {
      case .cancelled: self?.state.withLock { $0.removeToolContinuation(id: id) }
      default: break
      }
    }
    return AsyncIterator(base: stream.makeAsyncIterator())
  }
}

extension EdgeToolsSessionStream {
  public struct Tokens: AsyncSequence {
    fileprivate let stream: EdgeToolsSessionStream

    public struct AsyncIterator: AsyncIteratorProtocol {
      fileprivate var base: AsyncThrowingStream<EdgeToolsToken, any Error>.AsyncIterator

      public mutating func next() async throws -> EdgeToolsToken? {
        try await self.base.next()
      }

      @available(iOS 18.0, macOS 15.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
      public mutating func next(
        isolation actor: isolated (any Actor)?
      ) async throws -> EdgeToolsToken? {
        try await self.base.next(isolation: actor)
      }
    }

    public func makeAsyncIterator() -> AsyncIterator {
      let (tokenStream, continuation, id) = self.stream.state.withLock { $0.takeTokenStream() }
      continuation.onTermination = { [weak stream] reason in
        switch reason {
        case .cancelled: stream?.state.withLock { $0.removeTokenContinuation(id: id) }
        default: break
        }
      }
      return AsyncIterator(base: tokenStream.makeAsyncIterator())
    }
  }

  public var tokens: Tokens {
    Tokens(stream: self)
  }
}

extension EdgeToolsSessionStream {
  fileprivate struct State {
    private(set) var result: Result<EdgeToolsSessionGeneration, any Error>?
    private(set) var wasStoppedBeforeGeneration = false
    var generationTaskStop: (@Sendable () -> Void)?
    private var tokenContinuations =
      [Int: AsyncThrowingStream<EdgeToolsToken, any Error>.Continuation]()
    private var toolsContinuations =
      [Int: AsyncThrowingStream<AnyEdgeToolCall, any Error>.Continuation]()
    private var nextTokenContinuationID = 0
    private var nextToolContinuationID = 0
    private(set) var toolCalls = EdgeToolCallCollection()
    var task: Task<EdgeToolsSessionGeneration, any Error>?

    mutating func markStoppedBeforeGeneration() {
      if self.result == nil {
        self.wasStoppedBeforeGeneration = true
      }
    }

    mutating func takeTokenStream() -> (
      AsyncThrowingStream<EdgeToolsToken, any Error>,
      AsyncThrowingStream<EdgeToolsToken, any Error>.Continuation,
      Int
    ) {
      let (stream, continuation) = AsyncThrowingStream<EdgeToolsToken, any Error>.makeStream()
      let id = self.nextTokenContinuationID
      self.nextTokenContinuationID += 1
      self.tokenContinuations[id] = continuation
      return (stream, continuation, id)
    }

    mutating func takeToolCallStream() -> (
      AsyncThrowingStream<AnyEdgeToolCall, any Error>,
      AsyncThrowingStream<AnyEdgeToolCall, any Error>.Continuation,
      Int
    ) {
      let (stream, continuation) = AsyncThrowingStream<AnyEdgeToolCall, any Error>.makeStream()
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

    func emitToken(token: EdgeToolsToken) {
      for continuation in self.tokenContinuations.values {
        continuation.yield(token)
      }
    }

    mutating func emitToolCall(
      _ toolCall: AnyEdgeToolCall,
      shouldInvoke: (AnyEdgeToolCall) -> Bool
    ) {
      self.toolCalls.append(toolCall)
      if shouldInvoke(toolCall) {
        _ = Task { _ = try await toolCall.output }
      }
      for continuation in self.toolsContinuations.values {
        continuation.yield(toolCall)
      }
    }

    mutating func finish(with result: Result<EdgeToolsSessionGeneration, any Error>) {
      self.generationTaskStop = nil
      self.result = result
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

extension EdgeToolsSession {
  public func stream(
    tools: [any EdgeTool],
    with prompt: String,
    systemPromptOverride: String? = nil,
    parameters: Engine.GenerateParameters = .default,
    shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool = { _ in true }
  ) -> EdgeToolsSessionStream {
    if let message = duplicateToolNameError(tools.map(\.name)) {
      assertionFailure(message)
    }
    let stream = EdgeToolsSessionStream(
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

// MARK: - Duplicate Tool Name Error

package func duplicateToolNameError(_ names: some Sequence<String>) -> String? {
  var grouped = [String: [String]]()
  for name in names {
    grouped[name.snakeCased(), default: []].append(name)
  }
  let duplicates = grouped.filter { $0.value.count > 1 }
  guard !duplicates.isEmpty else { return nil }
  return
    duplicates
    .sorted { $0.key < $1.key }
    .map { normalized, originals in
      "The names \(originals.sorted().map { "'\($0)'" }.joined(separator: " and ")) all normalize to '\(normalized)'."
    }
    .joined(separator: "\n")
}
