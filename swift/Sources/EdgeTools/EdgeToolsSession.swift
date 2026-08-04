import _Concurrency

#if !$Embedded
  import Observation
#endif

// MARK: - EdgeToolsSession

public final class EdgeToolsSession<Engine: EdgeToolsEngine>: Sendable {
  public let engine: Engine
  private let _tools: Lock<[EdgeToolsSessionTool]>
  private let _activeStreams = Lock([EdgeToolsSessionStream]())
  private let observationRegistrar = _ObservationRegistrar()

  public var tools: [EdgeToolsSessionTool] {
    get {
      self.access(.tools)
      return self._tools.withLock { $0 }
    }
    set {
      self._tools.withLock { tools in
        self.withMutation(of: .tools) {
          tools = newValue
        }
      }
    }
  }

  public var isResponding: Bool {
    !self.activeStreams.isEmpty
  }

  public var activeStreams: [EdgeToolsSessionStream] {
    self.access(.activeStreams)
    return self._activeStreams.withLock { $0 }
  }

  #if !$Embedded
    public init(engine: sending Engine, tools: [any EdgeTool]) {
      self.engine = engine
      self._tools = Lock(tools.map { EdgeToolsSessionTool($0) })
    }
  #endif

  public init(engine: sending Engine, tools: [EdgeToolsSessionTool] = []) {
    self.engine = engine
    self._tools = Lock(tools)
  }

  public convenience init(
    engine: sending Engine,
    @EdgeToolsBuilder tools: () -> [EdgeToolsSessionTool]
  ) {
    self.init(engine: engine, tools: tools())
  }

  fileprivate func registerActiveStream(_ stream: EdgeToolsSessionStream) {
    self._activeStreams.withLock { streams in
      self.withMutation(of: .activeStreams) {
        streams.append(stream)
      }
    }
  }

  fileprivate func removeActiveStream(_ stream: EdgeToolsSessionStream) {
    self._activeStreams.withLock { streams in
      self.withMutation(of: .activeStreams) {
        streams.removeAll { $0 === stream }
      }
    }
  }
}

extension EdgeToolsSession {
  public func tokenize(prompt: Engine.Prompt) async throws -> [EdgeToolsToken] {
    let toolDefinitions = self.tools.map { $0.definition }
    return try await self.engine.tokenize(prompt: prompt, tools: toolDefinitions)
  }
}

extension EdgeToolsSession where Engine: EdgeToolsPrefillableEngine {
  public func prefill(promptPrefix: Engine.Prompt) async throws -> EdgeToolsEnginePrefill {
    let toolDefinitions = self.tools.map { $0.definition }
    return try await self.engine.prefill(promptPrefix: promptPrefix, tools: toolDefinitions)
  }
}

// MARK: - EdgeToolsSessionTool

// NB: Every requirement is non-generic so that it stays callable through the existential.
// Embedded Swift cannot open an existential back into a generic context.
private protocol _EdgeToolsSessionTool: AnyObject, Sendable {
  var erasedTool: any EdgeTool { get }
  var erasedName: String { get }
  var erasedDefinition: EdgeToolDefinition { get }
  func makeCall(id: EdgeToolCallID, rawInput: EdgeToolsValue) -> AnyEdgeToolCall?
}

private final class EdgeToolsSessionToolBox<Tool: EdgeTool>: _EdgeToolsSessionTool {
  let tool: Tool

  var erasedTool: any EdgeTool { self.tool }
  var erasedName: String { self.tool.name }
  var erasedDefinition: EdgeToolDefinition { self.tool.definition }

  init(_ tool: Tool) {
    self.tool = tool
  }

  func makeCall(id: EdgeToolCallID, rawInput: EdgeToolsValue) -> AnyEdgeToolCall? {
    guard let call = try? EdgeToolCall(id: id, tool: self.tool, rawInput: rawInput) else {
      return nil
    }
    return AnyEdgeToolCall.erasing(call)
  }
}

public struct EdgeToolsSessionTool: Sendable {
  private let base: any _EdgeToolsSessionTool

  public var tool: any EdgeTool { self.base.erasedTool }
  public var name: String { self.base.erasedName }
  public var definition: EdgeToolDefinition { self.base.erasedDefinition }

  public init(_ tool: some EdgeTool) {
    self.base = EdgeToolsSessionToolBox(tool)
  }

  func makeCall(id: EdgeToolCallID, rawInput: EdgeToolsValue) -> AnyEdgeToolCall? {
    self.base.makeCall(id: id, rawInput: rawInput)
  }
}

// MARK: - EdgeToolsBuilder

@resultBuilder
public enum EdgeToolsBuilder {
  public static func buildExpression(_ tool: some EdgeTool) -> EdgeToolsSessionTool {
    EdgeToolsSessionTool(tool)
  }

  public static func buildExpression(_ tool: EdgeToolsSessionTool) -> EdgeToolsSessionTool {
    tool
  }

  public static func buildBlock(_ tools: EdgeToolsSessionTool...) -> [EdgeToolsSessionTool] {
    tools
  }

  public static func buildOptional(_ tools: [EdgeToolsSessionTool]?) -> [EdgeToolsSessionTool] {
    tools ?? []
  }

  public static func buildEither(first tools: [EdgeToolsSessionTool]) -> [EdgeToolsSessionTool] {
    tools
  }

  public static func buildEither(second tools: [EdgeToolsSessionTool]) -> [EdgeToolsSessionTool] {
    tools
  }

  public static func buildArray(_ tools: [[EdgeToolsSessionTool]]) -> [EdgeToolsSessionTool] {
    tools.flatMap { $0 }
  }
}

// MARK: - Generation

public struct EdgeToolsSessionGeneration: Sendable {
  public let engineGeneration: EdgeToolsEngineGeneration
  public let toolCalls: EdgeToolCallCollection

  public var response: String {
    self.engineGeneration.response
  }

  public func decoded<Response: ConvertibleFromEdgeToolsValue>(
    as type: Response.Type
  ) throws -> Response {
    return try Response(edgeToolsValue: EdgeToolsValue(json: self.response))
  }
}

// MARK: - Stream

public final class EdgeToolsSessionStream: Sendable, Identifiable {
  public typealias Element = EdgeToolCallCollection.Element

  private struct State {
    var task: Task<EdgeToolsSessionGeneration, any Error>?
    var hasStarted = false
    var result: Result<EdgeToolsSessionGeneration, any Error>?
    var wasStoppedBeforeGeneration = false
    var stop: (@Sendable () -> Void)?
    var toolCalls = EdgeToolCallCollection()
    var tokens = [EdgeToolsToken]()
    var tokenSubscribers = [Int: @Sendable (EdgeToolsToken) -> Void]()
    var callSubscribers = [Int: @Sendable (Element) -> Void]()
    var finishSubscribers =
      [Int: @Sendable (Result<EdgeToolsSessionGeneration, any Error>) -> Void]()
    var onFinish: (@Sendable () -> Void)?
    var nextID = 0
  }

  private let state = Lock(State())
  private let registrar = _ObservationRegistrar()
  private let toolsByName: [String: EdgeToolsSessionTool]
  private let shouldInvokeTools: @Sendable (AnyEdgeToolCall) -> Bool

  public var isGenerating: Bool {
    self.result == nil
  }

  public var isFinished: Bool {
    self.result != nil
  }

  public var toolCalls: EdgeToolCallCollection {
    self.access(.toolCalls)
    return self.state.withLock { $0.toolCalls }
  }

  public var result: Result<EdgeToolsSessionGeneration, any Error>? {
    self.access(.result)
    return self.state.withLock { $0.result }
  }

  public var response: Result<String, any Error>? {
    self.result.map { $0.map { $0.response } }
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

  fileprivate init(
    tools: [EdgeToolsSessionTool],
    shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool
  ) {
    self.toolsByName = Dictionary(
      uniqueKeysWithValues: tools.map { ($0.name.snakeCased(), $0) }
    )
    self.shouldInvokeTools = shouldInvokeTools
  }
}

// MARK: - Subscribing

extension EdgeToolsSessionStream {
  public func onToken(
    _ body: @escaping @Sendable (EdgeToolsToken) -> Void
  ) -> EdgeToolsSubscription {
    self.subscribe(onToken: body, onToolCall: nil, onFinish: nil)
  }

  public func onToolCall(
    _ body: @escaping @Sendable (Element) -> Void
  ) -> EdgeToolsSubscription {
    self.subscribe(onToken: nil, onToolCall: body, onFinish: nil)
  }

  public func onFinish(
    _ body: @escaping @Sendable (Result<EdgeToolsSessionGeneration, any Error>) -> Void
  ) -> EdgeToolsSubscription {
    self.subscribe(onToken: nil, onToolCall: nil, onFinish: body)
  }

  private func subscribe(
    onToken: (@Sendable (EdgeToolsToken) -> Void)?,
    onToolCall: (@Sendable (Element) -> Void)?,
    onFinish: (@Sendable (Result<EdgeToolsSessionGeneration, any Error>) -> Void)?
  ) -> EdgeToolsSubscription {
    let (id, tokens, calls, result) = self.state.withLock { state in
      let id = state.nextID
      state.nextID += 1
      guard state.result == nil else {
        return (id, state.tokens, state.toolCalls, state.result)
      }
      if let onToken { state.tokenSubscribers[id] = onToken }
      if let onToolCall { state.callSubscribers[id] = onToolCall }
      if let onFinish { state.finishSubscribers[id] = onFinish }
      return (id, state.tokens, state.toolCalls, nil)
    }
    if let onToken { for token in tokens { onToken(token) } }
    if let onToolCall { for call in calls { onToolCall(call) } }
    if let result { onFinish?(result) }
    return EdgeToolsSubscription { [self] in
      self.state.withLock { state in
        state.tokenSubscribers.removeValue(forKey: id)
        state.callSubscribers.removeValue(forKey: id)
        state.finishSubscribers.removeValue(forKey: id)
      }
    }
  }
}

// MARK: - Generating

extension EdgeToolsSessionStream {
  fileprivate func start<Engine: EdgeToolsEngine>(
    session: EdgeToolsSession<Engine>,
    prompt: Engine.Prompt,
    toolDefinitions: [EdgeToolDefinition],
    parameters: sending Engine.GenerateParameters
  ) {
    let shouldStart = self.state.withLock { state in
      guard !state.hasStarted else { return false }
      state.hasStarted = true
      return true
    }
    guard shouldStart else { return }

    // NB: Compiler region isolation checker limitation, this is safe because params are not
    // accessed after being sent to generate.
    nonisolated(unsafe) let parameters = parameters
    let task = Task {
      try await self.runGeneration(
        session: session,
        prompt: prompt,
        toolDefinitions: toolDefinitions,
        parameters: parameters
      )
    }
    self.state.withLock { $0.task = task }
  }

  fileprivate func setOnFinish(_ onFinish: @escaping @Sendable () -> Void) {
    self.state.withLock { $0.onFinish = onFinish }
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

  public func decodedResponse<Response: ConvertibleFromEdgeToolsValue>(
    as type: Response.Type
  ) async throws -> Response {
    let generation = try await self.finalGeneration
    return try generation.decoded(as: type)
  }
}

extension EdgeToolsSessionStream {
  private func runGeneration<Engine: EdgeToolsEngine>(
    session: EdgeToolsSession<Engine>,
    prompt: Engine.Prompt,
    toolDefinitions: [EdgeToolDefinition],
    parameters: sending Engine.GenerateParameters
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
      onToken: { token in self.emit(token: token) },
      onToolCall: { call in self.emit(rawCall: call) }
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
    let subscribers = self.state.withLock { state in
      state.tokens.append(token)
      return Array(state.tokenSubscribers.values)
    }
    for subscriber in subscribers {
      subscriber(token)
    }
  }

  private func emit(rawCall: EdgeRawToolCall) {
    guard let call = self.resolve(rawCall) else { return }
    let subscribers = self.withMutation(of: .toolCalls) {
      self.state.withLock { state in
        state.toolCalls.append(call)
        return Array(state.callSubscribers.values)
      }
    }
    if self.shouldInvokeTools(call) {
      _ = Task { _ = try await call.output }
    }
    for subscriber in subscribers {
      subscriber(call)
    }
  }

  private func finish(with result: Result<EdgeToolsSessionGeneration, any Error>) {
    let completion = self.withMutation(of: .result) {
      self.state.withLock { state in
        state.result = result
        let completion = (state.onFinish, Array(state.finishSubscribers.values))
        state.tokenSubscribers.removeAll()
        state.callSubscribers.removeAll()
        state.finishSubscribers.removeAll()
        state.stop = nil
        // NB: Cleared so that the retain cycle through the session is broken deterministically.
        state.onFinish = nil
        return completion
      }
    }
    completion.0?()
    for subscriber in completion.1 { subscriber(result) }
  }

  private func resolve(_ rawCall: EdgeRawToolCall) -> AnyEdgeToolCall? {
    self.toolsByName[rawCall.name.snakeCased()]?
      .makeCall(id: EdgeToolCallID(), rawInput: rawCall.arguments)
  }
}

// MARK: - Async Sequences

#if !$Embedded
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

  extension EdgeToolsSessionStream: AsyncSequence {
    public struct AsyncIterator: AsyncIteratorProtocol {
      fileprivate var base: AsyncThrowingStream<Element, any Error>.AsyncIterator

      public mutating func next() async throws -> Element? {
        try await self.base.next()
      }
    }

    public var tokens: EdgeToolsSessionTokens {
      EdgeToolsSessionTokens(makeIterator: { self.makeTokenIterator() })
    }

    public func makeAsyncIterator() -> AsyncIterator {
      let (stream, continuation) = AsyncThrowingStream<Element, any Error>.makeStream()
      let subscription = self.subscribe(
        onToken: nil,
        onToolCall: { continuation.yield($0) },
        onFinish: { result in
          switch result {
          case .success: continuation.finish()
          case .failure(let error): continuation.finish(throwing: error)
          }
        }
      )
      continuation.onTermination = { _ in subscription.cancel() }
      return AsyncIterator(base: stream.makeAsyncIterator())
    }

    private func makeTokenIterator() -> EdgeToolsSessionTokens.AsyncIterator {
      let (stream, continuation) = AsyncThrowingStream<EdgeToolsToken, any Error>.makeStream()
      let subscription = self.subscribe(
        onToken: { continuation.yield($0) },
        onToolCall: nil,
        onFinish: { result in
          switch result {
          case .success: continuation.finish()
          case .failure(let error): continuation.finish(throwing: error)
          }
        }
      )
      continuation.onTermination = { _ in subscription.cancel() }
      return EdgeToolsSessionTokens.AsyncIterator(base: stream.makeAsyncIterator())
    }
  }
#endif

// MARK: - Streaming

extension EdgeToolsSession {
  public func stream(
    prompt: Engine.Prompt,
    parameters: sending Engine.GenerateParameters = .default,
    shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool = { _ in true }
  ) -> EdgeToolsSessionStream {
    let tools = self.tools
    if let message = duplicateToolNameError(tools.map { $0.name }) {
      assertionFailure(message)
    }
    let stream = EdgeToolsSessionStream(tools: tools, shouldInvokeTools: shouldInvokeTools)
    self.registerActiveStream(stream)
    // NB: The strong capture cannot leak the session, because the stream clears its finish
    // handler once it completes, and the session drops the stream at that same point.
    stream.setOnFinish { [self] in self.removeActiveStream(stream) }
    stream.start(
      session: self,
      prompt: prompt,
      toolDefinitions: tools.map { $0.definition },
      parameters: parameters
    )
    return stream
  }

  @concurrent
  public func generate(
    prompt: Engine.Prompt,
    parameters: sending Engine.GenerateParameters = .default,
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

// MARK: - Observation

extension EdgeToolsSession {
  fileprivate enum ObservedProperty {
    case tools
    case activeStreams
  }

  fileprivate func access(_ property: ObservedProperty) {
    #if !$Embedded
      switch property {
      case .tools: self.observationRegistrar.access(self, keyPath: \.tools)
      case .activeStreams: self.observationRegistrar.access(self, keyPath: \.activeStreams)
      }
    #endif
  }

  fileprivate func withMutation<Result>(
    of property: ObservedProperty,
    _ body: () -> Result
  ) -> Result {
    #if !$Embedded
      switch property {
      case .tools:
        self.observationRegistrar.withMutation(of: self, keyPath: \.tools, body)
      case .activeStreams:
        self.observationRegistrar.withMutation(of: self, keyPath: \.activeStreams, body)
      }
    #else
      body()
    #endif
  }
}

extension EdgeToolsSessionStream {
  fileprivate enum ObservedProperty {
    case toolCalls
    case result
  }

  fileprivate func access(_ property: ObservedProperty) {
    #if !$Embedded
      switch property {
      case .toolCalls: self.registrar.access(self, keyPath: \.toolCalls)
      case .result: self.registrar.access(self, keyPath: \.result)
      }
    #endif
  }

  fileprivate func withMutation<Result>(
    of property: ObservedProperty,
    _ body: () -> Result
  ) -> Result {
    #if !$Embedded
      switch property {
      case .toolCalls: self.registrar.withMutation(of: self, keyPath: \.toolCalls, body)
      case .result: self.registrar.withMutation(of: self, keyPath: \.result, body)
      }
    #else
      body()
    #endif
  }
}

#if !$Embedded
  extension EdgeToolsSession: Observable {}
  extension EdgeToolsSessionStream: Observable {}
#endif

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
