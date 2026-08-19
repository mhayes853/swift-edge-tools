import EdgeToolsCore
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
    @EdgeToolsToolBuilder tools: () -> [EdgeToolsSessionTool]
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

  fileprivate func resolveContext(_ context: Engine.Context?) -> Engine.Context {
    context ?? self.context()
  }
}

// MARK: - Contexts

extension EdgeToolsSession {
  public func context() -> Engine.Context {
    self.engine.context()
  }

  public func context(_ parameters: Engine.ContextParameters) -> Engine.Context {
    self.engine.context(parameters)
  }
}

extension EdgeToolsSession where Engine: EdgeToolsTokenizingEngine {
  public func tokenize(
    prompt: Engine.Prompt,
    context: Engine.Context? = nil
  ) async throws -> [EdgeToolsToken] {
    let toolDefinitions = self.tools.map { $0.definition }
    let context = self.resolveContext(context)
    return try await self.engine.tokenize(
      prompt: prompt,
      tools: toolDefinitions,
      context: context
    )
  }
}

// MARK: - Extraction

extension EdgeToolsSession {
  @concurrent
  public func extract<Response: EdgeToolsGenerable>(
    prompt: Engine.Prompt,
    as type: Response.Type,
    context: Engine.Context? = nil,
    parameters: sending Engine.GenerateParameters = .default
  ) async throws -> Response? {
    let definition = Response.extractionToolDefinition
    let task = try self.engine.generate(
      prompt: prompt,
      tools: [definition],
      parameters: parameters,
      context: self.resolveContext(context),
      channel: EdgeToolsGenerationChannel()
    )
    let generation = try await task.value
    let call = generation.toolCalls.first { $0.name == definition.name }
    guard let call else {
      return nil
    }
    return try Response(edgeToolsValue: call.arguments)
  }
}

extension EdgeToolsSession where Engine: EdgeToolsPrefillableEngine {
  public func prefill(
    promptPrefix: Engine.Prompt,
    context: Engine.Context
  ) async throws -> EdgeToolsEnginePrefill {
    let toolDefinitions = self.tools.map { $0.definition }
    return try await self.engine.prefill(
      promptPrefix: promptPrefix,
      tools: toolDefinitions,
      context: context
    )
  }
}

// MARK: - Agent Generation

public struct EdgeToolsAgentTurn<Context: Sendable>: Sendable {
  public let index: Int
  public let context: Context
  public let prompt: EdgeToolsTranscript.Prompt

  public init(
    index: Int,
    context: sending Context,
    prompt: EdgeToolsTranscript.Prompt
  ) {
    self.index = index
    self.context = context
    self.prompt = prompt
  }
}

public struct EdgeToolsAgentResult<Result: Sendable>: Sendable {
  public let output: Result
  public let generations: [EdgeToolsSessionGeneration]
  public let toolCalls: EdgeToolCallCollection

  public init(
    output: sending Result,
    generations: [EdgeToolsSessionGeneration],
    toolCalls: EdgeToolCallCollection
  ) {
    self.output = output
    self.generations = generations
    self.toolCalls = toolCalls
  }
}

public enum EdgeToolsAgentError: Error, Sendable {
  case maximumTurnsExceeded(Int)
  case invalidToolOutput(String)
}

extension EdgeToolsSession
where
  Engine.Prompt == EdgeToolsTranscript.Prompt,
  Engine.GenerateParameters: EdgeToolsConstrainedGenerateParameters,
  Engine.GenerateParameters.Constraint: EdgeToolsTurnGenerationConstraint
{
  @concurrent
  public func respond<Result: EdgeToolsGenerable>(
    to initialPrompt: Engine.Prompt,
    as type: Result.Type,
    context: Engine.Context? = nil,
    maximumTurns: Int? = nil,
    parameters: @escaping @Sendable (EdgeToolsAgentTurn<Engine.Context>) -> Engine.GenerateParameters = {
      _ in .default
    },
    constraint: @escaping @Sendable (
      Result.Type,
      EdgeToolsAgentTurn<Engine.Context>
    ) -> Engine.GenerateParameters.Constraint = {
      type, _ in .toolCallsOrResponse(type, toolCallRange: .unbounded(minimum: 1))
    }
  ) async throws -> EdgeToolsAgentResult<Result> {
    var prompt = initialPrompt
    let context = self.resolveContext(context)
    var generations = [EdgeToolsSessionGeneration]()
    var toolCalls = EdgeToolCallCollection()

    var index = 0
    while maximumTurns.map({ index < $0 }) ?? true {
      let turn = EdgeToolsAgentTurn(index: index, context: context, prompt: prompt)
      var generationParameters = parameters(turn)
      generationParameters.constraint = constraint(Result.self, turn)
      let generation = try await self.generate(
        prompt: prompt,
        context: context,
        parameters: generationParameters,
        shouldInvokeTools: { _ in false }
      )
      generations.append(generation)
      toolCalls.append(contentsOf: generation.toolCalls)

      guard !generation.toolCalls.isEmpty else {
        return EdgeToolsAgentResult(
          output: try generation.decoded(as: Result.self),
          generations: generations,
          toolCalls: toolCalls
        )
      }

      prompt = .tools(try await agentToolResponses(for: generation.toolCalls))
      index += 1
    }

    throw EdgeToolsAgentError.maximumTurnsExceeded(maximumTurns!)
  }
}

#if Needle2
  // MARK: - Needle 2 Loops

  public struct Needle2LoopResponse: Sendable {
    public let steps: [Needle2LoopStep]
    public let terminationCause: Needle2LoopTerminationCause

    public init(
      steps: [Needle2LoopStep],
      terminationCause: Needle2LoopTerminationCause
    ) {
      self.steps = steps
      self.terminationCause = terminationCause
    }
  }

  public struct Needle2LoopStep: Sendable {
    public let generation: EdgeToolsSessionGeneration
    public let toolResponses: [EdgeToolsValue]

    public init(
      generation: EdgeToolsSessionGeneration,
      toolResponses: [EdgeToolsValue]
    ) {
      self.generation = generation
      self.toolResponses = toolResponses
    }
  }

  public enum Needle2LoopTerminationCause: Hashable, Sendable {
    case responded
    case refused
    case noToolCalls
    case maximumTurnsReached
  }

  extension EdgeToolsSession where Engine: Needle2SessionEngine {
    @concurrent
    public func runLoop(
      prompt: String,
      context: Engine.Context? = nil,
      maximumTurns: Int = 8,
      parameters: @escaping @Sendable (Int) -> Engine.GenerateParameters = { _ in .default },
      shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool = { _ in true }
    ) async throws -> Needle2LoopResponse {
      precondition(
        (1...8).contains(maximumTurns),
        "Needle 2 supports between 1 and 8 loop turns."
      )

      let context = self.resolveContext(context)
      var prompt = Needle2Prompt.user(prompt)
      var steps = [Needle2LoopStep]()

      for turn in 0..<maximumTurns {
        let generation = try await self.generate(
          prompt: prompt,
          context: context,
          parameters: parameters(turn),
          shouldInvokeTools: shouldInvokeTools
        )
        guard generation.engineGeneration.metadata.needle2ResponseType == "call",
          !generation.toolCalls.isEmpty
        else {
          steps.append(Needle2LoopStep(generation: generation, toolResponses: []))
          return Needle2LoopResponse(
            steps: steps,
            terminationCause: needle2LoopTerminationCause(for: generation)
          )
        }

        let toolResponses = try await agentToolResponses(for: generation.toolCalls).map(\.response)
        steps.append(Needle2LoopStep(generation: generation, toolResponses: toolResponses))
        guard turn < maximumTurns - 1 else {
          return Needle2LoopResponse(
            steps: steps,
            terminationCause: .maximumTurnsReached
          )
        }
        prompt = .toolResponses(toolResponses)
      }

      fatalError("Needle 2 loop ended without a termination cause.")
    }
  }
#endif

// MARK: - EdgeToolsSessionTool

private protocol _SessionTool: AnyObject, Sendable {
  var erasedTool: any EdgeTool { get }
  var erasedName: String { get }
  var erasedDefinition: EdgeToolDefinition { get }
  func call(id: EdgeToolCallID, rawInput: EdgeToolsValue) -> AnyEdgeToolCall?
}

private final class SessionToolBox<Tool: EdgeTool>: _SessionTool {
  let tool: Tool
  let encodeOutput: (@Sendable (Tool.Output) throws -> EdgeToolsValue)?

  var erasedTool: any EdgeTool { self.tool }
  var erasedName: String { self.tool.name }
  var erasedDefinition: EdgeToolDefinition { self.tool.definition }

  init(
    _ tool: Tool,
    encodeOutput: (@Sendable (Tool.Output) throws -> EdgeToolsValue)? = nil
  ) {
    self.tool = tool
    self.encodeOutput = encodeOutput
  }

  func call(id: EdgeToolCallID, rawInput: EdgeToolsValue) -> AnyEdgeToolCall? {
    guard let call = try? EdgeToolCall(id: id, tool: self.tool, rawInput: rawInput) else {
      return nil
    }
    guard let encodeOutput else {
      return AnyEdgeToolCall.erasing(call)
    }
    return AnyEdgeToolCall.erasing(call, encodeOutput: encodeOutput)
  }
}

public struct EdgeToolsSessionTool: Sendable {
  private let base: any _SessionTool

  public var tool: any EdgeTool { self.base.erasedTool }
  public var name: String { self.base.erasedName }
  public var definition: EdgeToolDefinition { self.base.erasedDefinition }

  public init<Tool>(_ tool: Tool) where Tool: EdgeTool, Tool.Output: ConvertibleToEdgeToolsValue {
    self.base = SessionToolBox(tool, encodeOutput: { $0.edgeToolsValue })
  }

  public init<Tool>(
    _ tool: Tool,
    encodeOutput: @escaping @Sendable (Tool.Output) throws -> EdgeToolsValue
  ) where Tool: EdgeTool {
    self.base = SessionToolBox(tool, encodeOutput: encodeOutput)
  }

  public init(_ tool: some EdgeTool) {
    self.base = SessionToolBox(tool)
  }

  func call(id: EdgeToolCallID, rawInput: EdgeToolsValue) -> AnyEdgeToolCall? {
    self.base.call(id: id, rawInput: rawInput)
  }
}

// MARK: - EdgeToolsToolBuilder

@resultBuilder
public enum EdgeToolsToolBuilder {
  public static func buildExpression<Tool>(_ tool: Tool) -> EdgeToolsSessionTool
  where Tool: EdgeTool, Tool.Output: ConvertibleToEdgeToolsValue {
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

  public var text: String {
    self.engineGeneration.text
  }

  public var parts: [EdgeToolsGenerationPart] {
    self.engineGeneration.parts
  }

  public var reasoning: [String] {
    self.engineGeneration.reasoning
  }

  public func decoded<Response: ConvertibleFromEdgeToolsValue>(
    as type: Response.Type
  ) throws -> Response {
    return try Response(edgeToolsValue: EdgeToolsValue(json: self.text))
  }
}

// MARK: - Stream

public final class EdgeToolsSessionStream: Sendable, Identifiable {
  public typealias Element = EdgeToolCallCollection.Element

  @nonexhaustive
  public enum Event: Sendable {
    case token(EdgeToolsToken)
    case part(EdgeToolsGenerationPart)
    case finish(Result<EdgeToolsSessionGeneration, any Error>)
  }

  private struct State {
    var task: Task<EdgeToolsSessionGeneration, any Error>?
    var hasStarted = false
    var result: Result<EdgeToolsSessionGeneration, any Error>?
    var wasStoppedBeforeGeneration = false
    var stop: (@Sendable () -> Void)?
    var toolCalls = EdgeToolCallCollection()
    var events = [Event]()
    var eventSubscribers = [Int: @Sendable (Event) -> Void]()
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
  public func onEvent(
    _ body: @escaping @Sendable (Event) -> Void
  ) -> EdgeToolsSubscription {
    self.subscribe(onEvent: body)
  }

  public func onToken(
    _ body: @escaping @Sendable (EdgeToolsToken) -> Void
  ) -> EdgeToolsSubscription {
    self.onEvent {
      guard case .token(let token) = $0 else { return }
      body(token)
    }
  }

  public func onToolCall(
    _ body: @escaping @Sendable (Element) -> Void
  ) -> EdgeToolsSubscription {
    let deliveredCounts = RawToolCallDeliveryState()
    return self.onPart { part in
      guard case .toolCall(let rawCall) = part else { return }
      let calls = Array(self.toolCalls.filter { $0.rawValue == rawCall })
      if let index = deliveredCounts.nextIndex(for: rawCall, elementCount: calls.count) {
        body(calls[index])
      }
    }
  }

  public func onPart(
    _ body: @escaping @Sendable (EdgeToolsGenerationPart) -> Void
  ) -> EdgeToolsSubscription {
    self.onEvent {
      guard case .part(let part) = $0 else { return }
      body(part)
    }
  }

  public func onReasoning(
    _ body: @escaping @Sendable (String) -> Void
  ) -> EdgeToolsSubscription {
    self.onPart {
      guard case .reasoning(let reasoning) = $0 else { return }
      body(reasoning)
    }
  }

  public func onFinish(
    _ body: @escaping @Sendable (Result<EdgeToolsSessionGeneration, any Error>) -> Void
  ) -> EdgeToolsSubscription {
    self.onEvent {
      guard case .finish(let result) = $0 else { return }
      body(result)
    }
  }

  private func subscribe(
    onEvent: @escaping @Sendable (Event) -> Void
  ) -> EdgeToolsSubscription {
    let buffer = EventBuffer(deliver: onEvent)
    let (id, replayedEvents) = self.state.withLock { state in
      let id = state.nextID
      state.nextID += 1
      guard state.result == nil else { return (id, state.events) }
      state.eventSubscribers[id] = { event in buffer.append(event) }
      return (id, state.events)
    }
    buffer.replay(replayedEvents)
    return EdgeToolsSubscription { [self] in
      _ = self.state.withLock { subscribers in
        subscribers.eventSubscribers.removeValue(forKey: id)
      }
    }
  }

  private final class EventBuffer: Sendable {
    private struct State {
      var isReplaying = true
      var isDelivering = false
      var pending = [Event]()
    }

    private let state = Lock(State())
    private let deliver: @Sendable (Event) -> Void

    init(deliver: @escaping @Sendable (Event) -> Void) {
      self.deliver = deliver
    }

    func append(_ event: Event) {
      let shouldDrain = self.state.withLock { state in
        state.pending.append(event)
        guard !state.isReplaying, !state.isDelivering else { return false }
        state.isDelivering = true
        return true
      }
      if shouldDrain {
        self.drain()
      }
    }

    func replay(_ events: [Event]) {
      for event in events {
        self.deliver(event)
      }

      let shouldDrain = self.state.withLock { state in
        state.isReplaying = false
        guard !state.isDelivering, !state.pending.isEmpty else { return false }
        state.isDelivering = true
        return true
      }
      if shouldDrain {
        self.drain()
      }
    }

    private func drain() {
      while let event = self.dequeue() {
        self.deliver(event)
      }
    }

    private func dequeue() -> Event? {
      self.state.withLock { state in
        guard !state.pending.isEmpty else {
          state.isDelivering = false
          return nil
        }
        return state.pending.removeFirst()
      }
    }
  }
}

// MARK: - Generating

extension EdgeToolsSessionStream {
  fileprivate func start<Engine: EdgeToolsEngine>(
    session: EdgeToolsSession<Engine>,
    prompt: Engine.Prompt,
    context: Engine.Context?,
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
        context: context,
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
    context: Engine.Context?,
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
      onPart: { part in self.emit(part: part) }
    )
    do {
      let context = session.resolveContext(context)
      let generationTask = try session.engine.generate(
        prompt: prompt,
        tools: toolDefinitions,
        parameters: parameters,
        context: context,
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
    let event = Event.token(token)
    let subscribers = self.state.withLock { state in
      state.events.append(event)
      return Array(state.eventSubscribers.values)
    }
    for subscriber in subscribers {
      subscriber(event)
    }
  }

  private func emit(rawCall: EdgeRawToolCall) {
    guard let call = self.resolve(rawCall) else { return }
    self.withMutation(of: .toolCalls) {
      self.state.withLock { state in
        state.toolCalls.append(call)
      }
    }
    if self.shouldInvokeTools(call) {
      _ = Task { _ = try await call.output }
    }
  }

  private func emit(part: EdgeToolsGenerationPart) {
    if case .toolCall(let rawCall) = part {
      self.emit(rawCall: rawCall)
    }
    let event = Event.part(part)
    let subscribers = self.state.withLock { state in
      state.events.append(event)
      return Array(state.eventSubscribers.values)
    }
    for subscriber in subscribers {
      subscriber(event)
    }
  }

  private func finish(with result: Result<EdgeToolsSessionGeneration, any Error>) {
    let event = Event.finish(result)
    let completion = self.withMutation(of: .result) {
      self.state.withLock { state in
        state.result = result
        state.events.append(event)
        let completion = (state.onFinish, Array(state.eventSubscribers.values))
        state.eventSubscribers.removeAll()
        state.stop = nil
        // NB: Cleared so that the retain cycle through the session is broken deterministically.
        state.onFinish = nil
        return completion
      }
    }
    completion.0?()
    for subscriber in completion.1 {
      subscriber(event)
    }
  }

  private func resolve(_ rawCall: EdgeRawToolCall) -> AnyEdgeToolCall? {
    self.toolsByName[rawCall.name.snakeCased()]?
      .call(id: EdgeToolCallID(), rawInput: rawCall.arguments)
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
      let toolCallSubscription = self.onToolCall { continuation.yield($0) }
      let subscription = self.onEvent { event in
        switch event {
        case .token: break
        case .part: break
        case .finish(let result):
          switch result {
          case .success: continuation.finish()
          case .failure(let error): continuation.finish(throwing: error)
          }
        @unknown default: break
        }
      }
      continuation.onTermination = { _ in
        toolCallSubscription.cancel()
        subscription.cancel()
      }
      return AsyncIterator(base: stream.makeAsyncIterator())
    }

    private func makeTokenIterator() -> EdgeToolsSessionTokens.AsyncIterator {
      let (stream, continuation) = AsyncThrowingStream<EdgeToolsToken, any Error>.makeStream()
      let subscription = self.onEvent { event in
        switch event {
        case .token(let token): continuation.yield(token)
        case .part: break
        case .finish(let result):
          switch result {
          case .success: continuation.finish()
          case .failure(let error): continuation.finish(throwing: error)
          }
        @unknown default: break
        }
      }
      continuation.onTermination = { _ in subscription.cancel() }
      return EdgeToolsSessionTokens.AsyncIterator(base: stream.makeAsyncIterator())
    }
  }
#endif

// MARK: - Streaming

extension EdgeToolsSession {
  public func stream(
    prompt: Engine.Prompt,
    context: Engine.Context?,
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
      context: context,
      toolDefinitions: tools.map { $0.definition },
      parameters: parameters
    )
    return stream
  }

  @concurrent
  public func generate(
    prompt: Engine.Prompt,
    context: Engine.Context?,
    parameters: sending Engine.GenerateParameters = .default,
    shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool = { _ in true }
  ) async throws -> EdgeToolsSessionGeneration {
    try await self.stream(
      prompt: prompt,
      context: context,
      parameters: parameters,
      shouldInvokeTools: shouldInvokeTools
    )
    .finalGeneration
  }
}

// MARK: - RawToolCallDeliveryState

private final class RawToolCallDeliveryState: Sendable {
  private let counts = Lock([EdgeRawToolCall: Int]())

  func nextIndex(for rawCall: EdgeRawToolCall, elementCount: Int) -> Int? {
    self.counts.withLock { counts in
      let index = counts[rawCall, default: 0]
      guard index < elementCount else { return nil }
      counts[rawCall] = index + 1
      return index
    }
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

private func agentToolResponses(
  for toolCalls: EdgeToolCallCollection
) async throws -> [EdgeToolsTranscript.ToolMessage] {
  try await withThrowingTaskGroup(of: (Int, EdgeToolsTranscript.ToolMessage).self) { group in
    for (index, toolCall) in toolCalls.enumerated() {
      group.addTask {
        guard let value = try await toolCall.outputValue() else {
          throw EdgeToolsAgentError.invalidToolOutput(toolCall.tool.name)
        }
        return (index, EdgeToolsTranscript.ToolMessage(name: toolCall.tool.name, response: value))
      }
    }

    var responses = Array<EdgeToolsTranscript.ToolMessage?>(
      repeating: nil,
      count: toolCalls.count
    )
    for try await (index, response) in group {
      responses[index] = response
    }
    return responses.compactMap { $0 }
  }
}

#if Needle2
  private func needle2LoopTerminationCause(
    for generation: EdgeToolsSessionGeneration
  ) -> Needle2LoopTerminationCause {
    switch generation.engineGeneration.metadata.needle2ResponseType {
    case "respond": .responded
    case "refuse": .refused
    default: .noToolCalls
    }
  }
#endif
