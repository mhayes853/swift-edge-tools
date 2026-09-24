import EdgeToolsCore
import _Concurrency

#if !$Embedded
  import Observation
#endif

// MARK: - EdgeToolsGenerationStream

public final class EdgeToolsGenerationStream: Sendable, Identifiable {
  public typealias Element = EdgeToolCallCollection.Element

  @nonexhaustive
  public enum Event: Sendable {
    case token(EdgeToolsToken)
    case part(EdgeToolsGenerationPart)
    case finish(Result<EdgeToolsGeneration, any Error>)
  }

  private struct State {
    var task: Task<EdgeToolsGeneration, any Error>?
    var hasStarted = false
    var result: Result<EdgeToolsGeneration, any Error>?
    var wasStoppedBeforeGeneration = false
    var stop: (@Sendable () -> Void)?
    var toolCallOutcomes = [EdgeToolCallOutcome]()
    var events = [Event]()
    var eventSubscribers = [Int: @Sendable (Event) -> Void]()
    var nextID = 0
  }

  private let state = Lock(State())
  private let registrar = _ObservationRegistrar()
  private let toolsByName: [String: any EdgeTool]
  private let shouldInvokeTools: @Sendable (AnyEdgeToolCall) -> Bool

  public var isGenerating: Bool {
    self.result == nil
  }

  public var isFinished: Bool {
    self.result != nil
  }

  public var toolCalls: EdgeToolCallCollection {
    self.access(.toolCalls)
    return self.state.withLock {
      EdgeToolCallCollection($0.toolCallOutcomes.compactMap { $0.call })
    }
  }

  public var toolCallOutcomes: [EdgeToolCallOutcome] {
    self.access(.toolCalls)
    return self.state.withLock { $0.toolCallOutcomes }
  }

  public var result: Result<EdgeToolsGeneration, any Error>? {
    self.access(.result)
    return self.state.withLock { $0.result }
  }

  public var response: Result<String, any Error>? {
    self.result.map { $0.map { $0.response } }
  }

  public var finalGeneration: EdgeToolsGeneration {
    get async throws {
      let task = self.state.withLock { $0.task! }
      let generation = try await withTaskCancellationHandler {
        try await task.value
      } onCancel: {
        self.stop()
        task.cancel()
      }
      try Task.checkCancellation()
      return generation
    }
  }

  init(
    tools: [any EdgeTool],
    shouldInvokeTools: @escaping @Sendable (AnyEdgeToolCall) -> Bool
  ) {
    if let message = duplicateToolNameError(tools.map { $0.name }) {
      assertionFailure(message)
    }
    self.toolsByName = Dictionary(
      tools.map { ($0.name.snakeCased(), $0) },
      uniquingKeysWith: { _, tool in tool }
    )
    self.shouldInvokeTools = shouldInvokeTools
  }
}

// MARK: - Subscribing

extension EdgeToolsGenerationStream {
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
    _ body: @escaping @Sendable (Result<EdgeToolsGeneration, any Error>) -> Void
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

extension EdgeToolsGenerationStream {
  func start<Engine: EdgeToolsEngine>(
    engine: Engine,
    prompt: Engine.Prompt,
    context: Engine.Context,
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
        engine: engine,
        prompt: prompt,
        context: context,
        parameters: parameters
      )
    }
    self.state.withLock { $0.task = task }
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

extension EdgeToolsGenerationStream {
  private func runGeneration<Engine: EdgeToolsEngine>(
    engine: Engine,
    prompt: Engine.Prompt,
    context: Engine.Context,
    parameters: sending Engine.GenerateParameters
  ) async throws -> EdgeToolsGeneration {
    let wasStopped = self.state.withLock { $0.wasStoppedBeforeGeneration }
    if wasStopped {
      let generation = EdgeToolsGeneration(
        engineGeneration: .empty,
        toolCalls: EdgeToolCallCollection(),
        toolCallOutcomes: []
      )
      self.finish(with: .success(generation))
      return generation
    }

    let channel = EdgeToolsGenerationChannel(
      onToken: { token in self.emit(token: token) },
      onPart: { part in self.emit(part: part) }
    )
    do {
      let generationTask = try engine.generationTask(
        prompt: prompt,
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
      let generation = EdgeToolsGeneration(
        engineGeneration: engineGeneration,
        toolCalls: self.toolCalls,
        toolCallOutcomes: self.toolCallOutcomes
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
    let outcome = self.resolve(rawCall)
    self.withMutation(of: .toolCalls) {
      self.state.withLock { state in
        state.toolCallOutcomes.append(outcome)
      }
    }
    guard let call = outcome.call, self.shouldInvokeTools(call) else {
      return
    }
    _ = Task { _ = try await call.output }
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

  private func finish(with result: Result<EdgeToolsGeneration, any Error>) {
    let event = Event.finish(result)
    let subscribers = self.withMutation(of: .result) {
      self.state.withLock { state in
        state.result = result
        state.events.append(event)
        let subscribers = Array(state.eventSubscribers.values)
        state.eventSubscribers.removeAll()
        state.stop = nil
        return subscribers
      }
    }
    for subscriber in subscribers {
      subscriber(event)
    }
  }

  private func resolve(_ rawCall: EdgeRawToolCall) -> EdgeToolCallOutcome {
    guard let tool = self.toolsByName[rawCall.name.snakeCased()] else {
      return .unknownTool(rawCall)
    }
    do {
      return .resolved(
        try tool.toolCall(
          id: EdgeToolCallID(),
          arguments: rawCall.arguments
        )
      )
    } catch {
      return .invalidArguments(rawCall)
    }
  }
}

// MARK: - Async Sequences

#if !$Embedded
  public struct EdgeToolsGenerationTokens: AsyncSequence, Sendable {
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

  extension EdgeToolsGenerationStream: AsyncSequence {
    public struct AsyncIterator: AsyncIteratorProtocol {
      fileprivate var base: AsyncThrowingStream<Element, any Error>.AsyncIterator

      public mutating func next() async throws -> Element? {
        try await self.base.next()
      }
    }

    public var tokens: EdgeToolsGenerationTokens {
      EdgeToolsGenerationTokens(makeIterator: { self.tokenIterator() })
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

    private func tokenIterator() -> EdgeToolsGenerationTokens.AsyncIterator {
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
      return EdgeToolsGenerationTokens.AsyncIterator(base: stream.makeAsyncIterator())
    }
  }
#endif

// MARK: - Observation

extension EdgeToolsGenerationStream {
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
  extension EdgeToolsGenerationStream: Observable {}
#endif

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

// MARK: - Duplicate Tool Name Error

private func duplicateToolNameError(_ names: some Sequence<String>) -> String? {
  let grouped = Dictionary(grouping: names) { $0.snakeCased() }
  let duplicates = grouped.filter { $0.value.count > 1 }
  guard !duplicates.isEmpty else { return nil }
  return duplicates.sorted { $0.key < $1.key }
    .map { normalized, originals in
      "The names \(originals.sorted().map { "'\($0)'" }.joined(separator: " and ")) all normalize to '\(normalized)'."
    }
    .joined(separator: "\n")
}
