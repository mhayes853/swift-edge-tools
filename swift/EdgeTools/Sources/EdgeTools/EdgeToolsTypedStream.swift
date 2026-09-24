import EdgeToolsCore
import StreamParsing
import _Concurrency

#if !$Embedded
  import Observation
#endif

// MARK: - EdgeToolsTypedResult

/// A completed typed output with every generation and tool call that produced it.
public struct EdgeToolsTypedResult<Output: Sendable>: Sendable {
  /// The completed typed response.
  public let output: Output
  /// The generations produced while extracting or responding.
  public let generations: [EdgeToolsGeneration]
  /// Resolved tool calls across all turns.
  public let toolCalls: EdgeToolCallCollection

  /// All tool-call outcomes, including unknown tools and invalid arguments.
  public var toolCallOutcomes: [EdgeToolCallOutcome] {
    self.generations.flatMap(\.toolCallOutcomes)
  }

  public init(
    output: Output,
    generations: [EdgeToolsGeneration],
    toolCalls: EdgeToolCallCollection
  ) {
    self.output = output
    self.generations = generations
    self.toolCalls = toolCalls
  }
}

// MARK: - EdgeToolsTypedStreamError

/// An error converting a completed generation into its requested type.
public enum EdgeToolsTypedStreamError: Error, Sendable {
  /// The parsed partial cannot represent a complete output.
  case incompleteResponse
}

// MARK: - EdgeToolsTypedStream

/// Publishes partial output and tool activity while extracting or responding.
public final class EdgeToolsTypedStream<Output>: Sendable
where Output: StreamParseable & Sendable, Output.Partial: Sendable {
  @nonexhaustive
  public enum Event: Sendable {
    /// A generation turn has begun. Extraction uses turn zero.
    case turnStarted(Int)
    /// A token emitted during a generation turn.
    case token(turn: Int, value: EdgeToolsToken)
    /// A generation part emitted during a turn.
    case part(turn: Int, value: EdgeToolsGenerationPart)
    /// A snapshot of the current turn's parsed output. A tool-call turn's
    /// snapshots are provisional and are replaced when the next turn begins.
    case partial(turn: Int, value: Output.Partial)
    /// A tool call was parsed during the turn.
    case toolCall(turn: Int, outcome: EdgeToolCallOutcome)
    /// The generation turn has completed.
    case turnFinished(Int, EdgeToolsGeneration)
    /// The stream completed with a typed result or an error.
    case finish(Result<EdgeToolsTypedResult<Output>, any Error>)
  }

  /// Publishes typed stream activity from a custom producer.
  public struct Continuation: Sendable {
    private let stream: EdgeToolsTypedStream<Output>

    fileprivate init(stream: EdgeToolsTypedStream<Output>) {
      self.stream = stream
    }

    public func beginTurn(_ turn: Int) {
      self.stream.emit(.turnStarted(turn))
    }

    public func yield(token: EdgeToolsToken, turn: Int) {
      self.stream.emit(.token(turn: turn, value: token))
    }

    public func yield(part: EdgeToolsGenerationPart, turn: Int) {
      self.stream.emit(.part(turn: turn, value: part))
    }

    public func yield(partial: Output.Partial, turn: Int) {
      self.stream.emit(.partial(turn: turn, value: partial))
    }

    public func yield(toolCall: EdgeToolCallOutcome, turn: Int) {
      self.stream.emit(.toolCall(turn: turn, outcome: toolCall))
    }

    public func finishTurn(_ generation: EdgeToolsGeneration, turn: Int) {
      self.stream.emit(.turnFinished(turn, generation))
    }

    public func onStop(_ handler: @escaping @Sendable () -> Void) {
      self.stream.setStopHandler(handler)
    }

    public var isStopped: Bool {
      self.stream.state.withLock { $0.stopRequested }
    }

    func generation(
      _ raw: EdgeToolsGenerationStream,
      turn: Int
    ) async throws -> (EdgeToolsGeneration, EdgeToolsTypedTurnParser<Output>) {
      try await self.stream.generation(raw, turn: turn)
    }
  }

  private enum Delivery: Sendable {
    case event(Event)
    case completion(Result<EdgeToolsTypedResult<Output>, any Error>)
  }

  private struct StopAction {
    let task: Task<EdgeToolsTypedResult<Output>, any Error>?
    let generation: EdgeToolsGenerationStream?
    let handler: (@Sendable () -> Void)?
  }

  private struct State {
    var task: Task<EdgeToolsTypedResult<Output>, any Error>?
    var result: Result<EdgeToolsTypedResult<Output>, any Error>?
    var activeGeneration: EdgeToolsGenerationStream?
    var stopHandler: (@Sendable () -> Void)?
    var stopRequested = false
    var toolCallOutcomes = [EdgeToolCallOutcome]()
    var deliveries = [Delivery]()
    var subscribers = [Int: @Sendable (Delivery) -> Void]()
    var nextID = 0
  }

  private let state = Lock(State())
  private let registrar = _ObservationRegistrar()

  public var isGenerating: Bool { self.result == nil }

  public var isFinished: Bool { self.result != nil }

  /// Resolved tool calls received across all turns.
  public var toolCalls: EdgeToolCallCollection {
    self.access(.toolCalls)
    return self.state.withLock {
      EdgeToolCallCollection($0.toolCallOutcomes.compactMap(\.call))
    }
  }

  /// Tool-call outcomes received across all turns.
  public var toolCallOutcomes: [EdgeToolCallOutcome] {
    self.access(.toolCalls)
    return self.state.withLock { $0.toolCallOutcomes }
  }

  /// The current completion, if the stream has finished.
  public var result: Result<EdgeToolsTypedResult<Output>, any Error>? {
    self.access(.result)
    return self.state.withLock { $0.result }
  }

  /// The completed output and generation history, available without iterating events.
  public var finalResult: EdgeToolsTypedResult<Output> {
    get async throws {
      let task = self.state.withLock { $0.task! }
      let result = try await withTaskCancellationHandler {
        try await task.value
      } onCancel: {
        self.stop()
      }
      try Task.checkCancellation()
      return result
    }
  }

  /// Stops the active generation and prevents further turns.
  public func stop() {
    let action: StopAction = self.state.withLock { state in
      guard state.result == nil, !state.stopRequested else {
        return StopAction(task: nil, generation: nil, handler: nil)
      }
      state.stopRequested = true
      return StopAction(
        task: state.task,
        generation: state.activeGeneration,
        handler: state.stopHandler
      )
    }
    action.generation?.stop()
    action.handler?()
    action.task?.cancel()
  }

  /// Receives events in order, including events published before subscribing.
  /// Errors are also reported by ``finalResult`` or by async iteration.
  public func onEvent(
    _ body: @escaping @Sendable (Event) -> Void
  ) -> EdgeToolsSubscription {
    self.subscribe { delivery in
      guard case .event(let event) = delivery else { return }
      body(event)
    }
  }

  public func onToken(
    _ body: @escaping @Sendable (EdgeToolsToken) -> Void
  ) -> EdgeToolsSubscription {
    self.onEvent {
      guard case .token(_, let token) = $0 else { return }
      body(token)
    }
  }

  public func onPart(
    _ body: @escaping @Sendable (EdgeToolsGenerationPart) -> Void
  ) -> EdgeToolsSubscription {
    self.onEvent {
      guard case .part(_, let part) = $0 else { return }
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

  public func onPartial(
    _ body: @escaping @Sendable (Output.Partial) -> Void
  ) -> EdgeToolsSubscription {
    self.onEvent {
      guard case .partial(_, let partial) = $0 else { return }
      body(partial)
    }
  }

  public func onToolCall(
    _ body: @escaping @Sendable (AnyEdgeToolCall) -> Void
  ) -> EdgeToolsSubscription {
    self.onToolCallOutcome { outcome in
      if let call = outcome.call {
        body(call)
      }
    }
  }

  public func onToolCallOutcome(
    _ body: @escaping @Sendable (EdgeToolCallOutcome) -> Void
  ) -> EdgeToolsSubscription {
    self.onEvent {
      guard case .toolCall(_, let outcome) = $0 else { return }
      body(outcome)
    }
  }

  public func onFinish(
    _ body: @escaping @Sendable (Result<EdgeToolsTypedResult<Output>, any Error>) -> Void
  ) -> EdgeToolsSubscription {
    self.onEvent {
      guard case .finish(let result) = $0 else { return }
      body(result)
    }
  }

  public init(
    operation: @escaping @Sendable (Continuation) async throws -> EdgeToolsTypedResult<Output>
  ) {
    let task = Task {
      do {
        let result = try await operation(Continuation(stream: self))
        try Task.checkCancellation()
        guard self.finishSuccessfully(with: result) else {
          throw CancellationError()
        }
        return result
      } catch {
        self.finishWithError(error)
        throw error
      }
    }
    self.state.withLock { $0.task = task }
  }

  func emit(_ event: Event) {
    let append = {
      self.state.withLock { state in
        guard state.result == nil else { return [@Sendable (Delivery) -> Void]() }
        if case .toolCall(_, let outcome) = event {
          state.toolCallOutcomes.append(outcome)
        }
        let delivery = Delivery.event(event)
        state.deliveries.append(delivery)
        return Array(state.subscribers.values)
      }
    }
    let subscribers: [@Sendable (Delivery) -> Void]
    if case .toolCall = event {
      subscribers = self.withMutation(of: .toolCalls, append)
    } else {
      subscribers = append()
    }
    for subscriber in subscribers {
      subscriber(.event(event))
    }
  }

  func generation(
    _ raw: EdgeToolsGenerationStream,
    turn: Int
  ) async throws -> (EdgeToolsGeneration, EdgeToolsTypedTurnParser<Output>) {
    self.setActiveGeneration(raw)
    let parser = EdgeToolsTypedTurnParser<Output>()
    let subscription = raw.onEvent { event in
      switch event {
      case .token(let token):
        self.emit(.token(turn: turn, value: token))
      case .toolCall(let outcome):
        self.emit(.toolCall(turn: turn, outcome: outcome))
      case .part(let part):
        self.emit(.part(turn: turn, value: part))
        if case .text(let text) = part, let partial = parser.append(text) {
          self.emit(.partial(turn: turn, value: partial))
        }
      case .finish: break
      @unknown default: break
      }
    }
    defer {
      subscription.cancel()
      self.clearActiveGeneration()
    }
    let generation = try await raw.finalGeneration
    try Task.checkCancellation()
    return (generation, parser)
  }

  private func setActiveGeneration(_ generation: EdgeToolsGenerationStream) {
    let shouldStop = self.state.withLock { state in
      state.activeGeneration = generation
      return state.stopRequested
    }
    if shouldStop {
      generation.stop()
    }
  }

  private func clearActiveGeneration() {
    self.state.withLock { $0.activeGeneration = nil }
  }

  private func setStopHandler(_ handler: @escaping @Sendable () -> Void) {
    let shouldStop = self.state.withLock { state in
      state.stopHandler = handler
      return state.stopRequested
    }
    if shouldStop {
      handler()
    }
  }

  private func finishSuccessfully(with result: EdgeToolsTypedResult<Output>) -> Bool {
    let event = Delivery.event(.finish(.success(result)))
    let completion = Delivery.completion(.success(result))
    let subscribers: [@Sendable (Delivery) -> Void]? = self.withMutation(of: .result) {
      self.state.withLock { state in
        guard !state.stopRequested else { return nil }
        state.result = .success(result)
        state.activeGeneration = nil
        state.deliveries.append(event)
        state.deliveries.append(completion)
        let subscribers = Array(state.subscribers.values)
        state.subscribers.removeAll()
        return subscribers
      }
    }
    guard let subscribers else { return false }
    for subscriber in subscribers {
      subscriber(event)
      subscriber(completion)
    }
    return true
  }

  private func finishWithError(_ error: any Error) {
    let event = Delivery.event(.finish(.failure(error)))
    let completion = Delivery.completion(.failure(error))
    let subscribers: [@Sendable (Delivery) -> Void]? = self.withMutation(of: .result) {
      self.state.withLock { state in
        guard state.result == nil else { return nil }
        state.result = .failure(error)
        state.activeGeneration = nil
        state.deliveries.append(event)
        state.deliveries.append(completion)
        let subscribers = Array(state.subscribers.values)
        state.subscribers.removeAll()
        return subscribers
      }
    }
    for subscriber in subscribers ?? [] {
      subscriber(event)
      subscriber(completion)
    }
  }

  private func subscribe(
    _ body: @escaping @Sendable (Delivery) -> Void
  ) -> EdgeToolsSubscription {
    let buffer = DeliveryBuffer(deliver: body)
    let (id, history) = self.state.withLock { state in
      let id = state.nextID
      state.nextID += 1
      if state.result == nil {
        state.subscribers[id] = { buffer.append($0) }
      }
      return (id, state.deliveries)
    }
    buffer.replay(history)
    return EdgeToolsSubscription { [self] in
      _ = self.state.withLock { $0.subscribers.removeValue(forKey: id) }
    }
  }

  private final class DeliveryBuffer: Sendable {
    private struct State {
      var isReplaying = true
      var isDelivering = false
      var pending = [Delivery]()
    }

    private let state = Lock(State())
    private let deliver: @Sendable (Delivery) -> Void

    init(deliver: @escaping @Sendable (Delivery) -> Void) {
      self.deliver = deliver
    }

    func append(_ delivery: Delivery) {
      let shouldDrain = self.state.withLock { state in
        state.pending.append(delivery)
        guard !state.isReplaying, !state.isDelivering else { return false }
        state.isDelivering = true
        return true
      }
      if shouldDrain {
        self.drain()
      }
    }

    func replay(_ deliveries: [Delivery]) {
      for delivery in deliveries {
        self.deliver(delivery)
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
      while let delivery = self.dequeue() {
        self.deliver(delivery)
      }
    }

    private func dequeue() -> Delivery? {
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

// MARK: - Async Sequence

#if !$Embedded
  extension EdgeToolsTypedStream: AsyncSequence {
    public typealias Element = Event

    public struct AsyncIterator: AsyncIteratorProtocol {
      fileprivate var base: AsyncThrowingStream<Event, any Error>.AsyncIterator

      public mutating func next() async throws -> Event? {
        try await self.base.next()
      }
    }

    /// A nonthrowing event sequence whose final event contains the completion result.
    public var events: AsyncStream<Event> {
      let (events, continuation) = AsyncStream<Event>.makeStream()
      let subscription = self.onEvent { event in
        continuation.yield(event)
        if case .finish = event {
          continuation.finish()
        }
      }
      continuation.onTermination = { _ in subscription.cancel() }
      return events
    }

    /// The tokens emitted across all turns.
    public var tokens: AsyncThrowingStream<EdgeToolsToken, any Error> {
      let (tokens, continuation) = AsyncThrowingStream<EdgeToolsToken, any Error>.makeStream()
      let subscription = self.onEvent { event in
        switch event {
        case .token(_, let token): continuation.yield(token)
        case .finish(let result):
          switch result {
          case .success: continuation.finish()
          case .failure(let error): continuation.finish(throwing: error)
          }
        default: break
        }
      }
      continuation.onTermination = { _ in subscription.cancel() }
      return tokens
    }

    public func makeAsyncIterator() -> AsyncIterator {
      let (events, continuation) = AsyncThrowingStream<Event, any Error>.makeStream()
      let subscription = self.subscribe { delivery in
        switch delivery {
        case .event(let event):
          continuation.yield(event)
        case .completion(.success):
          continuation.finish()
        case .completion(.failure(let error)):
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { _ in subscription.cancel() }
      return AsyncIterator(base: events.makeAsyncIterator())
    }

  }
#endif

// MARK: - Consuming Events

extension EdgeToolsTypedStream {
  /// Consumes events in order and returns the completed result on every platform.
  public func consume(
    _ body: @Sendable (Event) async throws -> Void
  ) async throws -> EdgeToolsTypedResult<Output> {
    let queue = EdgeToolsEventQueue<Event>()
    let subscription = self.onEvent { queue.append($0) }
    defer { subscription.cancel() }
    do {
      return try await withTaskCancellationHandler {
        while true {
          let event = try await queue.next()
          try Task.checkCancellation()
          try await body(event)
          if case .finish(let result) = event {
            return try result.get()
          }
        }
      } onCancel: {
        queue.cancel()
        self.stop()
      }
    } catch {
      self.stop()
      throw error
    }
  }
}

// MARK: - Observation

extension EdgeToolsTypedStream {
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
  extension EdgeToolsTypedStream: Observable {}
#endif

// MARK: - EdgeToolsTypedTurnParser

final class EdgeToolsTypedTurnParser<Output>: Sendable
where Output: StreamParseable & Sendable, Output.Partial: Sendable {
  private struct State: ~Copyable {
    var parser: PartialsStream<Output.Partial>
    var error: (any Error)?
    var sawText = false
  }

  private let state = Lock(State(parser: PartialsStream(from: .json())))

  func append(_ text: String) -> Output.Partial? {
    self.state.withLock { state in
      guard state.error == nil, !text.isEmpty else { return nil }
      state.sawText = true
      do {
        try state.parser.next(text.utf8)
        return state.parser.current
      } catch {
        state.error = error
        return nil
      }
    }
  }

  func complete(fallbackText: String) throws -> Output {
    let partial = try self.state.withLock { state in
      if let error = state.error {
        throw error
      }
      if !state.sawText {
        try state.parser.next(fallbackText.utf8)
      }
      return try state.parser.finish()
    }
    guard let output = Output(streamPartial: partial) else {
      throw EdgeToolsTypedStreamError.incompleteResponse
    }
    return output
  }
}
