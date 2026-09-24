import EdgeToolsCore
import StreamParsing
import _Concurrency

// MARK: - EdgeToolsTypedResult

/// A completed typed output with every generation and tool call that produced it.
public struct EdgeToolsTypedResult<Output: Sendable>: Sendable {
  /// The completed typed response.
  public let output: Output
  /// The generations produced while extracting or responding.
  public let generations: [EdgeToolsGeneration]
  /// Resolved tool calls across all turns.
  public let toolCalls: EdgeToolCallCollection

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
where Output: EdgeToolsGenerable & StreamParseable & Sendable, Output.Partial: Sendable {
  @nonexhaustive
  public enum Event: Sendable {
    /// A generation turn has begun. Extraction uses turn zero.
    case turnStarted(Int)
    /// A snapshot of the current turn's parsed output. A tool-call turn's
    /// snapshots are provisional and are replaced when the next turn begins.
    case partial(turn: Int, value: Output.Partial)
    /// A tool call was parsed during the turn.
    case toolCall(turn: Int, outcome: EdgeToolCallOutcome)
    /// The generation turn has completed.
    case turnFinished(Int, EdgeToolsGeneration)
    /// The final response was converted into the requested output type.
    case finished(EdgeToolsTypedResult<Output>)
  }

  private enum Delivery: Sendable {
    case event(Event)
    case completion(Result<EdgeToolsTypedResult<Output>, any Error>)
  }

  private struct StopAction {
    let task: Task<EdgeToolsTypedResult<Output>, any Error>?
    let generation: EdgeToolsGenerationStream?
  }

  private struct State {
    var task: Task<EdgeToolsTypedResult<Output>, any Error>?
    var result: Result<EdgeToolsTypedResult<Output>, any Error>?
    var activeGeneration: EdgeToolsGenerationStream?
    var stopRequested = false
    var deliveries = [Delivery]()
    var subscribers = [Int: @Sendable (Delivery) -> Void]()
    var nextID = 0
  }

  private let state = Lock(State())

  /// The completed output and generation history, available without iterating events.
  public var result: EdgeToolsTypedResult<Output> {
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
      guard state.result == nil else {
        return StopAction(task: nil, generation: nil)
      }
      state.stopRequested = true
      return StopAction(task: state.task, generation: state.activeGeneration)
    }
    action.generation?.stop()
    action.task?.cancel()
  }

  /// Receives events in order, including events published before subscribing.
  /// Errors are reported by ``result`` or by async iteration.
  public func onEvent(
    _ body: @escaping @Sendable (Event) -> Void
  ) -> EdgeToolsSubscription {
    self.subscribe { delivery in
      guard case .event(let event) = delivery else { return }
      body(event)
    }
  }

  init(
    operation:
      @escaping @Sendable (EdgeToolsTypedStream<Output>) async throws ->
      EdgeToolsTypedResult<Output>
  ) {
    let task = Task {
      do {
        let result = try await operation(self)
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
    let subscribers = self.state.withLock { state in
      guard state.result == nil else { return [@Sendable (Delivery) -> Void]() }
      let delivery = Delivery.event(event)
      state.deliveries.append(delivery)
      return Array(state.subscribers.values)
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
    let subscription = raw.onPart { part in
      switch part {
      case .text(let text):
        if let partial = parser.append(text) {
          self.emit(.partial(turn: turn, value: partial))
        }
      case .toolCall:
        if let outcome = parser.nextToolCallOutcome(in: raw) {
          self.emit(.toolCall(turn: turn, outcome: outcome))
        }
      case .reasoning: break
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

  private func finishSuccessfully(with result: EdgeToolsTypedResult<Output>) -> Bool {
    let event = Delivery.event(.finished(result))
    let completion = Delivery.completion(.success(result))
    let subscribers: [@Sendable (Delivery) -> Void]? = self.state.withLock { state in
      guard !state.stopRequested else { return nil }
      state.result = .success(result)
      state.activeGeneration = nil
      state.deliveries.append(event)
      state.deliveries.append(completion)
      let subscribers = Array(state.subscribers.values)
      state.subscribers.removeAll()
      return subscribers
    }
    guard let subscribers else { return false }
    for subscriber in subscribers {
      subscriber(event)
      subscriber(completion)
    }
    return true
  }

  private func finishWithError(_ error: any Error) {
    let completion = Delivery.completion(.failure(error))
    let subscribers: [@Sendable (Delivery) -> Void]? = self.state.withLock { state in
      guard state.result == nil else { return nil }
      state.result = .failure(error)
      state.activeGeneration = nil
      state.deliveries.append(completion)
      let subscribers = Array(state.subscribers.values)
      state.subscribers.removeAll()
      return subscribers
    }
    for subscriber in subscribers ?? [] {
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

    /// Consumes events in order and returns the completed result.
    /// Stops generation if `body` throws or the consuming task is cancelled.
    public func consume(
      _ body: @Sendable (Event) async throws -> Void
    ) async throws -> EdgeToolsTypedResult<Output> {
      do {
        for try await event in self {
          try await body(event)
        }
        return try await self.result
      } catch {
        self.stop()
        throw error
      }
    }
  }
#endif

// MARK: - EdgeToolsTypedTurnParser

final class EdgeToolsTypedTurnParser<Output>: Sendable
where Output: EdgeToolsGenerable & StreamParseable & Sendable, Output.Partial: Sendable {
  private struct State: ~Copyable {
    var parser: PartialsStream<Output.Partial>
    var error: (any Error)?
    var sawText = false
    var nextToolCallIndex = 0
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

  func nextToolCallOutcome(in generation: EdgeToolsGenerationStream) -> EdgeToolCallOutcome? {
    self.state.withLock { state in
      let outcomes = generation.toolCallOutcomes
      guard state.nextToolCallIndex < outcomes.count else { return nil }
      defer { state.nextToolCallIndex += 1 }
      return outcomes[state.nextToolCallIndex]
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
