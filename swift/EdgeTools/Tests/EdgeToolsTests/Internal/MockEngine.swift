import EdgeTools
import Foundation

private typealias MockGenerationError = any Error
private typealias MockGenerationTaskValue = Task<EdgeToolsEngineGeneration, MockGenerationError>

final class MockEngine: EdgeToolsPrefillableEngine, EdgeToolsTokenizingEngine, Sendable {
  typealias Prompt = TestPrompt

  final class Context: EdgeToolsEngineContext {
    let tools: [any EdgeTool]

    init(tools: [any EdgeTool]) {
      self.tools = tools
    }

    var id: ObjectIdentifier {
      ObjectIdentifier(self)
    }
  }

  final class GenerationTask: EdgeToolsEngineGenerationTask {
    private let task: MockGenerationTaskValue
    private let stopGeneration: @Sendable () -> Void

    fileprivate init(
      task: sending MockGenerationTaskValue,
      stopGeneration: @escaping @Sendable () -> Void
    ) {
      self.task = task
      self.stopGeneration = stopGeneration
    }

    var value: EdgeToolsEngineGeneration {
      get async throws {
        try await self.task.value
      }
    }

    func stop() {
      self.stopGeneration()
    }
  }

  struct GenerationConstraint: EdgeToolsSchemaGenerationConstraint, Hashable {
    typealias Grammar = Void
    typealias Context = Void

    var schema: EdgeToolsGenerationSchema?

    var toolCallRange: GrammarToolCallRange? { nil }

    static func schema(_ schema: EdgeToolsGenerationSchema) -> Self {
      Self(schema: schema)
    }

    func grammar(toolCallGrammar: consuming Void?, context: Void) {}
  }

  struct GenerateParameters: EdgeToolsConstrainedGenerateParameters {
    static let `default` = GenerateParameters()

    var constraint = GenerationConstraint(schema: nil)
    var maxTokens: Int? { nil }
  }

  enum Event: Sendable {
    case token(EdgeToolsToken)
    case stop
    case error(any Error)
    case finish
  }

  private final class GenerationStorage: Sendable {
    private struct State {
      var queue: [Event?]
      var continuation: CheckedContinuation<Event?, Never>?
      var isStopped = false
    }

    private enum NextEvent {
      case suspended
      case value(Event?)
    }

    private let state: Lock<State>

    init(events: [Event?] = []) {
      self.state = Lock(State(queue: events))
    }

    func push(_ event: Event?) {
      let continuation: CheckedContinuation<Event?, Never>? = self.state.withLock { state in
        guard let continuation = state.continuation else {
          state.queue.append(event)
          return nil
        }
        state.continuation = nil
        return continuation
      }
      continuation?.resume(returning: event)
    }

    func stop() {
      let continuation: CheckedContinuation<Event?, Never>? = self.state.withLock { state in
        state.isStopped = true
        let continuation = state.continuation
        state.continuation = nil
        return continuation
      }
      continuation?.resume(returning: .stop)
    }

    func nextEvent() async -> Event? {
      await withCheckedContinuation { continuation in
        let nextEvent = self.state.withLock { state in
          if state.isStopped {
            return NextEvent.value(.stop)
          } else if !state.queue.isEmpty {
            return NextEvent.value(state.queue.removeFirst())
          } else {
            state.continuation = continuation
            return NextEvent.suspended
          }
        }
        if case .value(let event) = nextEvent {
          continuation.resume(returning: event)
        }
      }
    }
  }

  private final class Storage: Sendable {
    private struct State {
      var activeGenerations = [Int: GenerationStorage]()
      var order = [Int]()
      var nextID = 0
      var queuedScripts: [[Event?]]
    }

    private let state: Lock<State>

    init(queuedScripts: [[Event?]] = []) {
      self.state = Lock(State(queuedScripts: queuedScripts))
    }

    func makeGeneration() -> (Int, GenerationStorage) {
      self.state.withLock { state in
        let id = state.nextID
        state.nextID += 1
        let script = state.queuedScripts.isEmpty ? [] : state.queuedScripts.removeFirst()
        let generation = GenerationStorage(events: script)
        state.activeGenerations[id] = generation
        state.order.append(id)
        return (id, generation)
      }
    }

    func finishGeneration(id: Int) {
      self.state.withLock { state in
        state.activeGenerations.removeValue(forKey: id)
        state.order.removeAll { $0 == id }
      }
    }

    func push(_ event: Event?) {
      let generation = self.state.withLock { state in
        state.order.first.flatMap { state.activeGenerations[$0] }
      }
      generation?.push(event)
    }
  }

  private let storage: Storage
  private let _generateCallCount = Lock(0)
  private let _generationConstraints = Lock([GenerationConstraint]())
  private let _generationTools = Lock([[EdgeToolDefinition]]())
  private let tokenizeHandler: (@Sendable (TestPrompt, [EdgeToolDefinition]) -> [EdgeToolsToken])?
  private let _prefillHandler =
    Lock<(@Sendable (TestPrompt, [EdgeToolDefinition]) throws -> EdgeToolsEnginePrefill)?>(nil)
  private let _onGenerateStart = Lock<(@Sendable () -> Void)?>(nil)
  private let _onGenerateEnd = Lock<(@Sendable () -> Void)?>(nil)

  var onGenerateStart: (@Sendable () -> Void)? {
    get { self._onGenerateStart.withLock { $0 } }
    set { self._onGenerateStart.withLock { $0 = newValue } }
  }

  var onGenerateEnd: (@Sendable () -> Void)? {
    get { self._onGenerateEnd.withLock { $0 } }
    set { self._onGenerateEnd.withLock { $0 = newValue } }
  }

  var generateCallCount: Int {
    self._generateCallCount.withLock { $0 }
  }

  var generationTools: [[EdgeToolDefinition]] {
    self._generationTools.withLock { $0 }
  }

  var generationConstraints: [GenerationConstraint] {
    self._generationConstraints.withLock { $0 }
  }

  init() {
    let storage = Storage()
    self.storage = storage
    self.tokenizeHandler = nil
  }

  convenience init(script: [Event]) {
    self.init(scripts: [script])
  }

  init(scripts: [[Event]]) {
    let storage = Storage(queuedScripts: scripts.map { $0.map { $0 } + [nil] })
    self.storage = storage
    self.tokenizeHandler = nil
  }

  init(tokenize: @escaping @Sendable (TestPrompt, [EdgeToolDefinition]) -> [EdgeToolsToken]) {
    let storage = Storage()
    self.storage = storage
    self.tokenizeHandler = tokenize
  }

  init(
    prefill:
      @escaping @Sendable (TestPrompt, [EdgeToolDefinition]) throws -> EdgeToolsEnginePrefill
  ) {
    let storage = Storage()
    self.storage = storage
    self.tokenizeHandler = nil
    self._prefillHandler.withLock { $0 = prefill }
  }

  static func live() -> MockEngine {
    MockEngine()
  }

  func context() -> Context {
    Context(tools: [])
  }

  func context(tools: [any EdgeTool]) -> Context {
    Context(tools: tools)
  }

  func context(_ parameters: Void, tools: [any EdgeTool]) -> Context {
    Context(tools: tools)
  }

  func tokenize(
    prompt: TestPrompt,
    context: Context
  ) async throws -> [EdgeToolsToken] {
    self.tokenizeHandler?(prompt, context.tools.map(\.definition)) ?? []
  }

  func push(_ event: Event?) {
    self.storage.push(event)
  }

  func prefill(
    promptPrefix: TestPrompt,
    context: Context
  ) async throws -> EdgeToolsEnginePrefill {
    let handler = self._prefillHandler.withLock { $0 }
    return try handler?(promptPrefix, context.tools.map(\.definition))
      ?? EdgeToolsEnginePrefill(metrics: [:])
  }

  func generate(
    prompt: TestPrompt,
    parameters: GenerateParameters,
    context: Context,
    channel: sending EdgeToolsGenerationChannel
  ) throws -> GenerationTask {
    let tools = context.tools.map(\.definition)
    self._generateCallCount.withLock { $0 += 1 }
    self._generationConstraints.withLock { $0.append(parameters.constraint) }
    self._generationTools.withLock { $0.append(tools) }
    let (id, generationStorage) = self.storage.makeGeneration()
    let onStart = self._onGenerateStart.withLock { $0 }
    let onEnd = self._onGenerateEnd.withLock { $0 }
    let task = Task {
      var parser = TestGenerationParser()
      onStart?()
      defer {
        onEnd?()
        self.storage.finishGeneration(id: id)
      }
      var emittedTokens = [EdgeToolsToken]()
      var parts = [EdgeToolsGenerationPart]()
      var wasStopped = false
      var thrownError: (any Error)?

      try await withTaskCancellationHandler {
        while let event = await generationStorage.nextEvent() {
          try Task.checkCancellation()
          switch event {
          case .token(let token):
            channel.emit(token: token)
            for part in parser.accept(token: token) {
              parts.append(part)
              channel.emit(part: part)
            }
            emittedTokens.append(token)
          case .stop:
            wasStopped = true
          case .error(let error):
            thrownError = error
          case .finish:
            break
          }
          if wasStopped || thrownError != nil { break }
        }
      } onCancel: {
        generationStorage.stop()
      }

      if let error = thrownError { throw error }
      for part in parser.finish() {
        parts.append(part)
        channel.emit(part: part)
      }
      let response = emittedTokens.map(\.stringValue).joined()
      var metrics = EdgeToolsMetrics()
      metrics.decodeTokens = emittedTokens.count
      return EdgeToolsEngineGeneration(
        wasStopped: wasStopped,
        tokens: emittedTokens,
        response: response,
        parts: parts,
        metrics: metrics
      )
    }
    return GenerationTask(task: task) {
      generationStorage.stop()
    }
  }
}
