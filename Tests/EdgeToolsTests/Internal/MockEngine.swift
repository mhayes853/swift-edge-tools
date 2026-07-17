import EdgeTools
import Foundation

private typealias MockGenerationError = any Error
private typealias MockGenerationTaskValue = Task<EdgeToolsEngineGeneration, MockGenerationError>

final class MockEngine: EdgeToolsPrefillableEngine, Sendable {
  typealias Prompt = NeedlePrompt

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

  struct GenerateParameters: EdgeToolsEngineGenerateParameters {
    static let `default` = GenerateParameters()
  }

  enum Event: Sendable {
    case token(EdgeToolsToken)
    case stop
    case error(any Error)
    case finish
  }

  // NB: @unchecked Sendable is safe because all mutable state is guarded by condition.
  private final class GenerationStorage: @unchecked Sendable {
    private let condition = NSCondition()
    private var queue = [Event?]()
    private var isStopped = false

    init(events: [Event?] = []) {
      self.queue = events
    }

    func push(_ event: Event?) {
      self.condition.lock()
      self.queue.append(event)
      self.condition.signal()
      self.condition.unlock()
    }

    func stop() {
      self.condition.lock()
      self.isStopped = true
      self.condition.broadcast()
      self.condition.unlock()
    }

    func nextEvent() -> Event? {
      self.condition.lock()
      defer { self.condition.unlock() }

      while self.queue.isEmpty, !self.isStopped {
        self.condition.wait()
      }

      guard !self.isStopped else { return .stop }
      return self.queue.removeFirst()
    }
  }

  // NB: @unchecked Sendable is safe because all mutable state is guarded by condition.
  private final class Storage: @unchecked Sendable {
    private let condition = NSCondition()
    private var activeGenerations = [Int: GenerationStorage]()
    private var order = [Int]()
    private var nextID = 0
    private var queuedScripts = [[Event?]]()

    init(queuedScripts: [[Event?]] = []) {
      self.queuedScripts = queuedScripts
    }

    func makeGeneration() -> (Int, GenerationStorage) {
      self.condition.lock()
      defer { self.condition.unlock() }

      let id = self.nextID
      self.nextID += 1
      let script = self.queuedScripts.isEmpty ? [] : self.queuedScripts.removeFirst()
      let generation = GenerationStorage(events: script)
      self.activeGenerations[id] = generation
      self.order.append(id)
      return (id, generation)
    }

    func finishGeneration(id: Int) {
      self.condition.lock()
      self.activeGenerations.removeValue(forKey: id)
      self.order.removeAll { $0 == id }
      self.condition.unlock()
    }

    func push(_ event: Event?) {
      self.condition.lock()
      let generation = self.order.first.flatMap { self.activeGenerations[$0] }
      self.condition.unlock()
      generation?.push(event)
    }
  }

  private let storage: Storage
  private let _generateCallCount = Lock(0)
  private let tokenizeHandler: (@Sendable (NeedlePrompt, [EdgeToolDefinition]) -> [EdgeToolsToken])?
  private let _prefillHandler =
    Lock<(@Sendable (NeedlePrompt, [EdgeToolDefinition]) throws -> EdgeToolsEnginePrefill)?>(nil)
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

  init() {
    let storage = Storage()
    self.storage = storage
    self.tokenizeHandler = nil
  }

  init(script: [Event]) {
    let storage = Storage(queuedScripts: [script.map { $0 } + [nil]])
    self.storage = storage
    self.tokenizeHandler = nil
  }

  init(tokenize: @escaping @Sendable (NeedlePrompt, [EdgeToolDefinition]) -> [EdgeToolsToken]) {
    let storage = Storage()
    self.storage = storage
    self.tokenizeHandler = tokenize
  }

  init(
    prefill:
      @escaping @Sendable (NeedlePrompt, [EdgeToolDefinition]) throws -> EdgeToolsEnginePrefill
  ) {
    let storage = Storage()
    self.storage = storage
    self.tokenizeHandler = nil
    self._prefillHandler.withLock { $0 = prefill }
  }

  static func live() -> MockEngine {
    MockEngine()
  }

  func tokenize(
    prompt: NeedlePrompt,
    tools: [EdgeToolDefinition]
  ) async throws -> [EdgeToolsToken] {
    self.tokenizeHandler?(prompt, tools) ?? []
  }

  func push(_ event: Event?) {
    self.storage.push(event)
  }

  func prefill(
    promptPrefix: NeedlePrompt,
    tools: [EdgeToolDefinition]
  ) async throws -> EdgeToolsEnginePrefill {
    let handler = self._prefillHandler.withLock { $0 }
    return try handler?(promptPrefix, tools)
      ?? EdgeToolsEnginePrefill(
        metrics: EdgeToolsPrefillMetrics(tokens: 0, duration: .zero)
      )
  }

  func generate(
    prompt: NeedlePrompt,
    tools: [EdgeToolDefinition] = [],
    parameters: GenerateParameters,
    channel: EdgeToolsGenerationChannel
  ) throws -> GenerationTask {
    self._generateCallCount.withLock { $0 += 1 }
    let (id, generationStorage) = self.storage.makeGeneration()
    let onStart = self._onGenerateStart.withLock { $0 }
    let onEnd = self._onGenerateEnd.withLock { $0 }
    let task = Task {
      var parser = NeedleToolCallParser()
      onStart?()
      defer {
        onEnd?()
        self.storage.finishGeneration(id: id)
      }
      var emittedTokens = [EdgeToolsToken]()
      var wasStopped = false
      var thrownError: (any Error)?

      try await withTaskCancellationHandler {
        while let event = generationStorage.nextEvent() {
          try Task.checkCancellation()
          switch event {
          case .token(let token):
            let rawToolCall = parser.accept(token: token)
            channel.emit(token: token)
            if let rawToolCall {
              channel.emit(toolCall: rawToolCall)
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
      let response = emittedTokens.map(\.stringValue).joined()
      return EdgeToolsEngineGeneration(
        prefillMetrics: EdgeToolsPrefillMetrics(tokens: 0, duration: .zero),
        decodeMetrics: EdgeToolsDecodeMetrics(
          tokens: emittedTokens.count,
          duration: .zero,
          durationToFirstToken: .zero
        ),
        wasStopped: wasStopped,
        tokens: emittedTokens,
        response: response
      )
    }
    return GenerationTask(task: task) {
      generationStorage.stop()
    }
  }
}
