// MARK: - NeedleSession

public final class NeedleSession<Engine: NeedleEngine>: Sendable {
  private struct State {
    var isResponding = false
    let engine: Engine
    var systemPrompt: String
  }

  public var isResponding: Bool {
    self.state.withLock { $0.isResponding }
  }

  private let state: RecursiveLock<State>

  public var systemPrompt: String {
    get { self.state.withLock { $0.systemPrompt } }
    set { self.state.withLock { $0.systemPrompt = newValue } }
  }

  public init(engine: sending Engine, systemPrompt: String = "") {
    self.state = RecursiveLock(State(engine: engine, systemPrompt: systemPrompt))
  }

  public func withEngine<T, E: Error>(
    perform body: (Engine) throws(E) -> sending T
  ) throws(E) -> sending T {
    try self.state.withLock { (state: inout sending State) throws(E) -> sending T in
      try body(state.engine)
    }
  }

  public func reset() {
    self.state.withLock {
      $0.engine.reset()
      $0.isResponding = false
    }
  }
}

// MARK: - Generate

public struct NeedleSessionDynamicGeneration: Sendable {
  public let prefillMetrics: NeedlePrefillMetrics
  public let decodeMetrics: NeedleDecodeMetrics
  public let toolCalls: NeedleDynamicToolCalls
}

public struct NeedleSessionStaticGeneration<Collection: NeedleStaticToolsCollection> {
  public let prefillMetrics: NeedlePrefillMetrics
  public let decodeMetrics: NeedleDecodeMetrics
  public let toolCalls: NeedleStaticToolCalls<Collection>
}

extension NeedleSession {
  public func generate(
    tools: sending [any NeedleTool],
    with prompt: String,
    parameters: Engine.GenerateParameters = .default
  ) async throws -> NeedleSessionDynamicGeneration {
    fatalError()
  }

  public func generate<Collection: NeedleStaticToolsCollection>(
    tools: Collection.Type,
    with prompt: String,
    parameters: Engine.GenerateParameters = .default
  ) async throws -> NeedleSessionStaticGeneration<Collection> {
    fatalError()
  }
}

// MARK: - Stream

public struct NeedleSessionStream<ToolCalls>: Sendable {
  public var finalValue: ToolCalls {
    get async throws { fatalError() }
  }

  public func stop() {

  }
}

extension NeedleSessionStream: AsyncSequence {
  public struct AsyncIterator: AsyncIteratorProtocol {
    public func next() async throws -> NeedleToken? {
      nil
    }

    @available(iOS 18.0, macOS 15.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public func next(isolation actor: isolated (any Actor)?) async throws -> NeedleToken? {
      nil
    }
  }

  public func makeAsyncIterator() -> AsyncIterator {
    AsyncIterator()
  }
}

extension NeedleSession {
  public func stream(
    tools: sending [any NeedleTool],
    with prompt: String,
    parameters: Engine.GenerateParameters = .default
  ) -> NeedleSessionStream<NeedleDynamicToolCalls> {
    fatalError()
  }

  public func stream<Collection: NeedleStaticToolsCollection>(
    tools: Collection.Type,
    with prompt: String,
    parameters: Engine.GenerateParameters = .default
  ) -> NeedleSessionStream<NeedleStaticToolCalls<Collection>> {
    fatalError()
  }
}
