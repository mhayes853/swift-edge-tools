import Observation

// MARK: - NeedleSession

public final class NeedleSession<Engine: NeedleEngine>: Sendable, Observable {
  private struct State {
    let engine: Engine
    var systemPrompt: String
  }

  private let state: RecursiveLock<State>
  private let observationRegistrar = ObservationRegistrar()

  public var systemPrompt: String {
    get {
      self.observationRegistrar.access(self, keyPath: \.systemPrompt)
      return self.state.withLock { $0.systemPrompt }
    }
    set {
      self.observationRegistrar.withMutation(of: self, keyPath: \.systemPrompt) {
        self.state.withLock { $0.systemPrompt = newValue }
      }
    }
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
    self.withEngine { $0.reset() }
  }
}

// MARK: - Generate

public struct NeedleSessionGeneration: Sendable {
  public let engineGeneration: NeedleEngineGeneration
  public let toolCalls: NeedleToolCallCollection
}

extension NeedleSession {
  @concurrent
  public func generate(
    tools: [any NeedleTool],
    with prompt: String,
    overriding systemPrompt: String? = nil,
    parameters: Engine.GenerateParameters = .default,
    shouldInvokeTools: Bool = true
  ) async throws -> NeedleSessionGeneration {
    fatalError()
  }
}

// MARK: - Stream

public final class NeedleSessionStream: Sendable, Observable {
  public var finalGeneration: NeedleSessionGeneration {
    get async throws { fatalError() }
  }

  public var result: Result<NeedleSessionGeneration, any Error>? {
    nil
  }

  private let registrar = ObservationRegistrar()

  public func stop() {

  }
}

extension NeedleSessionStream: AsyncSequence {
  public struct AsyncIterator: AsyncIteratorProtocol {
    public func next() async throws -> NeedleToolCallCollection.Element? {
      nil
    }

    @available(iOS 18.0, macOS 15.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    public func next(
      isolation actor: isolated (any Actor)?
    ) async throws -> NeedleToolCallCollection.Element? {
      nil
    }
  }

  public func makeAsyncIterator() -> AsyncIterator {
    AsyncIterator()
  }
}

extension NeedleSessionStream {
  public struct Tokens: AsyncSequence {
    public struct AsyncIterator: AsyncIteratorProtocol {
      public func next() async throws -> NeedleToken? {
        nil
      }

      @available(iOS 18.0, macOS 15.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
      public func next(
        isolation actor: isolated (any Actor)?
      ) async throws -> NeedleToken? {
        nil
      }
    }

    public func makeAsyncIterator() -> AsyncIterator {
      AsyncIterator()
    }
  }

  public var tokens: Tokens {
    Tokens()
  }
}

extension NeedleSession {
  public func stream(
    tools: [any NeedleTool],
    with prompt: String,
    overriding systemPrompt: String? = nil,
    parameters: Engine.GenerateParameters = .default,
    shouldInvokeTools: Bool = true
  ) -> NeedleSessionStream {
    fatalError()
  }
}
