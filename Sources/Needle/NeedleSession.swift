// MARK: - NeedleSession

public final class NeedleSession<Engine: NeedleEngine>: Sendable {
  public var isResponding: Bool {
    false
  }

  public init(engine: sending Engine, instructions: String) {

  }
}

// MARK: - Prefill

extension NeedleSession {
  @discardableResult
  public func prefill(promptPrefix: String) async throws -> NeedlePrefillMetrics {
    fatalError()
  }

  @discardableResult
  public func prefill(
    tools: (some NeedleStaticToolsCollection).Type,
    prompt: String
  ) async throws -> NeedlePrefillMetrics {
    fatalError()
  }
}

// MARK: - Reset

extension NeedleSession {
  public func reset() {
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
    with prompt: String
  ) async throws -> NeedleSessionDynamicGeneration {
    fatalError()
  }

  public func generate<Collection: NeedleStaticToolsCollection>(
    tools: Collection.Type,
    with prompt: String,
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
    with prompt: String
  ) -> NeedleSessionStream<NeedleDynamicToolCalls> {
    fatalError()
  }

  public func stream<Collection: NeedleStaticToolsCollection>(
    tools: Collection.Type,
    with prompt: String
  ) -> NeedleSessionStream<NeedleStaticToolCalls<Collection>> {
    fatalError()
  }
}
