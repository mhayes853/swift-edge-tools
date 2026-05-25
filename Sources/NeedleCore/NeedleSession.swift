// MARK: - NeedleSession

public final class NeedleSession<Engine: NeedleEngine>: Sendable {
  public var isResponding: Bool {
    false
  }

  public init(engine: sending Engine) {

  }

  public func prefill(prompt: String, tools: sending [any NeedleTool]) async throws {

  }

  public func prefill(prompt: String, tools: (some NeedleStaticToolsCollection).Type) async throws {

  }

  public func invoke(
    prompt: String,
    tools: sending [any NeedleTool]
  ) async throws -> NeedleDynamicToolCalls {
    fatalError()
  }

  public func invoke<Collection: NeedleStaticToolsCollection>(
    prompt: String,
    tools: Collection.Type
  ) async throws -> NeedleStaticToolCalls<Collection> {
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
    prompt: String,
    tools: sending [any NeedleTool]
  ) -> NeedleSessionStream<NeedleDynamicToolCalls> {
    fatalError()
  }

  public func stream<Collection: NeedleStaticToolsCollection>(
    prompt: String,
    tools: Collection.Type
  ) -> NeedleSessionStream<NeedleStaticToolCalls<Collection>> {
    fatalError()
  }
}
