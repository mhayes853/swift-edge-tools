import EdgeToolsCore
import _Concurrency

#if !$Embedded
  // MARK: - EdgeToolsTokenSequence

  /// Replays emitted tokens and follows new ones until generation finishes.
  public struct EdgeToolsTokenSequence: AsyncSequence, Sendable {
    public typealias Element = EdgeToolsToken

    public struct AsyncIterator: AsyncIteratorProtocol {
      var base: AsyncThrowingStream<Element, any Error>.AsyncIterator

      public mutating func next() async throws -> Element? {
        try await self.base.next()
      }
    }

    let iterator: @Sendable () -> AsyncIterator

    public func makeAsyncIterator() -> AsyncIterator {
      self.iterator()
    }
  }

  // MARK: - EdgeToolsEventSequence

  /// Replays stream events, including the final completion event.
  public struct EdgeToolsEventSequence<Event: Sendable>: AsyncSequence, Sendable {
    public typealias Element = Event

    public struct AsyncIterator: AsyncIteratorProtocol {
      var base: AsyncStream<Event>.AsyncIterator

      public mutating func next() async -> Event? {
        await self.base.next()
      }
    }

    let iterator: @Sendable () -> AsyncIterator

    public func makeAsyncIterator() -> AsyncIterator {
      self.iterator()
    }
  }
#endif
