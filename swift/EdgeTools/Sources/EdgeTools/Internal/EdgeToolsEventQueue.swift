import _Concurrency

final class EdgeToolsEventQueue<Element: Sendable>: Sendable {
  private struct State {
    var events = [Element]()
    var nextIndex = 0
    var waiter: UnsafeContinuation<Result<Element, any Error>, Never>?
    var failure: (any Error)?
  }

  private let state = Lock(State())

  func append(_ event: Element) {
    let waiter: UnsafeContinuation<Result<Element, any Error>, Never>? = self.state.withLock {
      state in
      guard state.failure == nil else { return nil }
      guard let waiter = state.waiter else {
        state.events.append(event)
        return nil
      }
      state.waiter = nil
      return waiter
    }
    waiter?.resume(returning: .success(event))
  }

  func next() async throws -> Element {
    let result: Result<Element, any Error> = await withUnsafeContinuation { continuation in
      let immediate: Result<Element, any Error>? = self.state.withLock { state in
        if let failure = state.failure {
          return .failure(failure)
        }
        guard state.nextIndex < state.events.count else {
          state.waiter = continuation
          return nil
        }
        let event = state.events[state.nextIndex]
        state.nextIndex += 1
        return .success(event)
      }
      if let immediate {
        continuation.resume(returning: immediate)
      }
    }
    return try result.get()
  }

  func cancel() {
    let waiter = self.state.withLock { state in
      state.failure = CancellationError()
      let waiter = state.waiter
      state.waiter = nil
      return waiter
    }
    waiter?.resume(returning: .failure(CancellationError()))
  }
}
