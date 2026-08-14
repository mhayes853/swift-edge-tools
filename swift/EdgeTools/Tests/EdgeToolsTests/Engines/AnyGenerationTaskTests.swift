import CustomDump
import EdgeTools
import Testing

@Suite
struct `AnyGenerationTask tests` {
  @Test
  func `Stopper Observes A Running Stop`() async throws {
    let didStop = Lock(false)
    let started = AsyncStream<Void>.makeStream()
    let task = AnyGenerationTask { stopper in
      started.continuation.yield()
      while !stopper.isStopped {
        await Task.yield()
      }
      didStop.withLock { $0 = true }
      return .empty
    }
    var iterator = started.stream.makeAsyncIterator()
    await iterator.next()

    task.stop()
    _ = try await task.value

    expectNoDifference(didStop.withLock { $0 }, true)
  }

  @Test
  func `Multiple Values Await The Same Operation`() async throws {
    let invocationCount = Lock(0)
    let task = AnyGenerationTask { _ in
      invocationCount.withLock { $0 += 1 }
      return .empty
    }

    async let first = task.value
    async let second = task.value
    _ = try await (first, second)

    expectNoDifference(invocationCount.withLock { $0 }, 1)
  }
}
