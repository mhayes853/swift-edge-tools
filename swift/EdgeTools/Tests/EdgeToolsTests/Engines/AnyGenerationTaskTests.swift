import CustomDump
import EdgeTools
import Testing

@Suite
struct `AnyGenerationTask tests` {
  @Test
  func `Stopper Observes A Running Stop`() async throws {
    let started = AsyncStream<Void>.makeStream()
    let resume = AsyncStream<Void>.makeStream()
    let task = AnyGenerationTask { stopper in
      started.continuation.yield()
      var iterator = resume.stream.makeAsyncIterator()
      await iterator.next()
      return EdgeToolsEngineGeneration(wasStopped: stopper.isStopped, tokens: [], response: "")
    }
    var iterator = started.stream.makeAsyncIterator()
    await iterator.next()

    task.stop()
    resume.continuation.yield()
    let generation = try await task.value

    expectNoDifference(generation.wasStopped, true)
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
