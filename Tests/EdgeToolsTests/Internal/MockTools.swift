import EdgeTools

// MARK: - Tools

struct EchoTool: EdgeTool {
  typealias Input = String
  typealias Output = String

  let name = "echo"
  let description = "Returns a fixed string."

  func invoke(input: String) async throws -> sending String {
    "echo: \(input)"
  }
}

struct ThrowingTool: EdgeTool {
  typealias Input = String
  typealias Output = String

  let name: String
  let description = "Always throws."
  let error: any Error

  init(name: String = "throwing", error: any Error) {
    self.name = name
    self.error = error
  }

  func invoke(input: String) async throws -> sending String {
    throw self.error
  }
}

struct DelayedCountingTool: EdgeTool {
  typealias Input = String
  typealias Output = String

  let name = "counting"
  let description = "Counts invocations after an optional delay."
  let duration: Duration
  let output: String
  let counter = AtomicCounter()

  var invokeCount: Int { self.counter.value }

  func invoke(input: String) async throws -> sending String {
    self.counter.increment()
    if self.duration > .zero {
      try await Task.sleep(for: self.duration)
    }
    return self.output
  }
}

struct CountingThrowingTool: EdgeTool {
  typealias Input = String
  typealias Output = String

  let name = "countingThrowing"
  let description = "Counts invocations and always throws."
  let duration: Duration
  let error: any Error
  let counter = AtomicCounter()

  var invokeCount: Int { self.counter.value }

  func invoke(input: String) async throws -> sending String {
    self.counter.increment()
    if self.duration > .zero {
      try await Task.sleep(for: self.duration)
    }
    throw self.error
  }
}

struct CancellableTool: EdgeTool {
  typealias Input = String
  typealias Output = String

  let name = "cancellable"
  let description = "Participates in cooperative cancellation."
  let duration: Duration

  func invoke(input: String) async throws -> sending String {
    try await Task.sleep(for: self.duration)
    return "done"
  }
}

// MARK: - Errors

struct ToolError: Error, Equatable {
  let message: String
}

// MARK: - Concurrency Helpers

final class AtomicCounter: Sendable {
  private let counter = Lock(0)

  var value: Int {
    self.counter.withLock { $0 }
  }

  func increment() {
    self.counter.withLock { $0 += 1 }
  }
}

// MARK: - EdgeToolCallStatus

extension EdgeToolCallStatus {
  var isIdle: Bool {
    switch self {
    case .idle: true
    default: false
    }
  }

  var isRunning: Bool {
    switch self {
    case .running: true
    default: false
    }
  }

  var isFinished: Bool {
    switch self {
    case .finished: true
    default: false
    }
  }

  var result: Result<Output, any Error>? {
    switch self {
    case .finished(let result): result
    default: nil
    }
  }
}
