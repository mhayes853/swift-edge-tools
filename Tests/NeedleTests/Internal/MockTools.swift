import Needle

// MARK: - Tools

struct EchoTool: NeedleTool {
  typealias Input = String
  typealias Output = String

  let name = "echo"
  let description = "Returns a fixed string."

  func invoke(input: String) async throws -> sending String {
    "echo: \(input)"
  }
}

struct ThrowingTool: NeedleTool {
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

final class CountingTool: NeedleTool, Sendable {
  typealias Input = String
  typealias Output = String

  let name: String
  let description = "Counts invocations."
  let output: String
  private let counter = Lock(0)

  init(name: String, output: String) {
    self.name = name
    self.output = output
  }

  var invokeCount: Int {
    self.counter.withLock { $0 }
  }

  func invoke(input: String) async throws -> sending String {
    self.counter.withLock { $0 += 1 }
    return self.output
  }
}

struct DelayedCountingTool: NeedleTool {
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

struct CountingThrowingTool: NeedleTool {
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

struct CancellableTool: NeedleTool {
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

struct ProbeTool: NeedleTool {
  typealias Input = String
  typealias Output = String

  let name = "probe"
  let description = "Tracks concurrency."
  let probe: ParallelProbe

  func invoke(input: String) async throws -> sending String {
    self.probe.enter()
    try await Task.sleep(for: .milliseconds(50))
    self.probe.exit()
    return "done"
  }
}

// MARK: - Errors

struct ToolError: Error, Equatable {
  let message: String
}

// MARK: - Concurrency Helpers

final class ParallelProbe: Sendable {
  private let state = Lock((current: 0, max: 0))

  func enter() {
    self.state.withLock { state in
      state.current += 1
      state.max = Swift.max(state.max, state.current)
    }
  }

  func exit() {
    self.state.withLock { $0.current -= 1 }
  }

  var maxConcurrent: Int {
    self.state.withLock { $0.max }
  }
}

final class AtomicCounter: Sendable {
  private let counter = Lock(0)

  var value: Int {
    self.counter.withLock { $0 }
  }

  func increment() {
    self.counter.withLock { $0 += 1 }
  }
}

// MARK: - NeedleToolCallStatus

extension NeedleToolCallStatus {
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

// MARK: - NeedleSessionStream.Status

extension NeedleSessionStream.Status {
  var isAwaitingExecution: Bool {
    switch self {
    case .awaitingExecution: true
    default: false
    }
  }

  var isFinished: Bool {
    switch self {
    case .finished: true
    default: false
    }
  }

  var result: Result<NeedleSessionGeneration, any Error>? {
    switch self {
    case .finished(let result): result
    default: nil
    }
  }
}