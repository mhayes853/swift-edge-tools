import _Concurrency

// MARK: - AnyGenerationTask

public final class AnyGenerationTask: EdgeToolsEngineGenerationTask {
  public struct Stopper: Sendable {
    fileprivate let state: StopperState

    public var isStopped: Bool {
      self.state.isStopped
    }

    fileprivate init(state: StopperState) {
      self.state = state
    }

    public func stop() {
      self.state.stop()
    }
  }

  private let task: Task<EdgeToolsEngineGeneration, any Error>
  private let stopper: Stopper

  public init(
    operation: sending @escaping (Stopper) async throws -> EdgeToolsEngineGeneration
  ) {
    let stopper = Stopper(state: StopperState())
    self.stopper = stopper
    self.task = Task {
      guard !stopper.isStopped else { return .empty }
      return try await operation(stopper)
    }
  }

  public var value: EdgeToolsEngineGeneration {
    get async throws { try await self.task.cancellableValue }
  }

  public func stop() {
    self.stopper.stop()
  }
}

// MARK: - StopperState

private final class StopperState: Sendable {
  private let stopped = Lock(false)

  var isStopped: Bool {
    self.stopped.withLock { $0 }
  }

  func stop() {
    self.stopped.withLock { $0 = true }
  }
}
