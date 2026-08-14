import Atomics
import _Concurrency

// MARK: - AnyGenerationTask

public final class AnyGenerationTask: EdgeToolsEngineGenerationTask {
  public struct Stopper: Sendable {
    fileprivate let stopped: ManagedAtomic<Bool>

    public var isStopped: Bool {
      self.stopped.load(ordering: .relaxed)
    }

    fileprivate init(stopped: ManagedAtomic<Bool>) {
      self.stopped = stopped
    }

    public func stop() {
      self.stopped.store(true, ordering: .relaxed)
    }
  }

  private let task: Task<EdgeToolsEngineGeneration, any Error>
  private let stopper: Stopper

  public init(
    operation: sending @escaping (Stopper) async throws -> EdgeToolsEngineGeneration
  ) {
    let stopper = Stopper(stopped: ManagedAtomic(false))
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
