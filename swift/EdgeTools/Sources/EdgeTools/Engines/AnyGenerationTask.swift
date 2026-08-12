import Atomics

// MARK: - AnyGenerationTask

public final class AnyGenerationTask: EdgeToolsEngineGenerationTask {
  fileprivate enum State: UInt8 {
    case queued
    case running
    case stopped
  }

  public struct Stopper: Sendable {
    fileprivate let state: ManagedAtomic<UInt8>

    public var isStopped: Bool {
      self.state.load(ordering: .relaxed) == State.stopped.rawValue
    }

    fileprivate init(state: ManagedAtomic<UInt8>) {
      self.state = state
    }

    public func stop() {
      _ = self.exchangeStopped()
    }

    fileprivate func begin() -> Bool {
      self.state
        .compareExchange(
          expected: State.queued.rawValue,
          desired: State.running.rawValue,
          ordering: .relaxed
        )
        .exchanged
    }

    fileprivate func exchangeStopped() -> State {
      State(
        rawValue: self.state.exchange(
          State.stopped.rawValue,
          ordering: .relaxed
        )
      )!
    }
  }

  private let task: Task<EdgeToolsEngineGeneration, any Error>
  private let stopper: Stopper

  public init(
    operation: sending @escaping (Stopper) async throws -> EdgeToolsEngineGeneration
  ) {
    let state = ManagedAtomic(State.queued.rawValue)
    let stopper = Stopper(state: state)
    self.stopper = stopper
    self.task = Task {
      guard stopper.begin() else { return .empty }
      return try await operation(stopper)
    }
  }

  public var value: EdgeToolsEngineGeneration {
    get async throws { try await self.task.cancellableValue }
  }

  public func stop() {
    if self.stopper.exchangeStopped() == .queued {
      self.task.cancel()
    }
  }
}
