#if Atomics && canImport(Atomics)
  import Atomics

  final class AtomicGenerationTask: EdgeToolsEngineGenerationTask {
    enum State: UInt8 {
      case queued
      case running
      case stopped
    }

    private let task: Task<EdgeToolsEngineGeneration, any Error>
    private let state: ManagedAtomic<UInt8>

    init(
      task: sending Task<EdgeToolsEngineGeneration, any Error>,
      state: ManagedAtomic<UInt8>
    ) {
      self.task = task
      self.state = state
    }

    var value: EdgeToolsEngineGeneration {
      get async throws { try await self.task.cancellableValue }
    }

    func stop() {
      let previousState = self.state.exchange(
        State.stopped.rawValue,
        ordering: .relaxed
      )
      if previousState == State.queued.rawValue {
        self.task.cancel()
      }
    }
  }
#endif
