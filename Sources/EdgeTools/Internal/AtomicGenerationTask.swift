#if canImport(Atomics)
  import Atomics

  final class AtomicGenerationTask: EdgeToolEngineGenerationTask {
    private let task: Task<EdgeToolEngineGeneration, any Error>
    private let isStopped: ManagedAtomic<Bool>

    init(
      task: sending Task<EdgeToolEngineGeneration, any Error>,
      isStopped: ManagedAtomic<Bool>
    ) {
      self.task = task
      self.isStopped = isStopped
    }

    var value: EdgeToolEngineGeneration {
      get async throws { try await self.task.cancellableValue }
    }

    func stop() {
      self.isStopped.store(true, ordering: .relaxed)
    }
  }
#endif
