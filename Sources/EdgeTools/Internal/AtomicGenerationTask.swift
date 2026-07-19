#if Atomics && canImport(Atomics)
  import Atomics

  final class AtomicGenerationTask: EdgeToolsEngineGenerationTask {
    private let task: Task<EdgeToolsEngineGeneration, any Error>
    private let isStopped: ManagedAtomic<Bool>

    init(
      task: sending Task<EdgeToolsEngineGeneration, any Error>,
      isStopped: ManagedAtomic<Bool>
    ) {
      self.task = task
      self.isStopped = isStopped
    }

    var value: EdgeToolsEngineGeneration {
      get async throws { try await self.task.cancellableValue }
    }

    func stop() {
      self.isStopped.store(true, ordering: .relaxed)
    }
  }
#endif
