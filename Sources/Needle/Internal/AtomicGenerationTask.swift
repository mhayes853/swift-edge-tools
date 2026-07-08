import Atomics

final class AtomicGenerationTask: NeedleEngineGenerationTask {
  private let task: Task<NeedleEngineGeneration, any Error>
  private let isStopped: ManagedAtomic<Bool>

  init(
    task: sending Task<NeedleEngineGeneration, any Error>,
    isStopped: ManagedAtomic<Bool>
  ) {
    self.task = task
    self.isStopped = isStopped
  }

  var value: NeedleEngineGeneration {
    get async throws { try await self.task.cancellableValue }
  }

  func stop() {
    self.isStopped.store(true, ordering: .relaxed)
  }
}
