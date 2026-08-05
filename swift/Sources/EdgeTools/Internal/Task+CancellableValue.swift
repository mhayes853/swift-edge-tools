import _Concurrency

#if $Embedded
  extension Task where Success == Never, Failure == Never {
    package static func checkCancellation() throws {
      if Task.isCancelled { throw _Concurrency.CancellationError() }
    }
  }
#endif

extension Task where Success: Sendable, Failure: Error {
  package var cancellableValue: Success {
    get async throws(Failure) {
      do {
        return try await withTaskCancellationHandler {
          try await self.value
        } onCancel: {
          self.cancel()
        }
      } catch {
        guard let failure = error as? Failure else {
          preconditionFailure("Task produced an error outside its declared failure type.")
        }
        throw failure
      }
    }
  }
}
