extension Task where Success: Sendable, Failure: Error {
  package var cancellableValue: Success {
    get async throws {
      try await withTaskCancellationHandler {
        try await self.value
      } onCancel: {
        self.cancel()
      }
    }
  }
}
