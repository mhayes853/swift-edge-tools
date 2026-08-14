public struct EdgeToolsLogitsProcessor<Prompt: Sendable, Logits: ~Copyable & ~Escapable>:
  Sendable
{
  private let promptBody: @Sendable (Prompt) -> Void
  private let processBody: nonisolated(nonsending) @Sendable (inout Logits) async throws -> Void
  private let didSampleBody: @Sendable (EdgeToolsToken.ID) -> Void

  public init(
    prompt: @escaping @Sendable (Prompt) -> Void = { _ in },
    process:
      nonisolated(nonsending) @escaping @Sendable (
        inout Logits
      ) async throws -> Void,
    didSample: @escaping @Sendable (EdgeToolsToken.ID) -> Void = { _ in }
  ) {
    self.promptBody = prompt
    self.processBody = process
    self.didSampleBody = didSample
  }

  public func prompt(_ prompt: Prompt) {
    self.promptBody(prompt)
  }

  public nonisolated(nonsending) func process(logits: inout Logits) async throws {
    try await self.processBody(&logits)
  }

  public func didSample(tokenId: EdgeToolsToken.ID) {
    self.didSampleBody(tokenId)
  }
}
