public struct EdgeToolsLogitsProcessor<Prompt, Logits: ~Copyable & ~Escapable> {
  private let promptBody: (Prompt) -> Void
  private let processBody: nonisolated(nonsending) (inout Logits) async throws -> Void
  private let didSampleBody: (EdgeToolsToken.ID) -> Void

  public init(
    prompt: @escaping (Prompt) -> Void = { _ in },
    process: nonisolated(nonsending) @escaping (inout Logits) async throws -> Void,
    didSample: @escaping (EdgeToolsToken.ID) -> Void = { _ in }
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
