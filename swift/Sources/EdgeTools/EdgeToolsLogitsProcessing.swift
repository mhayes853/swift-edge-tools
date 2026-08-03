// MARK: - EdgeToolsLogitsProcessor

public protocol EdgeToolsLogitsProcessor<Prompt, Logits> {
  associatedtype Prompt
  associatedtype Logits: ~Copyable & ~Escapable

  func prompt(_ prompt: Prompt)

  nonisolated(nonsending) func process(logits: inout Logits) async throws

  func didSample(tokenId: EdgeToolsToken.ID)
}
