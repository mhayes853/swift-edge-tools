// MARK: - EdgeToolsLogitsProcessor

public protocol EdgeToolsLogitsProcessor<Prompt, Logits> {
  associatedtype Prompt
  associatedtype Logits

  func prompt(_ prompt: Prompt)

  nonisolated(nonsending) func process(logits: inout Logits) async throws -> Logits

  func didSample(tokenId: EdgeToolsToken.ID)
}
