import EdgeToolsCore

public protocol EdgeToolsLogitsProcessor<Prompt, Logits>: Sendable {
  associatedtype Prompt: Sendable
  associatedtype Logits: ~Copyable & ~Escapable

  func prompt(_ prompt: Prompt)

  nonisolated(nonsending) func process(logits: inout Logits) async throws

  func didSample(tokenId: EdgeToolsToken.ID)
}
