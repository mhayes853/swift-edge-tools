// MARK: - EdgeToolsLogitsProcessor

public protocol EdgeToolsLogitsProcessor<Prompt, Logits> {
  associatedtype Prompt
  associatedtype Logits

  mutating func prompt(_ prompt: Prompt)

  func process(logits: inout Logits) async throws -> Logits

  mutating func didSample(token: EdgeToolsToken)
}
