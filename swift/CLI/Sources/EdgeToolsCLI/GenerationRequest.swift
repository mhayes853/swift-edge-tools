import EdgeTools

public struct GenerationRequest: Sendable {
  public var system: String
  public var user: String
  public var images: [EdgeToolsConversationalPrompt.Asset]
  public var tools: [EdgeToolDefinition]
  public var grammar: GrammarOption
  public var toolCallRange: GrammarToolCallRange
  public var maxTokens: Int?
  public var sampling: EdgeToolsFusedSamplingOverrides
  public var reasoning: EdgeToolsReasoningEffort

  public init(
    system: String = "",
    user: String,
    images: [EdgeToolsConversationalPrompt.Asset] = [],
    tools: [EdgeToolDefinition] = [],
    grammar: GrammarOption = .auto,
    toolCallRange: GrammarToolCallRange = .unbounded(minimum: 0),
    maxTokens: Int? = 1024,
    sampling: EdgeToolsFusedSamplingOverrides = EdgeToolsFusedSamplingOverrides(),
    reasoning: EdgeToolsReasoningEffort = .default
  ) {
    self.system = system
    self.user = user
    self.images = images
    self.tools = tools
    self.grammar = grammar
    self.toolCallRange = toolCallRange
    self.maxTokens = maxTokens
    self.sampling = sampling
    self.reasoning = reasoning
  }
}
