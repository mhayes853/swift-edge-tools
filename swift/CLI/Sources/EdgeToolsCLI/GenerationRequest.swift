import EdgeTools

public struct GenerationRequest: Sendable {
  public var system: String
  public var user: String
  public var tools: [EdgeToolDefinition]
  public var grammar: GrammarOption
  public var toolCallRange: GrammarToolCallRange
  public var maxTokens: Int?
  public var temperature: Float
  public var topP: Float

  public init(
    system: String = "",
    user: String,
    tools: [EdgeToolDefinition] = [],
    grammar: GrammarOption = .auto,
    toolCallRange: GrammarToolCallRange = .unbounded(minimum: 0),
    maxTokens: Int? = 1024,
    temperature: Float = 0,
    topP: Float = 1
  ) {
    self.system = system
    self.user = user
    self.tools = tools
    self.grammar = grammar
    self.toolCallRange = toolCallRange
    self.maxTokens = maxTokens
    self.temperature = temperature
    self.topP = topP
  }
}
