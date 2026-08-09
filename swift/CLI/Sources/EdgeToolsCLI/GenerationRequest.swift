import EdgeTools

public struct GenerationRequest: Sendable {
  public var system: String
  public var user: String
  public var images: [EdgeToolsConversationalPrompt.Asset]
  public var tools: [EdgeToolDefinition]
  public var grammar: GrammarOption
  public var toolCallRange: GrammarToolCallRange
  public var maxTokens: Int?
  public var temperature: Float?
  public var topK: Int?
  public var topP: Float?
  public var minP: Float?
  public var repetitionPenalty: Float?
  public var presencePenalty: Float?
  public var repetitionContextSize: Int?
  public var seed: UInt64?
  public var reasoning: EdgeToolsReasoningEffort

  public init(
    system: String = "",
    user: String,
    images: [EdgeToolsConversationalPrompt.Asset] = [],
    tools: [EdgeToolDefinition] = [],
    grammar: GrammarOption = .auto,
    toolCallRange: GrammarToolCallRange = .unbounded(minimum: 0),
    maxTokens: Int? = 1024,
    temperature: Float? = nil,
    topK: Int? = nil,
    topP: Float? = nil,
    minP: Float? = nil,
    repetitionPenalty: Float? = nil,
    presencePenalty: Float? = nil,
    repetitionContextSize: Int? = nil,
    seed: UInt64? = nil,
    reasoning: EdgeToolsReasoningEffort = .default
  ) {
    self.system = system
    self.user = user
    self.images = images
    self.tools = tools
    self.grammar = grammar
    self.toolCallRange = toolCallRange
    self.maxTokens = maxTokens
    self.temperature = temperature
    self.topK = topK
    self.topP = topP
    self.minP = minP
    self.repetitionPenalty = repetitionPenalty
    self.presencePenalty = presencePenalty
    self.repetitionContextSize = repetitionContextSize
    self.seed = seed
    self.reasoning = reasoning
  }

  public var fusedSamplingParameters: EdgeToolsFusedSamplingParameters? {
    guard self.hasSamplingOverride else {
      return nil
    }
    return EdgeToolsFusedSamplingParameters(
      temperature: self.temperature ?? 0.6,
      topK: self.topK,
      topP: self.topP,
      minP: self.minP,
      repetitionPenalty: self.repetitionPenalty ?? 1,
      presencePenalty: self.presencePenalty ?? 0,
      repetitionContextSize: self.repetitionContextSize ?? 20,
      seed: self.seed
    )
  }

  public var hasSamplingOverride: Bool {
    self.temperature != nil
      || self.topK != nil
      || self.topP != nil
      || self.minP != nil
      || self.repetitionPenalty != nil
      || self.presencePenalty != nil
      || self.repetitionContextSize != nil
      || self.seed != nil
  }
}
