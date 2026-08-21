#if Llama && canImport(CLlama)
  import EdgeToolsCore
  import EdgeToolsTokenizers

  #if XGrammar
    import EdgeToolsXGrammar
  #endif

  // MARK: - LlamaModelProfile

  public protocol LlamaModelProfile: EdgeToolsModelProfile
  where GenerateParameters: LlamaGenerateParameters, Prompt == EdgeToolsTranscript {
    static func tokenIds(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      addGenerationPrompt: Bool
    ) throws -> [EdgeToolsToken.ID]
  }

  extension LlamaModelProfile {
    public static func tokenIds(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      addGenerationPrompt: Bool
    ) throws -> [EdgeToolsToken.ID] {
      guard let tokenizer = tokenizer as? any EdgeToolsChatTokenizer else {
        throw EdgeToolsError.unsupportedTokenizer
      }
      let tokens = try tokenizer.applyChatTemplate(
        messages: prompt.chatTemplateMessages(),
        tools: tools.chatTemplateToolValues,
        addGenerationPrompt: addGenerationPrompt,
        additionalContext: Self.templateContext(prompt: prompt)
      )
      return tokens.map(\.id)
    }
  }

  // MARK: - LlamaGenerateParameters

  public protocol LlamaGenerateParameters: EdgeToolsEngineGenerateParameters {
    var sampling: EdgeToolsFusedSamplingParameters { get }
    var confidence: EdgeToolsConfidenceOptions { get }
  }

  extension LlamaGenerateParameters {
    public var confidence: EdgeToolsConfidenceOptions { [] }
  }

  #if XGrammar
    public struct DefaultLlamaGenerateParameters:
      LlamaGenerateParameters,
      EdgeToolsConstrainedGenerateParameters
    {
      public static var `default`: Self { Self() }

      public var sampling: EdgeToolsFusedSamplingParameters
      public var constraint: XGRGenerationConstraint
      public var confidence: EdgeToolsConfidenceOptions
      public var maxTokens: Int?

      public init(
        sampling: EdgeToolsFusedSamplingParameters = EdgeToolsFusedSamplingParameters(),
        constraint: XGRGenerationConstraint = .unconstrained,
        confidence: EdgeToolsConfidenceOptions = [],
        maxTokens: Int? = 1024
      ) {
        self.sampling = sampling
        self.constraint = constraint
        self.confidence = confidence
        self.maxTokens = maxTokens
      }
    }
  #endif

  // MARK: - LlamaContext

  public typealias LlamaContext<Profile: LlamaModelProfile> = EdgeToolsTranscriptContext<
    LlamaContextState<Profile>
  >
#endif
