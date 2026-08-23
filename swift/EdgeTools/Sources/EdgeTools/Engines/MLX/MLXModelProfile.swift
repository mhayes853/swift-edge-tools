#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && canImport(MLX)
  import EdgeToolsCore
  import EdgeToolsTokenizers
  import MLXLMCommon

  // MARK: - MLXModelProfile

  public protocol MLXModelProfile: EdgeToolsModelProfile
  where GenerateParameters == MLXGenerateParameters {
    static nonisolated(nonsending) func input(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput

    static nonisolated(nonsending) func prefillInput(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput
  }

  public protocol MLXLLMModelProfile: MLXModelProfile {}

  #if canImport(CoreImage) && canImport(MLXVLM)
    public protocol MLXVLMModelProfile: MLXModelProfile {}
  #endif

  extension MLXModelProfile {
    public static nonisolated(nonsending) func prefillInput(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput {
      try await self.input(
        prompt: prompt,
        tools: tools,
        tokenizer: tokenizer,
        processor: processor
      )
    }
  }

  public struct MLXEngineError: Hashable, Sendable, Error {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let emptyInput = Self(rawValue: "empty-input")
    }

    public let code: Code
    public let message: String

    public init(code: Code, message: String) {
      self.code = code
      self.message = message
    }
  }

  // MARK: - MLXGenerateParameters

  #if XGrammar
    public struct MLXGenerateParameters: EdgeToolsConstrainedGenerateParameters, Sendable {
      public static var `default`: Self { Self() }

      public var sampler: (@Sendable () -> any LogitSampler)?
      public var sampling: EdgeToolsFusedSamplingParameters
      public var processor: (@Sendable () -> any LogitProcessor)?
      public var constraint: XGRGenerationConstraint
      public var confidence: EdgeToolsConfidenceOptions
      public var maxTokens: Int?
      public var synchronizeStreamForMemorySnapshots: Bool

      public init(
        sampler: (@Sendable () -> any LogitSampler)? = nil,
        sampling: EdgeToolsFusedSamplingParameters = EdgeToolsFusedSamplingParameters(),
        processor: (@Sendable () -> any LogitProcessor)? = nil,
        constraint: XGRGenerationConstraint = .unconstrained,
        confidence: EdgeToolsConfidenceOptions = [],
        maxTokens: Int? = 1024,
        synchronizeStreamForMemorySnapshots: Bool = true
      ) {
        self.sampler = sampler
        self.sampling = sampling
        self.processor = processor
        self.constraint = constraint
        self.confidence = confidence
        self.maxTokens = maxTokens
        self.synchronizeStreamForMemorySnapshots = synchronizeStreamForMemorySnapshots
      }
    }

  #endif
#endif
