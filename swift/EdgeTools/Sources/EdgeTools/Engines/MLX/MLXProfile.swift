#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && canImport(MLX)
  import _EdgeToolsFoundation
  import MLX
  import MLXLLM
  import MLXLMCommon
  import MLXNN
  import Observation

  #if canImport(CoreImage) && canImport(MLXVLM)
    import CoreImage
    import Foundation
    import MLXVLM
  #endif

  // MARK: - MLXModelProfile

  public protocol MLXModelProfile: SendableMetatype {
    associatedtype Prompt: Sendable
    associatedtype GenerateParameters: MLXGenerateParameters
    associatedtype GenerationParser: EdgeToolsGenerationParser
    associatedtype GrammarEngine: EdgeToolsGrammarEngine

    static var extraStopTokens: Set<String> { get }

    static func grammarEngine(
      tokenizer: any EdgeToolsTokenizer,
      vocabularySize: Int,
      stopTokenIds: Set<EdgeToolsToken.ID>
    ) throws -> GrammarEngine

    static func grammar(
      prompt: Prompt,
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      grammarEngine: borrowing GrammarEngine
    ) throws -> GrammarEngine.Grammar

    static func prepare(
      prompt: inout Prompt,
      tools: [EdgeToolDefinition],
      parser: inout GenerationParser
    )

    static func templateContext(prompt: Prompt) -> [String: EdgeToolsValue]?

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

    static func defaultSampling(
      prompt: Prompt,
      parameters: GenerateParameters
    ) -> EdgeToolsFusedSamplingParameters?
  }

  public protocol MLXLLMModelProfile: MLXModelProfile {}

  #if canImport(CoreImage) && canImport(MLXVLM)
    public protocol MLXVLMModelProfile: MLXModelProfile {}
  #endif

  extension MLXModelProfile
  where
    GenerateParameters: EdgeToolsConstrainedGenerateParameters,
    GenerateParameters.Constraint.Grammar == GrammarEngine.Grammar,
    GenerateParameters.Constraint.Context == GrammarEngine
  {
    public static func constrainedGrammar(
      tools: [EdgeToolDefinition],
      parameters: GenerateParameters,
      grammarEngine: borrowing GrammarEngine,
      toolCallGrammar: (GrammarToolCallRange) throws -> GrammarEngine.Grammar
    ) throws -> GrammarEngine.Grammar {
      let constraint = parameters.constraint
      let grammar = try constraint.toolCallRange.map(toolCallGrammar)
      return try constraint.grammar(toolCallGrammar: grammar, context: grammarEngine)
    }
  }

  extension MLXModelProfile {
    public static var extraStopTokens: Set<String> { [] }

    public static func defaultSampling(
      prompt: Prompt,
      parameters: GenerateParameters
    ) -> EdgeToolsFusedSamplingParameters? {
      nil
    }

    public static func prepare(
      prompt: inout Prompt,
      tools: [EdgeToolDefinition],
      parser: inout GenerationParser
    ) {}

    public static func templateContext(prompt: Prompt) -> [String: EdgeToolsValue]? {
      nil
    }

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

  public protocol MLXGenerateParameters: EdgeToolsEngineGenerateParameters {
    var sampler: (any LogitSampler)? { get }
    var sampling: EdgeToolsFusedSamplingParameters { get }
    var processor: (any LogitProcessor)? { get }
    var kvCacheQuantizationBits: Int? { get }
    var kvCacheQuantizationGroupSize: Int { get }
    var quantizedKVStart: Int { get }
    var synchronizeStreamForMemorySnapshots: Bool { get }
  }

  extension MLXGenerateParameters {
    public var sampling: EdgeToolsFusedSamplingParameters {
      EdgeToolsFusedSamplingParameters()
    }
  }

  // MARK: - DefaultMLXGenerateParameters

  #if XGrammar
    public struct DefaultMLXGenerateParameters:
      MLXGenerateParameters,
      EdgeToolsConstrainedGenerateParameters
    {
      public static var `default`: Self { Self() }

      public var sampler: (any LogitSampler)?
      public var sampling: EdgeToolsFusedSamplingParameters
      public var processor: (any LogitProcessor)?
      public var constraint: XGRGenerationConstraint
      public var maxTokens: Int?
      public var kvCacheQuantizationBits: Int?
      public var kvCacheQuantizationGroupSize: Int
      public var quantizedKVStart: Int
      public var synchronizeStreamForMemorySnapshots: Bool

      public init(
        sampler: (any LogitSampler)? = nil,
        sampling: EdgeToolsFusedSamplingParameters = EdgeToolsFusedSamplingParameters(),
        processor: (any LogitProcessor)? = nil,
        constraint: XGRGenerationConstraint = .unconstrained,
        maxTokens: Int? = 1024,
        kvCacheQuantizationBits: Int? = nil,
        kvCacheQuantizationGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        synchronizeStreamForMemorySnapshots: Bool = true
      ) {
        self.sampler = sampler
        self.sampling = sampling
        self.processor = processor
        self.constraint = constraint
        self.maxTokens = maxTokens
        self.kvCacheQuantizationBits = kvCacheQuantizationBits
        self.kvCacheQuantizationGroupSize = kvCacheQuantizationGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.synchronizeStreamForMemorySnapshots = synchronizeStreamForMemorySnapshots
      }
    }

    // MARK: - MLXModelProfile + XGrammar

    extension MLXModelProfile where GrammarEngine == XGrammarEngine {
      public static func grammarEngine(
        tokenizer: any EdgeToolsTokenizer,
        vocabularySize: Int,
        stopTokenIds: Set<EdgeToolsToken.ID>
      ) throws -> XGrammarEngine {
        guard let tokenizer = tokenizer as? any XGRTokenizer else {
          throw EdgeToolsError.unsupportedTokenizer
        }
        return try XGrammarEngine(
          tokenizerInfo: tokenizer.tokenizerInfo(
            modelVocabularySize: vocabularySize,
            extraStopTokenIds: stopTokenIds
          )
        )
      }
    }
  #endif
#endif
