#if XGrammar
  import EdgeToolsXGrammar
#endif

#if ONNXCore
  // MARK: - EdgeToolsONNXGenerateParameters

  public struct EdgeToolsONNXGenerateParameters: EdgeToolsModelEngineGenerateParameters {
    public static var `default`: Self { Self() }

    public var sampler: any EdgeToolsSampler<[Float]>
    public var processor: (any EdgeToolsLogitsProcessor<[EdgeToolsToken.ID], [Float]>)?
    public var constraint: EdgeToolsXGRGenerationConstraint
    public var maxTokens: Int?

    public init(
      sampler: any EdgeToolsSampler<[Float]> = ONNXArgmaxSampler(),
      processor: (any EdgeToolsLogitsProcessor<[EdgeToolsToken.ID], [Float]>)? = nil,
      constraint: EdgeToolsXGRGenerationConstraint = .tools,
      maxTokens: Int? = 1024
    ) {
      self.sampler = sampler
      self.processor = processor
      self.constraint = constraint
      self.maxTokens = maxTokens
    }
  }

  // MARK: - EdgeToolsONNXGenerationState

  public struct EdgeToolsONNXGenerationState<ModelState> {
    var modelState: ModelState
    let sampler: any EdgeToolsSampler<[Float]>
    var processor: (any EdgeToolsLogitsProcessor<[EdgeToolsToken.ID], [Float]>)?
  }

  // MARK: - EdgeToolsONNXError

  public struct EdgeToolsONNXError: Hashable, Sendable, Error {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let integerConversionFailure = Self(rawValue: "integer-conversion-failure")
      public static let invalidLogitsCount = Self(rawValue: "invalid-logits-count")
    }

    public let code: Code
    public let message: String

    public init(code: Code, message: String) {
      self.code = code
      self.message = message
    }
  }
#endif
