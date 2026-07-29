#if XGrammar
  import EdgeToolsXGrammar
#endif

#if ONNXCore
  // MARK: - EdgeToolsONNXGenerateParameters

  public struct EdgeToolsONNXGenerateParameters: EdgeToolsModelEngineGenerateParameters {
    public static var `default`: Self { Self() }

    public var sampler: any EdgeToolsONNXSampler
    public var processor: (any EdgeToolsONNXLogitsProcessor)?
    public var constraint: EdgeToolsXGRGenerationConstraint
    public var maxTokens: Int?

    public init(
      sampler: any EdgeToolsONNXSampler = ONNXArgmaxSampler(),
      processor: (any EdgeToolsONNXLogitsProcessor)? = nil,
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
    let sampler: any EdgeToolsONNXSampler
    var processor: (any EdgeToolsONNXLogitsProcessor)?
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
