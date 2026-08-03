#if XGrammar
  import EdgeToolsXGrammar
#endif

#if ONNXCore
  // MARK: - ONNXGenerateParameters

  public struct ONNXGenerateParameters: EdgeToolsModelEngineGenerateParameters {
    public static var `default`: Self { Self() }

    public var sampler: any EdgeToolsSampler<ONNXTensorView<Float>>
    public var processor:
      (any EdgeToolsLogitsProcessor<[EdgeToolsToken.ID], ONNXTensorView<Float>>)?
    public var constraint: EdgeToolsXGRGenerationConstraint
    public var maxTokens: Int?

    public init(
      sampler: any EdgeToolsSampler<ONNXTensorView<Float>> = ONNXArgmaxSampler(),
      processor: (
        any EdgeToolsLogitsProcessor<
          [EdgeToolsToken.ID], ONNXTensorView<Float>
        >
      )? = nil,
      constraint: EdgeToolsXGRGenerationConstraint = .tools,
      maxTokens: Int? = 1024
    ) {
      self.sampler = sampler
      self.processor = processor
      self.constraint = constraint
      self.maxTokens = maxTokens
    }
  }

  // MARK: - ONNXError

  public struct ONNXError: Hashable, Sendable, Error {
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
