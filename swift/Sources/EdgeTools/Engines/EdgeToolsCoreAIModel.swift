#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI

  // MARK: - EdgeToolsCoreAIGenerateParameters

  @available(anyAppleOS 27.0, *)
  public struct EdgeToolsCoreAIGenerateParameters: EdgeToolsModelEngineGenerateParameters {
    public static var `default`: Self { Self() }

    public var sampler: any EdgeToolsSampler<NDArray>
    public var processor: (any EdgeToolsLogitsProcessor<NDArray, NDArray>)?
    public var constraint: EdgeToolsXGRGenerationConstraint
    public var maxTokens: Int?

    public var computeStream: ComputeStream?

    public init(
      sampler: any EdgeToolsSampler<NDArray> = CoreAIArgmaxSampler(),
      processor: (any EdgeToolsLogitsProcessor<NDArray, NDArray>)? = nil,
      computeStream: ComputeStream? = nil,
      constraint: EdgeToolsXGRGenerationConstraint = .tools,
      maxTokens: Int? = 1024
    ) {
      self.sampler = sampler
      self.processor = processor
      self.computeStream = computeStream
      self.constraint = constraint
      self.maxTokens = maxTokens
    }
  }

  // MARK: - EdgeToolsCoreAIGenerationState

  @available(anyAppleOS 27.0, *)
  public struct EdgeToolsCoreAIGenerationState<ModelState> {
    public var modelState: ModelState

    private let sampler: any EdgeToolsSampler<NDArray>
    private var processor: (any EdgeToolsLogitsProcessor<NDArray, NDArray>)?

    public var computeStream: ComputeStream?

    public init(
      modelState: ModelState,
      parameters: EdgeToolsCoreAIGenerateParameters
    ) {
      self.modelState = modelState
      self.sampler = parameters.sampler
      self.processor = parameters.processor
      self.computeStream = parameters.computeStream
    }

    public nonisolated(nonsending) func sample(
      logits: inout NDArray,
      bitmask: GrammarBitmask
    ) async throws -> EdgeToolsModelSample {
      let processedLogits = try await self.processor?.process(logits: &logits) ?? logits
      var maskedLogits = processedLogits
      applyBitmaskCoreAI(logits: &maskedLogits, mask: bitmask)
      let confidence = try tokenConfidenceCoreAI(logits: maskedLogits)
      let tokenId = try await self.sampler.sample(logits: maskedLogits)
      return EdgeToolsModelSample(tokenId: tokenId, confidence: confidence)
    }

    public func didAccept(token: EdgeToolsToken) {
      self.processor?.didSample(token: token)
    }
  }
#endif
