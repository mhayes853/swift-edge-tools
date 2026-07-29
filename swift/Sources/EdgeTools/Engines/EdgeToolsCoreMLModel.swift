#if CoreML && canImport(CoreML)
  import CoreML

  // MARK: - EdgeToolsCoreMLGenerateParameters

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  public struct EdgeToolsCoreMLGenerateParameters: EdgeToolsModelEngineGenerateParameters {
    public static var `default`: Self { Self() }

    public var sampler: any EdgeToolsSampler<MLTensor>
    public var processor: (any EdgeToolsLogitsProcessor<MLTensor, MLTensor>)?
    public var constraint: EdgeToolsXGRGenerationConstraint
    public var maxTokens: Int?

    public init(
      sampler: any EdgeToolsSampler<MLTensor> = CoreMLArgmaxSampler(),
      processor: (any EdgeToolsLogitsProcessor<MLTensor, MLTensor>)? = nil,
      constraint: EdgeToolsXGRGenerationConstraint = .tools,
      maxTokens: Int? = 1024
    ) {
      self.sampler = sampler
      self.processor = processor
      self.constraint = constraint
      self.maxTokens = maxTokens
    }
  }

  // MARK: - EdgeToolsCoreMLGenerationState

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  public struct EdgeToolsCoreMLGenerationState<ModelState> {
    public var modelState: ModelState

    private let sampler: any EdgeToolsSampler<MLTensor>
    private var processor: (any EdgeToolsLogitsProcessor<MLTensor, MLTensor>)?

    public init(
      modelState: ModelState,
      parameters: EdgeToolsCoreMLGenerateParameters
    ) {
      self.modelState = modelState
      self.sampler = parameters.sampler
      self.processor = parameters.processor
    }

    public nonisolated(nonsending) func sample(
      logits: inout MLTensor,
      bitmask: GrammarBitmask
    ) async throws -> EdgeToolsModelSample {
      let processedLogits = try await self.processor?.process(logits: &logits) ?? logits
      let maskedLogits = applyBitmaskCoreML(logits: processedLogits, mask: bitmask)
      let confidence = await tokenConfidenceCoreML(logits: maskedLogits)
      let tokenId = try await self.sampler.sample(logits: maskedLogits)
      return EdgeToolsModelSample(tokenId: tokenId, confidence: confidence)
    }

    public func didAccept(token: EdgeToolsToken) {
      self.processor?.didSample(token: token)
    }
  }
#endif
