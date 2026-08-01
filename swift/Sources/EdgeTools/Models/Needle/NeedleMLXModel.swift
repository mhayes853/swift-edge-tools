#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && canImport(MLX)
  import _EdgeToolsFoundation
  import MLX
  import MLXLLM
  import MLXLMCommon
  import MLXNN

  // MARK: - LMInput + Needle

  extension LMInput {
    public static func needle(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition],
      using tokenizer: some EdgeToolsXGRTokenizer
    ) throws -> Self {
      let tokens = tokenizer.encode(text: try prompt.formatted(tools: tools))
      return LMInput(text: LMInput.Text(tokens: MLXArray(tokens)))
    }
  }

  // MARK: - NeedleMLXModel

  public final class NeedleMLXModel: Module, LanguageModel, KVCacheDimensionProvider {
    public var vocabularySize: Int { self.configuration.vocabularySize }

    public var kvHeads: [Int] {
      [Int](repeating: self.configuration.kvHeads, count: self.configuration.hiddenLayers)
    }

    public let configuration: NeedleModelConfiguration

    private let model: NeedleSimpleAttentionNetwork
    @ModuleInfo(key: "lm_head") private var lmHead: Embedding?
    private var crossAttentionMask: MLXArray?
    private var crossAttentionKV: [ProjectedAttentionKV]?
    private var encoderOutput: MLXArray?

    private var defaultEncoderOutput: MLXArray {
      .zeros([1, 0, self.configuration.dimensions])
    }

    public init(configuration: NeedleModelConfiguration) {
      self.configuration = configuration
      self.model = NeedleSimpleAttentionNetwork(configuration: configuration)

      if !configuration.tieWordEmbeddings {
        self._lmHead.wrappedValue = Embedding(
          embeddingCount: configuration.vocabularySize,
          dimensions: configuration.dimensions
        )
      }
    }

    public func prepare(
      _ input: LMInput,
      cache: [any KVCache],
      windowSize: Int?
    ) throws -> PrepareResult {
      guard input.text.tokens.size <= self.configuration.encoderMaxLength else {
        throw EdgeToolsError.contextLengthExceeded(
          tokens: input.text.tokens.size,
          maximum: self.configuration.encoderMaxLength
        )
      }
      let output = try self.prepare(input.text.tokens, cache: cache, windowSize: windowSize)
      return output.map { .logits($0) } ?? .tokens(input.text)
    }

    public func prepare(
      _ tokens: MLXArray,
      cache: [any KVCache],
      windowSize: Int?,
    ) throws -> LMOutput? {
      try Task.checkCancellation()

      let encoderInput = tokens[.newAxis, 0...]
      let encoderMask = paddingMask(
        inputIds: encoderInput,
        padTokenId: self.configuration.padTokenId
      )
      self.crossAttentionMask = encoderMask

      let encoderOutput = self.model.encode(
        encoderInput,
        previous: self.defaultEncoderOutput,
        mask: encoderMask
      )
      let crossAttentionKV = self.model.precomputeCrossAttention(encoderOutput: encoderOutput)
      self.crossAttentionKV = crossAttentionKV
      self.encoderOutput = encoderOutput

      let output = self.model.decode(
        MLXArray([self.configuration.decoderStartTokenId])[.newAxis, 0...],
        crossAttentionMask: encoderMask,
        precomputedCrossAttention: crossAttentionKV,
        caches: cache
      )

      try Task.checkCancellation()
      eval(cache)
      return self.finalOutput(from: output, encoderOutput: encoderOutput)
    }

    public func callAsFunction(
      _ input: LMInput.Text,
      cache: [any KVCache]?,
      state: LMOutput.State?
    ) -> LMOutput {
      let encoderOutput =
        state?.crossAttentionStates ?? self.encoderOutput ?? self.defaultEncoderOutput
      let precomputedCrossAttention =
        self.crossAttentionKV ?? self.model.precomputeCrossAttention(encoderOutput: encoderOutput)
      let output = self.model.decode(
        input.tokens,
        crossAttentionMask: self.crossAttentionMask,
        precomputedCrossAttention: precomputedCrossAttention,
        caches: cache
      )
      return self.finalOutput(from: output, encoderOutput: encoderOutput)
    }

    public func loadWeights(from url: URL) throws {
      try self.loadWeights(weights: MLX.loadArrays(url: url))
    }

    public func loadWeights(data: Data) throws {
      try self.loadWeights(weights: MLX.loadArrays(data: data))
    }

    public func loadWeights(weights: [String: MLXArray]) throws {
      try self.update(
        parameters: ModuleParameters.unflattened(self.sanitize(weights: weights)),
        verify: .all
      )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
      var weights = weights
      if self.configuration.tieWordEmbeddings {
        weights.removeValue(forKey: "lm_head.weight")
      }
      return weights
    }

    public func reset() {
      self.encoderOutput = nil
      self.crossAttentionKV = nil
      self.crossAttentionMask = nil
    }

    private func finalOutput(from output: MLXArray, encoderOutput: MLXArray) -> LMOutput {
      LMOutput(
        logits: (self.lmHead ?? self.model.embedding).asLinear(output),
        state: LMOutput.State(crossAttentionStates: encoderOutput)
      )
    }
  }

  // MARK: - EdgeToolsMLXModel

  extension NeedleMLXModel: EdgeToolsMLXModel {
    public typealias ModelConfiguration = NeedleModelConfiguration
    public typealias Prompt = NeedlePrompt
    public typealias ToolCallParser = NeedleToolCallParser

    public func grammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try XGRGrammar.needle(tools: tools, range: range)
    }

    public func input(
      prompt: NeedlePrompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsXGRTokenizer
    ) throws -> LMInput {
      try LMInput.needle(prompt: prompt, tools: tools, using: tokenizer)
    }
  }

  public typealias NeedleMLXModelEngine = EdgeToolsMLXEngine<NeedleMLXModel>

  extension EdgeToolsMLXEngine where Model == NeedleMLXModel {
    public convenience init(from directoryURL: URL) async throws {
      try await self.init(
        from: directoryURL,
        model: { NeedleMLXModel(configuration: $0) }
      )
    }
  }

  // MARK: - SimpleAttentionNetwork

  private struct ProjectedAttentionKV {
    let keys: MLXArray
    let values: MLXArray
  }

  private final class NeedleSimpleAttentionNetwork: Module {
    @ModuleInfo(key: "embed_tokens") var embedding: Embedding
    let encoder: NeedleEncoder
    let decoder: NeedleDecoder
    private let embedScale: Float

    init(configuration: NeedleModelConfiguration) {
      self.encoder = NeedleEncoder(configuration: configuration)
      self.decoder = NeedleDecoder(configuration: configuration)
      self._embedding.wrappedValue = Embedding(
        embeddingCount: configuration.vocabularySize,
        dimensions: configuration.dimensions
      )
      self.embedScale = sqrt(Float(configuration.dimensions))
    }

    func encode(_ input: MLXArray, previous: MLXArray, mask: MLXArray) -> MLXArray {
      let encoded = self.encoder(self.embedding(input) * self.embedScale, mask: .array(mask))
      return concatenated([previous, encoded], axis: 1)
    }

    func decode(
      _ input: MLXArray,
      crossAttentionMask: MLXArray?,
      precomputedCrossAttention: [ProjectedAttentionKV],
      caches: [any KVCache]?
    ) -> MLXArray {
      self.decoder(
        self.embedding(input) * self.embedScale,
        selfMask: .causal,
        crossMask: crossAttentionMask.map { .array($0) } ?? .none,
        precomputedCrossAttention: precomputedCrossAttention,
        caches: caches
      )
    }

    func precomputeCrossAttention(encoderOutput: MLXArray) -> [ProjectedAttentionKV] {
      self.decoder.layers.map { $0.crossAttention.project(kv: encoderOutput) }
    }
  }

  // MARK: - Encoder

  private final class NeedleEncoder: Module {
    let layers: [NeedleEncoderBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: ZCRMSNorm
    private let _inverseRope: MLXArray

    init(configuration: NeedleModelConfiguration) {
      self.layers = (0..<configuration.encoderLayers)
        .map { _ in NeedleEncoderBlock(configuration: configuration) }
      self._finalNorm.wrappedValue = ZCRMSNorm(configuration: configuration)
      self._inverseRope = RoPEFrequencies.inverse(
        headDimensions: configuration.attentionHeadDimensions,
        theta: configuration.ropeTheta
      )
    }

    func callAsFunction(
      _ input: MLXArray,
      mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
      let ropeFrequencies = RoPEFrequencies(
        inverse: self._inverseRope,
        sequenceLength: input.dim(1),
        dtype: input.dtype
      )
      var hiddenStates = input
      for layer in self.layers {
        hiddenStates = layer(hiddenStates, mask: mask, ropeFrequencies: ropeFrequencies)
      }
      return self.finalNorm(hiddenStates)
    }
  }

  private final class NeedleEncoderBlock: Module {
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: ZCRMSNorm
    @ModuleInfo(key: "self_attn") private var attention: NeedleAttention
    @ParameterInfo(key: "attn_gate") private var attentionGate: MLXArray

    init(configuration: NeedleModelConfiguration) {
      self._inputLayerNorm.wrappedValue = ZCRMSNorm(configuration: configuration)
      self._attention.wrappedValue = NeedleAttention(configuration: configuration)
      self._attentionGate.wrappedValue = MLXArray([0])
    }

    func callAsFunction(
      _ input: MLXArray,
      mask: MLXFast.ScaledDotProductAttentionMaskMode,
      ropeFrequencies: RoPEFrequencies
    ) -> MLXArray {
      let normed = self.inputLayerNorm(input)
      let attention = self.attention(
        q: normed,
        kv: normed,
        mask: mask,
        ropeFrequencies: ropeFrequencies,
        cache: nil
      )
      return gatedResidual(input, gate: self.attentionGate, sublayer: attention)
    }
  }

  // MARK: - Decoder

  private final class NeedleDecoder: Module {
    let layers: [NeedleDecoderBlock]
    @ModuleInfo(key: "norm") var finalNorm: ZCRMSNorm
    private let _ropeTable: MLXArray

    init(configuration: NeedleModelConfiguration) {
      self.layers = (0..<configuration.decoderLayers)
        .map { _ in NeedleDecoderBlock(configuration: configuration) }
      self._finalNorm.wrappedValue = ZCRMSNorm(configuration: configuration)
      let inverseRope = RoPEFrequencies.inverse(
        headDimensions: configuration.attentionHeadDimensions,
        theta: configuration.ropeTheta
      )
      self._ropeTable = RoPEFrequencies.table(
        inverse: inverseRope,
        length: configuration.encoderMaxLength
      )
    }

    func callAsFunction(
      _ input: MLXArray,
      selfMask: MLXFast.ScaledDotProductAttentionMaskMode,
      crossMask: MLXFast.ScaledDotProductAttentionMaskMode,
      precomputedCrossAttention: [ProjectedAttentionKV],
      caches: [any KVCache]?,
    ) -> MLXArray {
      let ropeFrequencies = RoPEFrequencies.fromTable(
        self._ropeTable,
        offset: caches?.first?.offset ?? 0,
        sequenceLength: input.dim(1),
        dtype: input.dtype
      )
      var hiddenStates = input
      let optionalCaches = caches?.map { $0 as (any KVCache)? }
      let caches = optionalCaches ?? Array(repeating: nil, count: self.layers.count)
      for (index, (layer, caches)) in zip(self.layers, caches).enumerated() {
        hiddenStates = layer(
          hiddenStates,
          selfMask: selfMask,
          crossMask: crossMask,
          precomputedCrossAttention: precomputedCrossAttention[index],
          ropeFrequencies: ropeFrequencies,
          cache: caches
        )
      }
      return self.finalNorm(hiddenStates)
    }
  }

  private final class NeedleDecoderBlock: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: ZCRMSNorm
    @ModuleInfo(key: "self_attn") var selfAttention: NeedleAttention
    @ParameterInfo(key: "self_attn_gate") var selfAttentionGate: MLXArray
    @ModuleInfo(key: "encoder_attn_layer_norm") var crossAttentionNorm: ZCRMSNorm
    @ModuleInfo(key: "encoder_attn") var crossAttention: NeedleAttention
    @ParameterInfo(key: "cross_attn_gate") var crossAttentionGate: MLXArray

    init(configuration: NeedleModelConfiguration) {
      self._inputLayerNorm.wrappedValue = ZCRMSNorm(configuration: configuration)
      self._selfAttention.wrappedValue = NeedleAttention(configuration: configuration)
      self._selfAttentionGate.wrappedValue = MLXArray([0])
      self._crossAttentionNorm.wrappedValue = ZCRMSNorm(configuration: configuration)
      self._crossAttention.wrappedValue = NeedleAttention(configuration: configuration)
      self._crossAttentionGate.wrappedValue = MLXArray([0])
    }

    func callAsFunction(
      _ input: MLXArray,
      selfMask: MLXFast.ScaledDotProductAttentionMaskMode,
      crossMask: MLXFast.ScaledDotProductAttentionMaskMode,
      precomputedCrossAttention: ProjectedAttentionKV,
      ropeFrequencies: RoPEFrequencies,
      cache: (any KVCache)?
    ) -> MLXArray {
      let selfNormed = self.inputLayerNorm(input)
      let selfAttention = self.selfAttention(
        q: selfNormed,
        kv: selfNormed,
        mask: selfMask,
        ropeFrequencies: ropeFrequencies,
        cache: cache
      )

      let gatedSelfAttention = gatedResidual(
        input,
        gate: self.selfAttentionGate,
        sublayer: selfAttention
      )
      let crossNormed = self.crossAttentionNorm(gatedSelfAttention)
      let crossAttention = self.crossAttention(
        q: crossNormed,
        projectedKV: precomputedCrossAttention,
        mask: crossMask,
        ropeFrequencies: nil,
        cache: nil
      )
      return gatedResidual(
        gatedSelfAttention,
        gate: self.crossAttentionGate,
        sublayer: crossAttention
      )
    }
  }

  // MARK: - Attention

  private final class NeedleAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let scale: Float
    @ModuleInfo(key: "q_norm") var queryNorm: ZCRMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: ZCRMSNorm

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "out_proj") var outProjection: Linear

    init(configuration: NeedleModelConfiguration) {
      self.heads = configuration.attentionHeads
      self.kvHeads = configuration.kvHeads
      self.headDimensions = configuration.attentionHeadDimensions
      self.scale = sqrt(1.0 / Float(configuration.attentionHeadDimensions))

      self._queryNorm.wrappedValue = ZCRMSNorm(
        dimensions: configuration.attentionHeadDimensions,
        eps: configuration.rmsNormEps
      )
      self._keyNorm.wrappedValue = ZCRMSNorm(
        dimensions: configuration.attentionHeadDimensions,
        eps: configuration.rmsNormEps
      )

      self._queryProjection.wrappedValue = Linear(
        inputDimensions: configuration.hiddenDimensions,
        outputDimensions: configuration.hiddenDimensions,
        bias: false
      )
      self._keyProjection.wrappedValue = Linear(
        inputDimensions: configuration.hiddenDimensions,
        outputDimensions: configuration.kvDimensions,
        bias: false
      )
      self._valueProjection.wrappedValue = Linear(
        inputDimensions: configuration.hiddenDimensions,
        outputDimensions: configuration.kvDimensions,
        bias: false
      )
      self._outProjection.wrappedValue = Linear(
        inputDimensions: configuration.hiddenDimensions,
        outputDimensions: configuration.hiddenDimensions,
        bias: false
      )
    }

    func callAsFunction(
      q query: MLXArray,
      kv: MLXArray,
      mask: MLXFast.ScaledDotProductAttentionMaskMode,
      ropeFrequencies: RoPEFrequencies?,
      cache: (any KVCache)?
    ) -> MLXArray {
      var projectedKV = self.project(kv: kv)

      if let ropeFrequencies {
        projectedKV = ProjectedAttentionKV(
          keys: ropeFrequencies.apply(to: projectedKV.keys),
          values: projectedKV.values
        )
      }

      return self.callAsFunction(
        q: query,
        projectedKV: projectedKV,
        mask: mask,
        ropeFrequencies: ropeFrequencies,
        cache: cache
      )
    }

    func project(kv: MLXArray) -> ProjectedAttentionKV {
      var keys = self.keyProjection(kv)
      var values = self.valueProjection(kv)

      keys = unflatten(keys, axis: -1, shape: [self.kvHeads, self.headDimensions])
        .transposed(0, 2, 1, 3)
      values = unflatten(values, axis: -1, shape: [self.kvHeads, self.headDimensions])
        .transposed(0, 2, 1, 3)
      keys = self.keyNorm(keys)

      return ProjectedAttentionKV(keys: keys, values: values)
    }

    func callAsFunction(
      q query: MLXArray,
      projectedKV: ProjectedAttentionKV,
      mask: MLXFast.ScaledDotProductAttentionMaskMode,
      ropeFrequencies: RoPEFrequencies?,
      cache: (any KVCache)?
    ) -> MLXArray {
      var queries = self.queryProjection(query)
      queries = unflatten(queries, axis: -1, shape: [self.heads, self.headDimensions])
        .transposed(0, 2, 1, 3)
      queries = self.queryNorm(queries)

      if let ropeFrequencies {
        queries = ropeFrequencies.apply(to: queries)
      }

      let output = attentionWithCacheUpdate(
        queries: queries,
        keys: projectedKV.keys,
        values: projectedKV.values,
        cache: cache,
        scale: self.scale,
        mask: mask
      )
      return self.outProjection(output.transposed(0, 2, 1, 3).flattened(start: -2, end: -1))
    }
  }

  // MARK: - ZCRMSNorm

  private final class ZCRMSNorm: Module, UnaryLayer {
    let eps: Float
    let weight: MLXArray

    convenience init(configuration: NeedleModelConfiguration) {
      self.init(dimensions: configuration.dimensions, eps: configuration.rmsNormEps)
    }

    init(dimensions: Int, eps: Float) {
      self.eps = eps
      self.weight = .zeros([dimensions])
      super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
      rmsNorm(input, weight: 1.0 + self.weight.asType(input.dtype), eps: self.eps)
    }
  }

  // MARK: - RoPE

  private struct RoPEFrequencies {
    let sin: MLXArray
    let cos: MLXArray

    static func inverse(headDimensions: Int, theta: Float) -> MLXArray {
      let positions = MLXArray(stride(from: 0, to: headDimensions, by: 2)).asType(.float32)
      return 1.0 / pow(theta, positions / headDimensions)
    }

    static func table(inverse: MLXArray, length: Int) -> MLXArray {
      let positions = MLXArray(0..<length).asType(.float32)
      let frequencies =
        expandedDimensions(positions, axis: 1) * expandedDimensions(inverse, axis: 0)
      let embeddings = concatenated([frequencies, frequencies], axis: -1)
      return stacked([MLX.sin(embeddings), MLX.cos(embeddings)])
    }

    init(inverse: MLXArray, sequenceLength: Int, dtype: DType) {
      let positions = MLXArray(0..<sequenceLength).asType(.float32)
      let frequencies =
        expandedDimensions(positions, axis: 1) * expandedDimensions(inverse, axis: 0)
      let embeddings = concatenated([frequencies, frequencies], axis: -1)
      self.sin = expandedDimensions(MLX.sin(embeddings), axis: 0).asType(dtype)
      self.cos = expandedDimensions(MLX.cos(embeddings), axis: 0).asType(dtype)
    }

    static func fromTable(
      _ table: MLXArray,
      offset: Int,
      sequenceLength: Int,
      dtype: DType
    ) -> Self {
      let positions = offset..<(offset + sequenceLength)
      return Self(
        sin: table[0..<1, positions, 0...].asType(dtype),
        cos: table[1..<2, positions, 0...].asType(dtype)
      )
    }

    private init(sin: MLXArray, cos: MLXArray) {
      self.sin = sin
      self.cos = cos
    }

    func apply(to input: MLXArray) -> MLXArray {
      let sin = expandedDimensions(self.sin, axis: 1)
      let cos = expandedDimensions(self.cos, axis: 1)
      let half = input.dim(-1) / 2
      let rotated = concatenated(
        [-input[.ellipsis, half...], input[.ellipsis, ..<half]],
        axis: -1
      )
      return (input * cos) + (rotated * sin)
    }
  }

  // MARK: - Helpers

  private func gatedResidual(
    _ input: MLXArray,
    gate: MLXArray,
    sublayer: MLXArray
  ) -> MLXArray {
    clip(
      input + sigmoid(gate).asType(sublayer.dtype) * sublayer,
      min: -NeedleNumerics.float16ClippingMagnitude,
      max: NeedleNumerics.float16ClippingMagnitude
    )
  }

  private func paddingMask(inputIds: MLXArray, padTokenId: Int) -> MLXArray {
    (inputIds .!= padTokenId)[.ellipsis, .newAxis, .newAxis, 0...]
  }
#endif
