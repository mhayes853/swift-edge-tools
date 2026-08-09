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
      using tokenizer: some EdgeToolsTokenizer
    ) throws -> Self {
      let tokens = tokenizer.encode(text: try prompt.formatted(tools: tools))
      return LMInput(text: LMInput.Text(tokens: MLXArray(tokens)))
    }
  }

  // MARK: - NeedleLanguageModel

  public final class NeedleLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
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
    private var compiledDecoderFunction: (@Sendable ([MLXArray]) -> [MLXArray])?
    private static let encoderOutputKey = LMOutput.Key<MLXArray>(
      "edge-tools.needle.encoder-output"
    )

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

    public func newCache(parameters _: MLXLMCommon.GenerateParameters?) -> [any KVCache] {
      (0..<self.configuration.decoderLayers)
        .map { _ in
          let dtype = self.model.embedding.weight.dtype
          return NeedleKVCache(configuration: self.configuration, dtype: dtype)
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

      let output = self.decode(
        MLXArray([self.configuration.decoderStartTokenId])[.newAxis, 0...],
        crossAttentionMask: encoderMask,
        precomputedCrossAttention: crossAttentionKV,
        caches: self.caches(from: cache),
        encoderOutput: encoderOutput
      )

      try Task.checkCancellation()
      eval(cache)
      return output
    }

    public func callAsFunction(
      _ input: LMInput.Text,
      cache: [any KVCache]?,
      state: LMOutput.State?
    ) -> LMOutput {
      precondition(input.tokens.dim(1) == 1, "Needle decoding requires exactly one token.")
      guard let crossAttentionMask, let crossAttentionKV else {
        preconditionFailure("Needle must be prepared before decoding.")
      }
      let encoderOutput =
        state?[Self.encoderOutputKey] ?? self.encoderOutput ?? self.defaultEncoderOutput
      return self.decode(
        input.tokens,
        crossAttentionMask: crossAttentionMask,
        precomputedCrossAttention: crossAttentionKV,
        caches: self.caches(from: cache),
        encoderOutput: encoderOutput
      )
    }

    public func loadWeights(from url: URL) throws {
      try self.loadWeights(weights: MLX.loadArrays(url: url))
    }

    public func loadWeights(data: Data) throws {
      try self.loadWeights(weights: MLX.loadArrays(data: data))
    }

    public func loadWeights(weights: [String: MLXArray]) throws {
      self.compiledDecoderFunction = nil
      try self.update(
        parameters: ModuleParameters.unflattened(self.sanitize(weights: weights)),
        verify: .all
      )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
      var weights = weights
      fuseNeedleAttentionWeights(&weights)
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

    private func decode(
      _ input: MLXArray,
      crossAttentionMask: MLXArray,
      precomputedCrossAttention: [ProjectedAttentionKV],
      caches: [NeedleKVCache],
      encoderOutput: MLXArray
    ) -> LMOutput {
      let offset = caches[0].offset
      let dtype = self.model.embedding.weight.dtype
      let cacheStates = caches.map { $0.fullState(dtype: dtype) }
      let inputs =
        [input, MLXArray([Int32(offset)]), crossAttentionMask]
        + precomputedCrossAttention.map(\.keys)
        + precomputedCrossAttention.map(\.values)
        + cacheStates.map(\.keys)
        + cacheStates.map(\.values)
      let outputs = self.compiledDecoder()(inputs)
      let layerCount = self.configuration.decoderLayers
      for index in 0..<layerCount {
        caches[index]
          .replace(
            state: NeedleKVCache.State(
              keys: outputs[1 + index],
              values: outputs[1 + layerCount + index]
            ),
            offset: offset + 1
          )
      }
      var state = LMOutput.State()
      state[Self.encoderOutputKey] = encoderOutput
      return LMOutput(logits: outputs[0], state: state)
    }

    private func caches(from caches: [any KVCache]?) -> [NeedleKVCache] {
      guard let caches = caches as? [NeedleKVCache],
        caches.count == self.configuration.decoderLayers,
        let offset = caches.first?.offset
      else {
        preconditionFailure(
          "Needle requires one cache per decoder layer from NeedleMLXLanguageModel."
        )
      }
      precondition(caches.allSatisfy { $0.offset == offset }, "Needle KV cache offsets must match.")
      return caches
    }

    private func compiledDecoder() -> @Sendable ([MLXArray]) -> [MLXArray] {
      if let compiledDecoderFunction {
        return compiledDecoderFunction
      }
      let layerCount = self.configuration.decoderLayers
      let model = self.model
      let outputProjection = self.lmHead ?? self.model.embedding
      let compiledDecoderFunction = compile { inputs in
        let crossAttention = zip(
          inputs[3..<(3 + layerCount)],
          inputs[(3 + layerCount)..<(3 + (2 * layerCount))]
        )
        .map { ProjectedAttentionKV(keys: $0, values: $1) }
        let caches = zip(
          inputs[(3 + (2 * layerCount))..<(3 + (3 * layerCount))],
          inputs[(3 + (3 * layerCount))..<(3 + (4 * layerCount))]
        )
        .map { NeedleKVCache.State(keys: $0, values: $1) }
        let output = model.decode(
          inputs[0],
          position: inputs[1],
          crossAttentionMask: inputs[2],
          precomputedCrossAttention: crossAttention,
          caches: caches
        )
        return [outputProjection.asLinear(output.output)]
          + output.caches.map(\.keys)
          + output.caches.map(\.values)
      }
      self.compiledDecoderFunction = compiledDecoderFunction
      return compiledDecoderFunction
    }
  }

  // MARK: - NeedleMLXGenerateParameters

  public struct NeedleMLXGenerateParameters:
    MLXGenerateParameters,
    NeedleGenerateParameters
  {
    public static var `default`: Self { Self() }

    public var sampler: (any LogitSampler)?
    public var processor: (any LogitProcessor)?
    public var maxTokens: Int?
    public var toolCallRange: GrammarToolCallRange
    public var synchronizeStreamForMemorySnapshots: Bool
    public var kvCacheQuantizationBits: Int? { nil }
    public var kvCacheQuantizationGroupSize: Int { 64 }
    public var quantizedKVStart: Int { 0 }

    public init(
      sampler: (any LogitSampler)? = ArgMaxSampler(),
      processor: (any LogitProcessor)? = nil,
      maxTokens: Int? = 1024,
      toolCallRange: GrammarToolCallRange = .unbounded(minimum: 0),
      synchronizeStreamForMemorySnapshots: Bool = true
    ) {
      self.sampler = sampler
      self.processor = processor
      self.maxTokens = maxTokens
      self.toolCallRange = toolCallRange
      self.synchronizeStreamForMemorySnapshots = synchronizeStreamForMemorySnapshots
    }
  }

  // MARK: - MLXModelProfile

  #if XGrammar
    public struct NeedleMLXProfile: MLXModelProfile {
      public typealias Prompt = NeedlePrompt
      public typealias GenerateParameters = NeedleMLXGenerateParameters
      public typealias GenerationParser = NeedleGenerationParser
      public typealias GrammarCompiler = XGRCompiler
      public typealias GrammarContext = XGRGrammarContext

      public let languageModel: NeedleLanguageModel

      public init(configuration: NeedleModelConfiguration) {
        self.languageModel = NeedleLanguageModel(configuration: configuration)
      }

      public func loadWeights(from directory: MLXModelDirectory) throws {
        let baseConfiguration = try directory.loadConfiguration(BaseConfiguration.self)
        try MLXLMCommon.loadWeights(
          modelDirectory: directory.url,
          model: self.languageModel,
          perLayerQuantization: baseConfiguration.perLayerQuantization
        )
      }

      public static func grammar(
        prompt _: NeedlePrompt,
        tools: [EdgeToolDefinition],
        parameters: NeedleMLXGenerateParameters,
        context _: XGRGrammarContext
      ) throws -> XGRGrammar {
        try .needle(tools: tools, range: parameters.toolCallRange)
      }

      public static nonisolated(nonsending) func input(
        prompt: NeedlePrompt,
        tools: [EdgeToolDefinition],
        tokenizer: any EdgeToolsTokenizer,
        processor _: (any UserInputProcessor)?
      ) async throws -> LMInput {
        try .needle(prompt: prompt, tools: tools, using: tokenizer)
      }
    }

    public typealias NeedleMLXModelEngine = MLXEngine<NeedleMLXProfile>

    extension EdgeToolsModelEngine where Model == EdgeToolsMLXModel<NeedleMLXProfile> {
      public init(from directoryURL: URL) async throws {
        try await self.init(from: MLXModelDirectory(url: directoryURL))
      }

      public init(from directory: MLXModelDirectory) async throws {
        let configuration = try directory.loadConfiguration(NeedleModelConfiguration.self)
        try await self.init(from: directory, configuration: configuration)
      }

      public init(
        from directory: MLXModelDirectory,
        configuration: NeedleModelConfiguration
      ) async throws {
        let tokenizer = try await directory.loadTokenizer()
        let languageModel = NeedleLanguageModel(configuration: configuration)
        let baseConfiguration = try directory.loadConfiguration(BaseConfiguration.self)
        try MLXLMCommon.loadWeights(
          modelDirectory: directory.url,
          model: languageModel,
          perLayerQuantization: baseConfiguration.perLayerQuantization
        )
        var extraStopTokenIds = try directory.loadStopTokenIds()
        if let eosTokenId = tokenizer.eosTokenId {
          extraStopTokenIds.remove(eosTokenId)
        }
        try self.init(
          languageModel: languageModel,
          tokenizer: tokenizer,
          vocabularySize: configuration.vocabularySize,
          extraStopTokenIds: extraStopTokenIds
        )
      }
    }
  #endif

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
      let encoded = self.encoder(self.embedding(input) * self.embedScale, mask: mask)
      return concatenated([previous, encoded], axis: 1)
    }

    func decode(
      _ input: MLXArray,
      position: MLXArray,
      crossAttentionMask: MLXArray,
      precomputedCrossAttention: [ProjectedAttentionKV],
      caches: [NeedleKVCache.State]
    ) -> (output: MLXArray, caches: [NeedleKVCache.State]) {
      self.decoder(
        self.embedding(input) * self.embedScale,
        position: position,
        crossMask: crossAttentionMask,
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

    func callAsFunction(_ input: MLXArray, mask: MLXArray) -> MLXArray {
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
    @ModuleInfo(key: "self_attn") private var attention: NeedleSelfAttention
    @ParameterInfo(key: "attn_gate") private var attentionGate: MLXArray

    init(configuration: NeedleModelConfiguration) {
      self._inputLayerNorm.wrappedValue = ZCRMSNorm(configuration: configuration)
      self._attention.wrappedValue = NeedleSelfAttention(configuration: configuration)
      self._attentionGate.wrappedValue = MLXArray([0])
    }

    func callAsFunction(
      _ input: MLXArray,
      mask: MLXArray,
      ropeFrequencies: RoPEFrequencies
    ) -> MLXArray {
      let attention = self.attention(
        self.inputLayerNorm(input),
        mask: mask,
        ropeFrequencies: ropeFrequencies,
        cache: nil,
        position: nil
      )
      return gatedResidual(input, gate: self.attentionGate, sublayer: attention.output)
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
      position: MLXArray,
      crossMask: MLXArray,
      precomputedCrossAttention: [ProjectedAttentionKV],
      caches: [NeedleKVCache.State]
    ) -> (output: MLXArray, caches: [NeedleKVCache.State]) {
      let ropeFrequencies = RoPEFrequencies.fromTable(
        self._ropeTable,
        position: position,
        dtype: input.dtype
      )
      let keyPositions = MLXArray(Int32(0)..<Int32(caches[0].keys.dim(2)))
      let selfMask = (keyPositions .<= position)[.newAxis, .newAxis, .newAxis, 0...]
      var hiddenStates = input
      var updatedCaches = [NeedleKVCache.State]()
      updatedCaches.reserveCapacity(self.layers.count)

      for index in self.layers.indices {
        let output =
          self.layers[index](
            hiddenStates,
            selfMask: selfMask,
            crossMask: crossMask,
            precomputedCrossAttention: precomputedCrossAttention[index],
            cache: caches[index],
            ropeFrequencies: ropeFrequencies,
            position: position
          )
        hiddenStates = output.output
        updatedCaches.append(output.cache)
      }
      return (self.finalNorm(hiddenStates), updatedCaches)
    }
  }

  private final class NeedleDecoderBlock: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: ZCRMSNorm
    @ModuleInfo(key: "self_attn") var selfAttention: NeedleSelfAttention
    @ParameterInfo(key: "self_attn_gate") var selfAttentionGate: MLXArray
    @ModuleInfo(key: "encoder_attn_layer_norm") var crossAttentionNorm: ZCRMSNorm
    @ModuleInfo(key: "encoder_attn") var crossAttention: NeedleCrossAttention
    @ParameterInfo(key: "cross_attn_gate") var crossAttentionGate: MLXArray

    init(configuration: NeedleModelConfiguration) {
      self._inputLayerNorm.wrappedValue = ZCRMSNorm(configuration: configuration)
      self._selfAttention.wrappedValue = NeedleSelfAttention(configuration: configuration)
      self._selfAttentionGate.wrappedValue = MLXArray([0])
      self._crossAttentionNorm.wrappedValue = ZCRMSNorm(configuration: configuration)
      self._crossAttention.wrappedValue = NeedleCrossAttention(configuration: configuration)
      self._crossAttentionGate.wrappedValue = MLXArray([0])
    }

    func callAsFunction(
      _ input: MLXArray,
      selfMask: MLXArray,
      crossMask: MLXArray,
      precomputedCrossAttention: ProjectedAttentionKV,
      cache: NeedleKVCache.State,
      ropeFrequencies: RoPEFrequencies,
      position: MLXArray
    ) -> (output: MLXArray, cache: NeedleKVCache.State) {
      let selfAttention = self.selfAttention(
        self.inputLayerNorm(input),
        mask: selfMask,
        ropeFrequencies: ropeFrequencies,
        cache: cache,
        position: position
      )
      let gatedSelfAttention = gatedResidual(
        input,
        gate: self.selfAttentionGate,
        sublayer: selfAttention.output
      )
      let crossAttention = self.crossAttention(
        self.crossAttentionNorm(gatedSelfAttention),
        projectedKV: precomputedCrossAttention,
        mask: crossMask
      )
      return (
        gatedResidual(
          gatedSelfAttention,
          gate: self.crossAttentionGate,
          sublayer: crossAttention
        ),
        selfAttention.cache
      )
    }
  }

  // MARK: - Attention

  private final class NeedleSelfAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let queryDimensions: Int
    let kvDimensions: Int
    let scale: Float
    @ModuleInfo(key: "q_norm") var queryNorm: ZCRMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: ZCRMSNorm
    @ModuleInfo(key: "qkv_proj") var queryKeyValueProjection: Linear
    @ModuleInfo(key: "out_proj") var outProjection: Linear

    init(configuration: NeedleModelConfiguration) {
      self.heads = configuration.attentionHeads
      self.kvHeads = configuration.kvHeads
      self.headDimensions = configuration.attentionHeadDimensions
      self.queryDimensions = configuration.hiddenDimensions
      self.kvDimensions = configuration.kvDimensions
      self.scale = sqrt(1.0 / Float(configuration.attentionHeadDimensions))
      self._queryNorm.wrappedValue = ZCRMSNorm(
        dimensions: configuration.attentionHeadDimensions,
        eps: configuration.rmsNormEps
      )
      self._keyNorm.wrappedValue = ZCRMSNorm(
        dimensions: configuration.attentionHeadDimensions,
        eps: configuration.rmsNormEps
      )
      self._queryKeyValueProjection.wrappedValue = Linear(
        inputDimensions: configuration.hiddenDimensions,
        outputDimensions: configuration.hiddenDimensions + (2 * configuration.kvDimensions),
        bias: false
      )
      self._outProjection.wrappedValue = Linear(
        inputDimensions: configuration.hiddenDimensions,
        outputDimensions: configuration.hiddenDimensions,
        bias: false
      )
    }

    func callAsFunction(
      _ input: MLXArray,
      mask: MLXArray,
      ropeFrequencies: RoPEFrequencies,
      cache: NeedleKVCache.State?,
      position: MLXArray?
    ) -> (output: MLXArray, cache: NeedleKVCache.State) {
      let (queries, keys, values) = self.project(input, ropeFrequencies: ropeFrequencies)
      let attention: NeedleKVCache.State
      if let cache, let position {
        let indices = broadcast(position.reshaped([1, 1, 1, 1]), to: keys.shape)
        attention = NeedleKVCache.State(
          keys: putAlong(cache.keys, indices, values: keys, axis: 2),
          values: putAlong(cache.values, indices, values: values, axis: 2)
        )
      } else {
        attention = NeedleKVCache.State(keys: keys, values: values)
      }
      let output = MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: attention.keys,
        values: attention.values,
        scale: self.scale,
        mask: .array(mask)
      )
      return (
        self.outProjection(output.transposed(0, 2, 1, 3).flattened(start: -2, end: -1)),
        attention
      )
    }

    private func project(
      _ input: MLXArray,
      ropeFrequencies: RoPEFrequencies
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
      let projected = self.queryKeyValueProjection(input)
      var queries = projected[.ellipsis, ..<self.queryDimensions]
      var keys = projected[
        .ellipsis,
        self.queryDimensions..<(self.queryDimensions + self.kvDimensions)
      ]
      var values = projected[.ellipsis, (self.queryDimensions + self.kvDimensions)...]

      queries = unflatten(queries, axis: -1, shape: [self.heads, self.headDimensions])
        .transposed(0, 2, 1, 3)
      keys = unflatten(keys, axis: -1, shape: [self.kvHeads, self.headDimensions])
        .transposed(0, 2, 1, 3)
      values = unflatten(values, axis: -1, shape: [self.kvHeads, self.headDimensions])
        .transposed(0, 2, 1, 3)
      queries = ropeFrequencies.apply(to: self.queryNorm(queries))
      keys = ropeFrequencies.apply(to: self.keyNorm(keys))
      return (queries, keys, values)
    }
  }

  private final class NeedleCrossAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let kvDimensions: Int
    let scale: Float
    @ModuleInfo(key: "q_norm") var queryNorm: ZCRMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: ZCRMSNorm
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "kv_proj") var keyValueProjection: Linear
    @ModuleInfo(key: "out_proj") var outProjection: Linear

    init(configuration: NeedleModelConfiguration) {
      self.heads = configuration.attentionHeads
      self.kvHeads = configuration.kvHeads
      self.headDimensions = configuration.attentionHeadDimensions
      self.kvDimensions = configuration.kvDimensions
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
      self._keyValueProjection.wrappedValue = Linear(
        inputDimensions: configuration.hiddenDimensions,
        outputDimensions: 2 * configuration.kvDimensions,
        bias: false
      )
      self._outProjection.wrappedValue = Linear(
        inputDimensions: configuration.hiddenDimensions,
        outputDimensions: configuration.hiddenDimensions,
        bias: false
      )
    }

    func project(kv: MLXArray) -> ProjectedAttentionKV {
      let projected = self.keyValueProjection(kv)
      var keys = projected[.ellipsis, ..<self.kvDimensions]
      var values = projected[.ellipsis, self.kvDimensions...]
      keys = unflatten(keys, axis: -1, shape: [self.kvHeads, self.headDimensions])
        .transposed(0, 2, 1, 3)
      values = unflatten(values, axis: -1, shape: [self.kvHeads, self.headDimensions])
        .transposed(0, 2, 1, 3)
      return ProjectedAttentionKV(keys: self.keyNorm(keys), values: values)
    }

    func callAsFunction(
      _ input: MLXArray,
      projectedKV: ProjectedAttentionKV,
      mask: MLXArray
    ) -> MLXArray {
      var queries = self.queryProjection(input)
      queries = unflatten(queries, axis: -1, shape: [self.heads, self.headDimensions])
        .transposed(0, 2, 1, 3)
      queries = self.queryNorm(queries)

      let output = MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: projectedKV.keys,
        values: projectedKV.values,
        scale: self.scale,
        mask: .array(mask)
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

    static func fromTable(_ table: MLXArray, position: MLXArray, dtype: DType) -> Self {
      let frequencies = table.take(position, axis: 1).asType(dtype)
      return Self(sin: frequencies[0..<1, 0..., 0...], cos: frequencies[1..<2, 0..., 0...])
    }

    private init(sin: MLXArray, cos: MLXArray) {
      self.sin = sin
      self.cos = cos
    }

    func apply(to input: MLXArray) -> MLXArray {
      let (firstHalf, secondHalf) = input.split(axis: -1)
      let rotated = concatenated([-secondHalf, firstHalf], axis: -1)
      return (input * expandedDimensions(self.cos, axis: 1))
        + (rotated * expandedDimensions(self.sin, axis: 1))
    }
  }

  // MARK: - KVCache

  private final class NeedleKVCache: KVCache {
    struct State {
      let keys: MLXArray
      let values: MLXArray
    }

    var offset = 0
    let kvHeads: Int
    let headDimensions: Int
    let initialCapacity: Int
    let maximumCapacity: Int
    private(set) var capacity: Int
    private var keys: MLXArray
    private var values: MLXArray

    var maxSize: Int? { self.maximumCapacity }

    init(configuration: NeedleModelConfiguration, dtype: DType) {
      let capacity = min(128, configuration.decoderMaxLength)
      let shape = [1, configuration.kvHeads, capacity, configuration.attentionHeadDimensions]
      self.kvHeads = configuration.kvHeads
      self.headDimensions = configuration.attentionHeadDimensions
      self.initialCapacity = capacity
      self.maximumCapacity = configuration.decoderMaxLength
      self.capacity = capacity
      self.keys = MLXArray.zeros(shape, dtype: dtype)
      self.values = MLXArray.zeros(shape, dtype: dtype)
    }

    func innerState() -> [MLXArray] {
      [self.keys, self.values]
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
      let newOffset = self.offset + keys.dim(2)
      self.ensureCapacity(required: newOffset, dtype: keys.dtype)
      self.keys[.ellipsis, self.offset..<newOffset, 0...] = keys
      self.values[.ellipsis, self.offset..<newOffset, 0...] = values
      self.offset = newOffset
      let keys = self.keys[.ellipsis, ..<self.offset, 0...]
      let values = self.values[.ellipsis, ..<self.offset, 0...]
      return (keys, values)
    }

    var state: [MLXArray] {
      get {
        [self.keys[.ellipsis, ..<self.offset, 0...], self.values[.ellipsis, ..<self.offset, 0...]]
      }
      set {
        precondition(newValue.count == 2, "NeedleKVCache state must contain keys and values.")
        self.offset = newValue[0].dim(2)
        precondition(self.offset <= self.maximumCapacity, "Needle KV cache capacity exceeded.")
        self.capacity = self.initialCapacity
        while self.capacity < self.offset {
          self.capacity = min(self.maximumCapacity, self.capacity * 2)
        }
        self.keys = self.padded(newValue[0], capacity: self.capacity)
        self.values = self.padded(newValue[1], capacity: self.capacity)
      }
    }

    var metaState: [String] {
      get { [""] }
      set { _ = newValue }
    }

    var isTrimmable: Bool { true }

    @discardableResult
    func trim(_ tokenCount: Int) -> Int {
      let trimmed = min(self.offset, tokenCount)
      self.offset -= trimmed
      return trimmed
    }

    func copy() -> any KVCache {
      NeedleKVCache(
        kvHeads: self.kvHeads,
        headDimensions: self.headDimensions,
        initialCapacity: self.initialCapacity,
        maximumCapacity: self.maximumCapacity,
        capacity: self.capacity,
        offset: self.offset,
        keys: self.keys[.ellipsis],
        values: self.values[.ellipsis]
      )
    }

    func makeMask(
      n tokenCount: Int,
      windowSize: Int?,
      returnArray _: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
      tokenCount == 1
        ? .none
        : .array(createCausalMask(n: tokenCount, offset: self.offset, windowSize: windowSize))
    }

    func ensureCapacity(required: Int, dtype: DType) {
      precondition(required <= self.maximumCapacity, "Needle KV cache capacity exceeded.")
      while self.capacity < required {
        self.capacity = min(self.maximumCapacity, self.capacity * 2)
      }
      guard self.keys.dim(2) != self.capacity else { return }

      let added = self.capacity - self.keys.dim(2)
      let newKeys = MLXArray.zeros([1, self.kvHeads, added, self.headDimensions], dtype: dtype)
      self.keys = concatenated([self.keys, newKeys], axis: 2)

      let newValues = MLXArray.zeros([1, self.kvHeads, added, self.headDimensions], dtype: dtype)
      self.values = concatenated([self.values, newValues], axis: 2)
    }

    func fullState(dtype: DType) -> State {
      self.ensureCapacity(required: self.offset + 1, dtype: dtype)
      return State(keys: self.keys, values: self.values)
    }

    func replace(state: State, offset: Int) {
      self.keys = state.keys
      self.values = state.values
      self.offset = offset
      self.capacity = state.keys.dim(2)
    }

    private init(
      kvHeads: Int,
      headDimensions: Int,
      initialCapacity: Int,
      maximumCapacity: Int,
      capacity: Int,
      offset: Int,
      keys: MLXArray,
      values: MLXArray
    ) {
      self.kvHeads = kvHeads
      self.headDimensions = headDimensions
      self.initialCapacity = initialCapacity
      self.maximumCapacity = maximumCapacity
      self.capacity = capacity
      self.offset = offset
      self.keys = keys
      self.values = values
    }

    private func padded(_ array: MLXArray, capacity: Int) -> MLXArray {
      guard array.dim(2) < capacity else { return array }
      let padding = MLXArray.zeros(
        [array.dim(0), array.dim(1), capacity - array.dim(2), array.dim(3)],
        dtype: array.dtype
      )
      return concatenated([array, padding], axis: 2)
    }
  }

  // MARK: - Helpers

  private func fuseNeedleAttentionWeights(_ weights: inout [String: MLXArray]) {
    for queryKey in Array(weights.keys) where queryKey.contains(".self_attn.q_proj.") {
      let keyKey = queryKey.replacingOccurrences(of: ".q_proj.", with: ".k_proj.")
      let valueKey = queryKey.replacingOccurrences(of: ".q_proj.", with: ".v_proj.")
      guard let query = weights[queryKey],
        let key = weights[keyKey],
        let value = weights[valueKey]
      else { continue }

      let fusedKey = queryKey.replacingOccurrences(of: ".q_proj.", with: ".qkv_proj.")
      weights[fusedKey] = concatenated([query, key, value], axis: 0)
      weights.removeValue(forKey: queryKey)
      weights.removeValue(forKey: keyKey)
      weights.removeValue(forKey: valueKey)
    }

    for keyKey in Array(weights.keys) where keyKey.contains(".encoder_attn.k_proj.") {
      let valueKey = keyKey.replacingOccurrences(of: ".k_proj.", with: ".v_proj.")
      guard let key = weights[keyKey], let value = weights[valueKey] else { continue }

      let fusedKey = keyKey.replacingOccurrences(of: ".k_proj.", with: ".kv_proj.")
      weights[fusedKey] = concatenated([key, value], axis: 0)
      weights.removeValue(forKey: keyKey)
      weights.removeValue(forKey: valueKey)
    }
  }

  private func gatedResidual(
    _ input: MLXArray,
    gate: MLXArray,
    sublayer: MLXArray
  ) -> MLXArray {
    let residual = input + sigmoid(gate).asType(sublayer.dtype) * sublayer
    return clip(
      residual,
      min: -Float.needleClippingMagnitude,
      max: Float.needleClippingMagnitude
    )
  }

  private func paddingMask(inputIds: MLXArray, padTokenId: Int) -> MLXArray {
    (inputIds .!= padTokenId)[.ellipsis, .newAxis, .newAxis, 0...]
  }
#endif
