#if SwiftNeedleMLX
  import MLX
  import MLXNN
  import MLXLLM
  import MLXLMCommon
  import Foundation
  import Tokenizers

  // MARK: - NeedleMLXModel

  public final class NeedleMLXModel: Module, LanguageModel, KVCacheDimensionProvider {
    private static let attentionLayersPerDecoder = 2

    public var kvHeads: [Int] {
      [Int](
        repeating: self.configuration.kvHeads,
        count: self.configuration.hiddenLayers * Self.attentionLayersPerDecoder
      )
    }

    public let configuration: NeedleModelConfiguration
    private let model: NeedleSimpleAttentionNetwork

    public init(configuration: NeedleModelConfiguration) {
      self.configuration = configuration
      self.model = NeedleSimpleAttentionNetwork(configuration: configuration)
    }

    public func prepare(
      _ input: LMInput,
      cache: [any KVCache],
      windowSize: Int?
    ) throws -> PrepareResult {
      let prefillStepSize = windowSize ?? 512
      var y = input.text

      var lastOutput: LMOutput?
      while y.tokens.size > 0 {
        let count = min(prefillStepSize, y.tokens.size)
        lastOutput = self.model(
          y[.newAxis, ..<count].tokens,
          prefilledOutput: lastOutput?.state?.crossAttentionStates ?? .mlxNone,
          caches: self.caches(from: cache),
          forwardPhase: .prefill
        )
        y = y[count...]
      }

      eval(cache)

      guard let lastOutput else { return .tokens(input.text) }
      return .logits(lastOutput)
    }

    public func callAsFunction(
      _ input: LMInput.Text,
      cache: [any KVCache]?,
      state: LMOutput.State?
    ) -> LMOutput {
      self.model(
        input.tokens,
        prefilledOutput: state?.crossAttentionStates ?? .mlxNone,
        caches: cache.map(self.caches(from:)),
        forwardPhase: .decode
      )
    }

    private func caches(
      from caches: [any KVCache]
    ) -> [(selfCache: any KVCache, crossCache: any KVCache)] {
      stride(from: 0, to: caches.count, by: 2).map { (caches[$0], caches[$0 + 1]) }
    }
  }

  // MARK: - Prompt Formatting

  extension NeedleMLXModel {
    public static func input(
      from prompt: NeedlePrompt,
      using tokenizer: borrowing some TokenizingModel
    ) throws -> LMInput {
      let tokenStrings = try tokenizer.tokenize(text: prompt.formatted())
      let tokens = tokenStrings.compactMap(tokenizer.convertTokenToId)
      return LMInput(text: LMInput.Text(tokens: MLXArray(tokens)))
    }
  }

  // MARK: - SimpleAttentionNetwork

  private final class NeedleSimpleAttentionNetwork: Module {
    @ModuleInfo(key: "embed_tokens") var embedding: Embedding
    let encoder: NeedleEncoder
    let decoder: NeedleDecoder
    private let embedScale: Float
    private let padTokenId: Int

    init(configuration: NeedleModelConfiguration) {
      self.encoder = NeedleEncoder(configuration: configuration)
      self.decoder = NeedleDecoder(configuration: configuration)
      self._embedding.wrappedValue = Embedding(
        embeddingCount: configuration.vocabularySize,
        dimensions: configuration.dimensions
      )
      self.embedScale = sqrt(Float(configuration.dimensions))
      self.padTokenId = configuration.padTokenId
    }

    func callAsFunction(
      _ input: MLXArray,
      prefilledOutput: MLXArray,
      caches: [(selfCache: any KVCache, crossCache: any KVCache)]?,
      forwardPhase: ForwardPhase
    ) -> LMOutput {
      let prefilled =
        switch forwardPhase {
        case .prefill:
          self.encode(input, previous: prefilledOutput)
        case .decode:
          prefilledOutput
        }
      let logits = self.decoder(
        self.embedding(input) * self.embedScale,
        encoderOutput: prefilled,
        selfMask: .causal,
        crossMask: .none,
        caches: caches
      )
      return LMOutput(logits: logits, state: LMOutput.State(crossAttentionStates: prefilled))
    }

    private func encode(_ input: MLXArray, previous: MLXArray) -> MLXArray {
      let mask = paddingMask(inputIds: input, padTokenId: self.padTokenId)
      let encoded = self.encoder(self.embedding(input) * self.embedScale, mask: .array(mask))
      return concatenated([previous, encoded], axis: 1)
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
      _ x: MLXArray,
      mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
      let ropeFrequencies = RoPEFrequencies(
        inverse: self._inverseRope,
        sequenceLength: x.dim(1),
        dtype: x.dtype
      )
      var x = x
      for layer in self.layers {
        x = layer(x, mask: mask, ropeFrequencies: ropeFrequencies)
      }
      return self.finalNorm(x)
    }
  }

  private final class NeedleEncoderBlock: Module {
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: ZCRMSNorm
    @ModuleInfo(key: "self_attn") private var attention: NeedleAttention
    @ParameterInfo(key: "attn_gate") private var attentionGate: MLXArray

    init(configuration: NeedleModelConfiguration) {
      self._inputLayerNorm.wrappedValue = ZCRMSNorm(configuration: configuration)
      self._attention.wrappedValue = NeedleAttention(configuration: configuration)
      self._attentionGate.wrappedValue = MLXArray(0)
    }

    func callAsFunction(
      _ x: MLXArray,
      mask: MLXFast.ScaledDotProductAttentionMaskMode,
      ropeFrequencies: RoPEFrequencies
    ) -> MLXArray {
      let normed = self.inputLayerNorm(x)
      let attention = self.attention(
        q: normed,
        kv: normed,
        mask: mask,
        ropeFrequencies: ropeFrequencies,
        cache: nil
      )
      return gatedResidual(x, gate: self.attentionGate, sublayer: attention)
    }
  }

  // MARK: - Decoder

  private final class NeedleDecoder: Module {
    let layers: [NeedleDecoderBlock]
    @ModuleInfo(key: "norm") var finalNorm: ZCRMSNorm
    private let _inverseRope: MLXArray

    init(configuration: NeedleModelConfiguration) {
      self.layers = (0..<configuration.decoderLayers)
        .map { _ in NeedleDecoderBlock(configuration: configuration) }
      self._finalNorm.wrappedValue = ZCRMSNorm(configuration: configuration)
      self._inverseRope = RoPEFrequencies.inverse(
        headDimensions: configuration.attentionHeadDimensions,
        theta: configuration.ropeTheta
      )
    }

    func callAsFunction(
      _ x: MLXArray,
      encoderOutput: MLXArray,
      selfMask: MLXFast.ScaledDotProductAttentionMaskMode,
      crossMask: MLXFast.ScaledDotProductAttentionMaskMode,
      caches: [(selfCache: any KVCache, crossCache: any KVCache)]?,
    ) -> MLXArray {
      let ropeFrequencies = RoPEFrequencies(
        inverse: self._inverseRope,
        sequenceLength: x.dim(1),
        dtype: x.dtype
      )
      var x = x
      let caches = caches ?? Array(repeating: nil, count: self.layers.count)
      for (layer, caches) in zip(self.layers, caches) {
        x = layer(
          x,
          encoderOutput: encoderOutput,
          selfMask: selfMask,
          crossMask: crossMask,
          ropeFrequencies: ropeFrequencies,
          selfCache: caches?.selfCache,
          crossCache: caches?.crossCache
        )
      }
      return self.finalNorm(x)
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
      self._selfAttentionGate.wrappedValue = MLXArray(0)
      self._crossAttentionNorm.wrappedValue = ZCRMSNorm(configuration: configuration)
      self._crossAttention.wrappedValue = NeedleAttention(configuration: configuration)
      self._crossAttentionGate.wrappedValue = MLXArray(0)
    }

    func callAsFunction(
      _ x: MLXArray,
      encoderOutput: MLXArray,
      selfMask: MLXFast.ScaledDotProductAttentionMaskMode,
      crossMask: MLXFast.ScaledDotProductAttentionMaskMode,
      ropeFrequencies: RoPEFrequencies,
      selfCache: (any KVCache)?,
      crossCache: (any KVCache)?
    ) -> MLXArray {
      let selfNormed = self.inputLayerNorm(x)
      let selfAttention = self.selfAttention(
        q: selfNormed,
        kv: selfNormed,
        mask: selfMask,
        ropeFrequencies: ropeFrequencies,
        cache: selfCache
      )

      let crossNormed = self.crossAttentionNorm(
        gatedResidual(x, gate: self.selfAttentionGate, sublayer: selfAttention)
      )
      let crossAttention = self.crossAttention(
        q: crossNormed,
        kv: encoderOutput,
        mask: crossMask,
        ropeFrequencies: nil,
        cache: crossCache
      )
      return gatedResidual(x, gate: self.crossAttentionGate, sublayer: crossAttention)
    }
  }

  // MARK: - MultiheadAttention

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

      self._queryNorm.wrappedValue = ZCRMSNorm(configuration: configuration)
      self._keyNorm.wrappedValue = ZCRMSNorm(configuration: configuration)

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
      q: MLXArray,
      kv: MLXArray,
      mask: MLXFast.ScaledDotProductAttentionMaskMode,
      ropeFrequencies: RoPEFrequencies?,
      cache: (any KVCache)?
    ) -> MLXArray {
      var queries = self.queryProjection(q)
      var keys = self.keyProjection(kv)
      var values = self.valueProjection(kv)

      queries = self.queryNorm(queries)
      keys = self.keyNorm(keys)
      queries = unflatten(queries, axis: -1, shape: [self.heads, self.headDimensions])
        .transposed(0, 2, 1, 3)
      keys = unflatten(keys, axis: -1, shape: [self.kvHeads, self.headDimensions])
        .transposed(0, 2, 1, 3)
      values = unflatten(values, axis: -1, shape: [self.kvHeads, self.headDimensions])
        .transposed(0, 2, 1, 3)

      if let ropeFrequencies {
        queries = ropeFrequencies.apply(to: queries, offset: cache?.offset)
        keys = ropeFrequencies.apply(to: keys, offset: cache?.offset)
      }

      let output = attentionWithCacheUpdate(
        queries: queries,
        keys: keys,
        values: values,
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
      self.weight = .ones([dimensions])
      super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
      (self.weight + 1) * x / rmsNorm(x, weight: self.weight, eps: self.eps)
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

    init(inverse: MLXArray, sequenceLength: Int, dtype: DType) {
      let positions = MLXArray(0..<sequenceLength).asType(.float32)
      let frequencies =
        expandedDimensions(positions, axis: 1) * expandedDimensions(inverse, axis: 0)
      let embeddings = concatenated([frequencies, frequencies], axis: -1)
      self.sin = expandedDimensions(MLX.sin(embeddings), axis: 0).asType(dtype)
      self.cos = expandedDimensions(MLX.cos(embeddings), axis: 0).asType(dtype)
    }

    func apply(to x: MLXArray, offset: Int?) -> MLXArray {
      let offset = offset ?? 0
      let sin = expandedDimensions(
        self.sin[.ellipsis, offset..<offset + x.dim(1), .ellipsis],
        axis: 2
      )
      let cos = expandedDimensions(
        self.cos[.ellipsis, offset..<offset + x.dim(1), .ellipsis],
        axis: 2
      )
      let half = x.dim(-1) / 2
      let rotated = concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
      return (x * cos) + (rotated * sin)
    }
  }

  // MARK: - Gated Residual

  private func gatedResidual(_ x: MLXArray, gate: MLXArray, sublayer: MLXArray) -> MLXArray {
    x + sigmoid(gate).asType(sublayer.dtype) * sublayer
  }

  // MARK: - Padding Mask

  private func paddingMask(inputIds: MLXArray, padTokenId: Int) -> MLXArray {
    (inputIds .!= padTokenId)[0..., .newAxis, .newAxis, 0...]
  }

  // MARK: - ForwardPhase

  private enum ForwardPhase {
    case prefill, decode
  }
#endif
