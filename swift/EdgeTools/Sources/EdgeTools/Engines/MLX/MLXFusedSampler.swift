#if MLX && canImport(MLX)
  import EdgeToolsCore
  import MLX
  import MLXLMCommon
  import MLXNN

  // MARK: - MLXFusedSampler

  public final class MLXFusedSampler: LogitSampler {
    public let parameters: EdgeToolsFusedSamplingParameters
    public let history: MLXTokenHistory

    private var prngKey: MLXArray
    private let unpenalized: (MLXArray, MLXArray) -> MLXArray
    private let penalized: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray

    public convenience init(parameters: EdgeToolsFusedSamplingParameters) {
      self.init(
        parameters: parameters,
        history: MLXTokenHistory(capacity: parameters.repetitionContextSize ?? 20)
      )
    }

    public init(parameters: EdgeToolsFusedSamplingParameters, history: MLXTokenHistory) {
      self.parameters = parameters
      self.history = history
      self.prngKey = MLXRandom.key(parameters.seed ?? UInt64.random(in: 0...UInt64.max))
      self.unpenalized = compile {
        sampledToken(logits: $0, history: nil, prngKey: $1, parameters: parameters)
      }
      self.penalized = compile {
        sampledToken(logits: $0, history: $1, prngKey: $2, parameters: parameters)
      }
    }

    public func sample(logits: MLXArray) -> MLXArray {
      let isPenalized = self.parameters.penalizesHistory
      let prngKey = self.nextPRNGKey()
      let token =
        (isPenalized ? self.history.tokens : nil)
        .map { self.penalized(logits, $0, prngKey) } ?? self.unpenalized(logits, prngKey)
      if isPenalized {
        self.history.append(token)
      }
      return token
    }

    private func nextPRNGKey() -> MLXArray {
      guard !self.parameters.isGreedy else { return self.prngKey }
      let (next, subkey) = MLXRandom.split(key: self.prngKey)
      self.prngKey = next
      return subkey
    }
  }

  // MARK: - MLXTokenHistory

  public final class MLXTokenHistory {
    public let capacity: Int

    private var buffer: MLXArray?
    private var writeIndex = 0
    private let positions: MLXArray

    public init(capacity: Int) {
      precondition(capacity > 0, "Token history capacity must be greater than zero.")
      self.capacity = capacity
      self.positions = MLXArray(0..<capacity)
    }

    public var tokens: MLXArray? {
      self.buffer
    }

    public func reset() {
      self.buffer = nil
      self.writeIndex = 0
    }

    public func seed(_ tokens: some Sequence<EdgeToolsToken.ID>) {
      self.reset()
      let ids = Array(tokens.map { Int32($0) }.suffix(self.capacity))
      guard let last = ids.last else { return }
      var padded = [Int32](repeating: last, count: self.capacity)
      padded.replaceSubrange(0..<ids.count, with: ids)
      self.buffer = MLXArray(padded)
      self.writeIndex = ids.count % self.capacity
    }

    public func append(_ token: MLXArray) {
      let token = token.asType(.int32).reshaped([1])
      if let buffer = self.buffer {
        self.buffer = MLX.where(self.positions .== Int32(self.writeIndex), token, buffer)
      } else {
        self.buffer = broadcast(token, to: [self.capacity])
      }
      self.writeIndex = (self.writeIndex + 1) % self.capacity
    }
  }

  // MARK: - Graph

  private func sampledToken(
    logits: MLXArray,
    history: MLXArray?,
    prngKey: MLXArray,
    parameters: EdgeToolsFusedSamplingParameters
  ) -> MLXArray {
    var logits = logits.dtype == .bfloat16 ? logits.asType(.float32) : logits
    if let history {
      let indices = history.asType(.uint32)[.newAxis, 0...]
      let selected = takeAlong(logits, indices, axis: -1)
      let repetitionPenalty = parameters.repetitionPenalty ?? 1
      let scaled = MLX.where(
        selected .< 0,
        selected * repetitionPenalty,
        selected / repetitionPenalty
      )
      let penalized = parameters.presencePenalty.map { scaled - $0 } ?? scaled
      logits = putAlong(logits, indices, values: penalized, axis: -1)
    }
    guard !parameters.isGreedy else { return argMax(logits, axis: -1) }

    let logProbabilities = logSoftmax(logits, axis: -1)
    let temperature = 1 / (parameters.temperature ?? 0.6)
    guard let topK = parameters.topK, topK > 0, topK < logits.dim(-1) else {
      return categorical(
        filteredLogProbabilities(logProbabilities, parameters: parameters) * temperature,
        key: prngKey
      )
    }

    let partitioned = argPartition(-logProbabilities, kth: topK - 1, axis: -1)[0..., 0..<topK]
    let candidates = takeAlong(logProbabilities, partitioned, axis: -1)
    let descending = argSort(-candidates, axis: -1)
    let sorted = takeAlong(candidates, descending, axis: -1)
    let ids = takeAlong(partitioned, descending, axis: -1)

    var keep: MLXArray?
    if let topP = parameters.topP, topP < 1 {
      let probabilities = exp(sorted)
      let exclusive = cumsum(probabilities, axis: -1) - probabilities
      keep = (exclusive .< topP)
    }
    if let minP = parameters.minP, minP > 0 {
      let survives = sorted .>= sorted[0..., ..<1] + log(MLXArray(minP))
      keep = keep.map { $0 .&& survives } ?? survives
    }

    let filtered = keep.map { MLX.where($0, sorted, -Float.infinity) } ?? sorted
    let choice = categorical(filtered * temperature, key: prngKey)
    return takeAlong(ids, choice[0..., .newAxis], axis: -1).squeezed(axis: -1)
  }

  private func filteredLogProbabilities(
    _ logProbabilities: MLXArray,
    parameters: EdgeToolsFusedSamplingParameters
  ) -> MLXArray {
    var logProbabilities = logProbabilities
    if let minP = parameters.minP, minP > 0 {
      let threshold = logProbabilities.max(axis: -1, keepDims: true) + log(MLXArray(minP))
      logProbabilities = MLX.where(
        logProbabilities .>= threshold,
        logProbabilities,
        -Float.infinity
      )
    }
    guard let topP = parameters.topP, topP < 1 else { return logProbabilities }

    let ascending = argSort(logProbabilities, axis: -1)
    let sorted = takeAlong(logProbabilities, ascending, axis: -1)
    let cumulative = cumsum(exp(sorted), axis: -1)
    let filtered = MLX.where(cumulative .> (1 - topP), sorted, -Float.infinity)
    return putAlong(logProbabilities, ascending, values: filtered, axis: -1)
  }
#endif
