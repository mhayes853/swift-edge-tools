public struct EdgeToolsFusedSamplingParameters: Hashable, Sendable {
  /// The sampling temperature. Non-`nil` values must be finite and nonnegative.
  public var temperature: Float? {
    didSet {
      preconditionValidTemperature(self.temperature)
    }
  }
  /// The number of highest-probability tokens to sample. Non-`nil` values must be nonnegative.
  public var topK: Int? {
    didSet {
      preconditionValidTopK(self.topK)
    }
  }
  /// The nucleus sampling threshold. Non-`nil` values must be greater than zero and at most one.
  public var topP: Float? {
    didSet {
      preconditionValidTopP(self.topP)
    }
  }
  /// The minimum relative probability threshold. Non-`nil` values must be between zero and one.
  public var minP: Float? {
    didSet {
      preconditionValidMinP(self.minP)
    }
  }
  /// The repetition penalty. Non-`nil` values must be finite and greater than zero.
  public var repetitionPenalty: Float? {
    didSet {
      preconditionValidRepetitionPenalty(self.repetitionPenalty)
    }
  }
  /// The presence penalty. Non-`nil` values must be finite.
  public var presencePenalty: Float? {
    didSet {
      preconditionValidPresencePenalty(self.presencePenalty)
    }
  }
  /// The number of recent tokens considered for repetition penalties. Non-`nil` values must be
  /// positive.
  public var repetitionContextSize: Int? {
    didSet {
      preconditionValidRepetitionContextSize(self.repetitionContextSize)
    }
  }
  public var seed: UInt64?

  public static var greedy: Self {
    Self(temperature: 0)
  }

  /// Creates fused sampling parameters.
  ///
  /// - Precondition: Each non-`nil` option meets the bounds documented on its property.
  public init(
    temperature: Float? = nil,
    topK: Int? = nil,
    topP: Float? = nil,
    minP: Float? = nil,
    repetitionPenalty: Float? = nil,
    presencePenalty: Float? = nil,
    repetitionContextSize: Int? = nil,
    seed: UInt64? = nil
  ) {
    preconditionValidTemperature(temperature)
    preconditionValidTopK(topK)
    preconditionValidTopP(topP)
    preconditionValidMinP(minP)
    preconditionValidRepetitionPenalty(repetitionPenalty)
    preconditionValidPresencePenalty(presencePenalty)
    preconditionValidRepetitionContextSize(repetitionContextSize)
    self.temperature = temperature
    self.topK = topK
    self.topP = topP
    self.minP = minP
    self.repetitionPenalty = repetitionPenalty
    self.presencePenalty = presencePenalty
    self.repetitionContextSize = repetitionContextSize
    self.seed = seed
  }

  public var isGreedy: Bool {
    self.temperature == 0
  }

  public var penalizesHistory: Bool {
    (self.repetitionPenalty ?? 1) != 1 || (self.presencePenalty ?? 0) != 0
  }

  public var isEmpty: Bool {
    self.temperature == nil
      && self.topK == nil
      && self.topP == nil
      && self.minP == nil
      && self.repetitionPenalty == nil
      && self.presencePenalty == nil
      && self.repetitionContextSize == nil
      && self.seed == nil
  }

  public func applying(
    to parameters: EdgeToolsFusedSamplingParameters
  ) -> EdgeToolsFusedSamplingParameters {
    var parameters = parameters
    parameters.temperature = self.temperature ?? parameters.temperature
    parameters.topK = self.topK ?? parameters.topK
    parameters.topP = self.topP ?? parameters.topP
    parameters.minP = self.minP ?? parameters.minP
    parameters.repetitionPenalty = self.repetitionPenalty ?? parameters.repetitionPenalty
    parameters.presencePenalty = self.presencePenalty ?? parameters.presencePenalty
    parameters.repetitionContextSize =
      self.repetitionContextSize ?? parameters.repetitionContextSize
    parameters.seed = self.seed ?? parameters.seed
    return parameters
  }
}

private func preconditionValidTemperature(_ temperature: Float?) {
  precondition(
    temperature.map { $0.isFinite && $0 >= 0 } ?? true,
    "Temperature must be finite and nonnegative."
  )
}

private func preconditionValidTopK(_ topK: Int?) {
  precondition(topK.map { $0 >= 0 } ?? true, "Top K must be nonnegative.")
}

private func preconditionValidTopP(_ topP: Float?) {
  precondition(
    topP.map { $0 > 0 && $0 <= 1 } ?? true,
    "Top P must be greater than zero and at most one."
  )
}

private func preconditionValidMinP(_ minP: Float?) {
  precondition(
    minP.map { $0 >= 0 && $0 <= 1 } ?? true,
    "Min P must be between zero and one."
  )
}

private func preconditionValidRepetitionPenalty(_ repetitionPenalty: Float?) {
  precondition(
    repetitionPenalty.map { $0.isFinite && $0 > 0 } ?? true,
    "Repetition penalty must be finite and greater than zero."
  )
}

private func preconditionValidPresencePenalty(_ presencePenalty: Float?) {
  precondition(
    presencePenalty.map { $0.isFinite } ?? true,
    "Presence penalty must be finite."
  )
}

private func preconditionValidRepetitionContextSize(_ repetitionContextSize: Int?) {
  precondition(
    repetitionContextSize.map { $0 > 0 } ?? true,
    "Repetition context size must be greater than zero."
  )
}
