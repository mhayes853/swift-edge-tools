// MARK: - EdgeToolsSampler

public struct EdgeToolsSampler<Logits: ~Copyable & ~Escapable>: Sendable {
  private let body:
    nonisolated(nonsending) @Sendable (
      borrowing Logits
    ) async throws -> EdgeToolsToken.ID

  public init(
    _ body:
      nonisolated(nonsending) @escaping @Sendable (
        borrowing Logits
      ) async throws -> EdgeToolsToken.ID
  ) {
    self.body = body
  }

  public nonisolated(nonsending) func sample(
    logits: borrowing Logits
  ) async throws -> EdgeToolsToken.ID {
    try await self.body(logits)
  }
}

// MARK: - EdgeToolsFusedSamplingParameters

public struct EdgeToolsFusedSamplingParameters: Hashable, Sendable {
  public var temperature: Float?
  public var topK: Int?
  public var topP: Float?
  public var minP: Float?
  public var repetitionPenalty: Float?
  public var presencePenalty: Float?
  public var repetitionContextSize: Int?
  public var seed: UInt64?

  public static var greedy: Self {
    Self(temperature: 0)
  }

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
    precondition(
      repetitionPenalty.map { $0 > 0 } ?? true,
      "Repetition penalty must be greater than zero."
    )
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
