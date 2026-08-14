import _Concurrency

// MARK: - EdgeToolsGenerationLoop

public struct EdgeToolsGenerationLoop: Sendable {
  public struct Preparation: Sendable {
    public var metrics: EdgeToolsPrefillMetrics
    public var metadata: EdgeToolsMetadata

    public init(
      metrics: EdgeToolsPrefillMetrics,
      metadata: EdgeToolsMetadata = EdgeToolsMetadata()
    ) {
      self.metrics = metrics
      self.metadata = metadata
    }
  }

  private let tokenizer: any EdgeToolsTokenizer
  public let stopTokenIds: Set<EdgeToolsToken.ID>

  public init(
    tokenizer: any EdgeToolsTokenizer,
    extraStopTokenIds: Set<EdgeToolsToken.ID> = []
  ) {
    var stopTokenIds = extraStopTokenIds
    if let eosTokenId = tokenizer.eosTokenId {
      stopTokenIds.insert(eosTokenId)
    }
    self.tokenizer = tokenizer
    self.stopTokenIds = stopTokenIds
  }

  public func run<State, Parser, GrammarEngine>(
    state: inout State,
    stopper: AnyGenerationTask.Stopper,
    channel: EdgeToolsGenerationChannel,
    grammarEngine: GrammarEngine,
    maximumTokenCount: Int? = nil,
    grammar: (State) throws -> GrammarEngine.Grammar,
    prepare: (inout Parser, inout State) async throws -> Preparation,
    decode: (GrammarBitmask, inout State) async throws -> EdgeToolsToken.ID
  ) async throws -> EdgeToolsEngineGeneration
  where Parser: EdgeToolsGenerationParser, GrammarEngine: EdgeToolsGrammarEngine {
    try Task.checkCancellation()
    guard !stopper.isStopped else {
      return .empty
    }

    let clock = GenerationClock()
    let generateStart = clock.now
    var parser = Parser()
    let preparation = try await prepare(&parser, &state)
    var matcher = try grammarEngine.matcher(
      for: grammar(state),
      stopTokenIds: self.stopTokenIds
    )
    var detokenizer = StreamingDetokenizer()
    var generatedTokens = [EdgeToolsToken]()
    var parts = [EdgeToolsGenerationPart]()
    var durationToFirstToken: Duration?
    let maximumTokenCount = maximumTokenCount ?? .max

    var bitmask: GrammarBitmask?
    if !matcher.isTerminated,
      !stopper.isStopped,
      generatedTokens.count < maximumTokenCount,
      generatedTokens.last.map({ !self.stopTokenIds.contains($0.id) }) ?? true
    {
      try Task.checkCancellation()
      bitmask = matcher.grammarBitmask()
    }

    while let currentBitmask = bitmask {
      let tokenId = try await decode(currentBitmask, &state)
      durationToFirstToken = durationToFirstToken ?? clock.duration(since: generateStart)

      let tokenString = detokenizer.decode(tokenId: tokenId, using: self.tokenizer)
      let token = EdgeToolsToken(id: tokenId, stringValue: tokenString)
      generatedTokens.append(token)
      guard matcher.accept(tokenId: token.id) else {
        throw EdgeToolsError.grammarRejectedToken(token: token)
      }

      channel.emit(token: token)
      let parsedParts =
        self.stopTokenIds.contains(token.id)
        ? []
        : parser.accept(token: token)
      for part in parsedParts {
        parts.append(part)
        channel.emit(part: part)
      }

      bitmask = nil
      if !matcher.isTerminated,
        !stopper.isStopped,
        generatedTokens.count < maximumTokenCount,
        generatedTokens.last.map({ !self.stopTokenIds.contains($0.id) }) ?? true
      {
        try Task.checkCancellation()
        bitmask = matcher.grammarBitmask()
      }
    }

    for part in parser.finish() {
      parts.append(part)
      channel.emit(part: part)
    }

    let finalDurationToFirstToken = durationToFirstToken ?? .zero
    var responseTokenIds = detokenizer.tokenIds
    if let lastTokenId = responseTokenIds.last,
      self.stopTokenIds.contains(lastTokenId)
    {
      responseTokenIds.removeLast()
    }
    let response = self.tokenizer.decode(tokens: responseTokenIds)
    let decodeDuration = clock.duration(since: generateStart) - finalDurationToFirstToken
    return EdgeToolsEngineGeneration(
      prefillMetrics: preparation.metrics,
      decodeMetrics: EdgeToolsDecodeMetrics(
        tokens: generatedTokens.count,
        duration: decodeDuration,
        durationToFirstToken: finalDurationToFirstToken
      ),
      wasStopped: stopper.isStopped,
      tokens: generatedTokens,
      response: response,
      parts: parts,
      metadata: preparation.metadata
    )
  }
}

private struct GenerationClock {
  #if $Embedded
    struct Instant {}

    var now: Instant {
      Instant()
    }

    func duration(since instant: Instant) -> Duration {
      .zero
    }
  #else
    typealias Instant = ContinuousClock.Instant

    private let clock = ContinuousClock()

    var now: Instant {
      self.clock.now
    }

    func duration(since instant: Instant) -> Duration {
      instant.duration(to: self.clock.now)
    }
  #endif
}
