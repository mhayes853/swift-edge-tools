import EdgeToolsCore
import EdgeToolsTokenizers
import _Concurrency

// MARK: - EdgeToolsGenerationLoop

struct EdgeToolsGenerationLoop: Sendable {
  struct Preparation: Sendable {
    var metrics: EdgeToolsMetrics

    init(metrics: EdgeToolsMetrics = EdgeToolsMetrics()) {
      self.metrics = metrics
    }
  }

  private let tokenizer: any EdgeToolsTokenizer
  let stopTokenIds: Set<EdgeToolsToken.ID>

  init(
    tokenizer: any EdgeToolsTokenizer,
    extraStopTokenIds: Set<EdgeToolsToken.ID> = []
  ) {
    var stopTokenIds = extraStopTokenIds
    if let eosTokenId = tokenizer.eos?.id {
      stopTokenIds.insert(eosTokenId)
    }
    self.tokenizer = tokenizer
    self.stopTokenIds = stopTokenIds
  }

  func run<State, Parser, GrammarEngine>(
    state: inout State,
    stopper: AnyGenerationTask.Stopper,
    channel: EdgeToolsGenerationChannel,
    grammarEngine: GrammarEngine,
    maximumTokenCount: Int? = nil,
    grammar: (State) throws -> GrammarEngine.Grammar,
    prepare: (inout Parser, inout State) async throws -> Preparation,
    decode: (GrammarBitmask?, inout State) async throws -> EdgeToolsToken.ID
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

    while !matcher.isTerminated,
      !stopper.isStopped,
      generatedTokens.count < maximumTokenCount,
      generatedTokens.last.map({ !self.stopTokenIds.contains($0.id) }) ?? true
    {
      try Task.checkCancellation()
      let tokenId = try await decode(matcher.grammarBitmask(), &state)
      durationToFirstToken = durationToFirstToken ?? clock.duration(since: generateStart)

      try self.process(
        tokenId: tokenId,
        matcher: &matcher,
        detokenizer: &detokenizer,
        generatedTokens: &generatedTokens,
        parser: &parser,
        parts: &parts,
        channel: channel
      )
    }

    return self.generation(
      stopper: stopper,
      parser: &parser,
      preparation: preparation,
      clock: clock,
      generateStart: generateStart,
      durationToFirstToken: durationToFirstToken,
      detokenizer: detokenizer,
      generatedTokens: generatedTokens,
      parts: &parts,
      channel: channel
    )
  }

  func runPrepared<State, Parser, GrammarEngine>(
    state: inout State,
    parser: inout Parser,
    preparation: Preparation,
    stopper: AnyGenerationTask.Stopper,
    channel: EdgeToolsGenerationChannel,
    grammarEngine: GrammarEngine,
    maximumTokenCount: Int? = nil,
    grammar: (State) throws -> GrammarEngine.Grammar,
    decode: (GrammarBitmask?, inout State) throws -> EdgeToolsToken.ID
  ) throws -> EdgeToolsEngineGeneration
  where Parser: EdgeToolsGenerationParser, GrammarEngine: EdgeToolsGrammarEngine {
    try Task.checkCancellation()
    guard !stopper.isStopped else {
      return .empty
    }

    let clock = GenerationClock()
    let generateStart = clock.now
    var matcher = try grammarEngine.matcher(
      for: grammar(state),
      stopTokenIds: self.stopTokenIds
    )
    var detokenizer = StreamingDetokenizer()
    var generatedTokens = [EdgeToolsToken]()
    var parts = [EdgeToolsGenerationPart]()
    var durationToFirstToken: Duration?
    let maximumTokenCount = maximumTokenCount ?? .max

    while !matcher.isTerminated,
      !stopper.isStopped,
      generatedTokens.count < maximumTokenCount,
      generatedTokens.last.map({ !self.stopTokenIds.contains($0.id) }) ?? true
    {
      try Task.checkCancellation()
      let tokenId = try decode(matcher.grammarBitmask(), &state)
      durationToFirstToken = durationToFirstToken ?? clock.duration(since: generateStart)

      try self.process(
        tokenId: tokenId,
        matcher: &matcher,
        detokenizer: &detokenizer,
        generatedTokens: &generatedTokens,
        parser: &parser,
        parts: &parts,
        channel: channel
      )
    }

    return self.generation(
      stopper: stopper,
      parser: &parser,
      preparation: preparation,
      clock: clock,
      generateStart: generateStart,
      durationToFirstToken: durationToFirstToken,
      detokenizer: detokenizer,
      generatedTokens: generatedTokens,
      parts: &parts,
      channel: channel
    )
  }

  private func process<Parser, Matcher: ~Copyable>(
    tokenId: EdgeToolsToken.ID,
    matcher: inout Matcher,
    detokenizer: inout StreamingDetokenizer,
    generatedTokens: inout [EdgeToolsToken],
    parser: inout Parser,
    parts: inout [EdgeToolsGenerationPart],
    channel: EdgeToolsGenerationChannel
  ) throws
  where Parser: EdgeToolsGenerationParser, Matcher: EdgeToolsGrammarMatcher {
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
  }

  private func generation<Parser>(
    stopper: AnyGenerationTask.Stopper,
    parser: inout Parser,
    preparation: Preparation,
    clock: GenerationClock,
    generateStart: GenerationClock.Instant,
    durationToFirstToken: Duration?,
    detokenizer: StreamingDetokenizer,
    generatedTokens: [EdgeToolsToken],
    parts: inout [EdgeToolsGenerationPart],
    channel: EdgeToolsGenerationChannel
  ) -> EdgeToolsEngineGeneration
  where Parser: EdgeToolsGenerationParser {
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
    var metrics = preparation.metrics
    metrics.decodeTokens = generatedTokens.count
    metrics.decodeDuration = decodeDuration
    metrics.durationToFirstToken = finalDurationToFirstToken
    return EdgeToolsEngineGeneration(
      wasStopped: stopper.isStopped,
      tokens: generatedTokens,
      response: response,
      parts: parts,
      metrics: metrics
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
