#if XGrammar && Atomics
  import Atomics

  struct EdgeToolsGenerationLoop<Parser: EdgeToolCallParser>: ~Copyable {
    private let tokenizer: any EdgeToolsTokenizer
    private let channel: EdgeToolsGenerationChannel
    private let isStopped: ManagedAtomic<Bool>
    private let maximumTokenCount: Int
    private let generateStart: ContinuousClock.Instant
    private let clock = ContinuousClock()

    private var matcher: XGRMatcher
    private var detokenizer = StreamingDetokenizer()
    private var parser = Parser()
    private var generatedTokens = [EdgeToolsToken]()
    private var toolCalls = [EdgeRawToolCall]()
    private var confidence = EdgeToolsConfidenceState()
    private var durationToFirstToken: Duration?

    init(
      matcher: consuming XGRMatcher,
      tokenizer: any EdgeToolsTokenizer,
      channel: EdgeToolsGenerationChannel,
      isStopped: ManagedAtomic<Bool>,
      maximumTokenCount: Int?,
      generateStart: ContinuousClock.Instant
    ) {
      self.matcher = consume matcher
      self.tokenizer = tokenizer
      self.channel = channel
      self.isStopped = isStopped
      self.maximumTokenCount = maximumTokenCount ?? .max
      self.generateStart = generateStart
    }

    var generatedTokenCount: Int {
      self.generatedTokens.count
    }

    mutating func nextBitmask() throws -> GrammarBitmask? {
      guard !self.matcher.isTerminated,
        !self.isStopped.load(ordering: .relaxed),
        self.generatedTokens.count < self.maximumTokenCount,
        self.generatedTokens.last?.id != self.tokenizer.eosTokenId
      else { return nil }
      try Task.checkCancellation()
      return self.matcher.bitmask()
    }

    mutating func accept(
      tokenID: EdgeToolsToken.ID,
      confidence: Float
    ) throws -> EdgeToolsToken {
      self.durationToFirstToken =
        self.durationToFirstToken ?? self.generateStart.duration(to: self.clock.now)
      self.confidence.add(confidence: confidence)

      let tokenString = self.detokenizer.decode(tokenId: tokenID, using: self.tokenizer)
      let token = EdgeToolsToken(id: tokenID, stringValue: tokenString)
      self.generatedTokens.append(token)
      guard self.matcher.accept(tokenId: token.id) else {
        throw EdgeToolsError.grammarRejectedToken(token: token)
      }

      let rawToolCall = self.parser.accept(token: token)
      self.channel.emit(token: token)
      if let rawToolCall {
        self.toolCalls.append(rawToolCall)
        self.channel.emit(toolCall: rawToolCall)
      }
      return token
    }

    func finish(
      prefillMetrics: EdgeToolsPrefillMetrics,
      metadata: EdgeToolsMetadata = EdgeToolsMetadata()
    ) -> EdgeToolsEngineGeneration {
      let finalDurationToFirstToken = self.durationToFirstToken ?? .zero
      var metadata = metadata
      metadata.generationConfidence = self.confidence.mean
      metadata.perTokenConfidences = self.confidence.perTokenConfidences
      let responseTokenIds = self.tokenizer.eosTokenId.map { eosTokenId in
        self.detokenizer.tokenIds.filter { $0 != eosTokenId }
      } ?? self.detokenizer.tokenIds
      let response = self.tokenizer.decode(tokens: responseTokenIds)
      let decodeDuration =
        self.generateStart.duration(to: self.clock.now) - finalDurationToFirstToken
      return EdgeToolsEngineGeneration(
        prefillMetrics: prefillMetrics,
        decodeMetrics: EdgeToolsDecodeMetrics(
          tokens: self.generatedTokens.count,
          duration: decodeDuration,
          durationToFirstToken: finalDurationToFirstToken
        ),
        wasStopped: self.isStopped.load(ordering: .relaxed),
        tokens: self.generatedTokens,
        response: response,
        toolCalls: self.toolCalls,
        metadata: metadata
      )
    }
  }
#endif
