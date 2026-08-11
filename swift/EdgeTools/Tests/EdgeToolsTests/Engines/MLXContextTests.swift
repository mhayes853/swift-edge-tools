#if MLX && XGrammar && canImport(MLX) && !os(WASI)
  import CustomDump
  import EdgeTools
  import MLX
  import MLXLMCommon
  import MLXNN
  import Testing

  @Suite(.serialized)
  struct `MLXContext tests` {
    @Test
    func `In Flight Transcript Mutation Does Not Change The Generation Snapshot`() async throws {
      let tokenizer = try testTokenizer()
      let eosTokenId = try requiredTestEOSToken(tokenizer: tokenizer)
      let engine = try MLXEngine<ContextTestProfile>(
        languageModel: ContextTestLanguageModel(eosTokenId: eosTokenId),
        tokenizer: tokenizer,
        vocabularySize: .needleVocabularySize
      )
      let session = EdgeToolsSession(engine: engine)
      let context = session.context(
        transcript: EdgeToolsTranscript(messages: [.system("System")])
      )

      let generation = Task {
        try await session.generate(
          prompt: .user("Question"),
          context: context,
          parameters: DefaultMLXGenerateParameters(maxTokens: 1)
        )
      }
      let capturedTranscript = await ContextTestProfile.gate.waitForCapture()
      _ = try await session.tokenize(prompt: .user("Tokenize"), context: context)
      expectNoDifference(context.isResponding, true)
      context.transcript.messages.append(.system("Injected while generating"))
      await ContextTestProfile.gate.resume()
      _ = try await generation.value

      expectNoDifference(
        capturedTranscript.messages,
        [.system("System"), .user("Question")]
      )
      expectNoDifference(
        context.transcript.messages,
        [
          .system("System"),
          .user("Question"),
          .system("Injected while generating"),
          .assistant([])
        ]
      )
      expectNoDifference(context.isResponding, false)
    }
  }

  private struct ContextTestProfile: MLXLLMModelProfile {
    typealias Prompt = EdgeToolsTranscript
    typealias GenerationParser = NeedleGenerationParser
    typealias GenerateParameters = DefaultMLXGenerateParameters
    typealias GrammarEngine = XGrammarEngine

    static let gate = TranscriptCaptureGate()

    static func grammar(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      parameters: DefaultMLXGenerateParameters,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      .universal
    }

    static func input(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput {
      await self.gate.capture(prompt)
      return LMInput(tokens: MLXArray([0]))
    }
  }

  private actor TranscriptCaptureGate {
    private var capturedTranscript: EdgeToolsTranscript?
    private var captureWaiter: CheckedContinuation<EdgeToolsTranscript, Never>?
    private var resumeWaiter: CheckedContinuation<Void, Never>?

    func capture(_ transcript: EdgeToolsTranscript) async {
      guard self.capturedTranscript == nil else {
        return
      }
      self.capturedTranscript = transcript
      self.captureWaiter?.resume(returning: transcript)
      self.captureWaiter = nil
      await withCheckedContinuation { self.resumeWaiter = $0 }
    }

    func waitForCapture() async -> EdgeToolsTranscript {
      if let capturedTranscript = self.capturedTranscript {
        return capturedTranscript
      }
      return await withCheckedContinuation { self.captureWaiter = $0 }
    }

    func resume() {
      self.resumeWaiter?.resume()
      self.resumeWaiter = nil
    }
  }

  private final class ContextTestLanguageModel:
    Module, LanguageModel, KVCacheDimensionProvider
  {
    let eosTokenId: EdgeToolsToken.ID
    var vocabularySize: Int { .needleVocabularySize }
    var kvHeads: [Int] { [1] }

    init(eosTokenId: EdgeToolsToken.ID) {
      self.eosTokenId = eosTokenId
      super.init()
    }

    func prepare(
      _ input: LMInput,
      cache: [any KVCache],
      windowSize: Int?
    ) throws -> PrepareResult {
      .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
      var logits = [Float](repeating: -100, count: self.vocabularySize)
      logits[self.eosTokenId] = 100
      return MLXArray(logits, [1, 1, self.vocabularySize])
    }
  }
#endif
