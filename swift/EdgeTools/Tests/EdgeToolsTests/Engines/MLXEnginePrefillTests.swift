#if MLX && XGrammar && canImport(MLX) && !os(WASI)
  import CustomDump
  import EdgeTools
  import MLX
  import MLXLMCommon
  import MLXNN
  import Testing

  @Suite(.serialized, .enabledIfMLXTests())
  struct `MLXEngineLLMPrefill tests` {
    @Test
    func `Extending Prefill Only Processes Suffix`() async throws {
      let engine = try makePrefillTestEngine(LLMPrefillTestProfile.self)
      let context = engine.context()

      context.transcript = .tokens([10, 11, 12])
      let initial = try await engine.prefill(context: context)
      context.transcript = .tokens([10, 11, 12, 13, 14])
      let extended = try await engine.prefill(context: context)

      expectNoDifference(initial.metrics.tokens, 3)
      expectNoDifference(extended.metrics.tokens, 2)
    }

    @Test
    func `Generation Reuses Prefill And Invalidates A Different Prefix`() async throws {
      let engine = try makePrefillTestEngine(LLMPrefillTestProfile.self)
      let context = engine.context()
      context.transcript = .tokens([10, 11, 12])
      _ = try await engine.prefill(context: context)

      context.transcript = .tokens([10, 11, 12, 13, 14])
      let generation = try await generate(using: engine, context: context)
      expectNoDifference(generation.prefillMetrics.tokens, 2)

      context.transcript = .tokens([10, 99, 12, 13])
      let prefill = try await engine.prefill(context: context)

      expectNoDifference(prefill.metrics.tokens, 4)
    }

    @Test
    func `Successful Generation Commits Its Prefill`() async throws {
      let generatedTokenId: EdgeToolsToken.ID = 42
      let engine = try makePrefillTestEngine(
        LLMPrefillTestProfile.self,
        sampleTokenId: generatedTokenId
      )
      let context = engine.context()
      context.transcript = .tokens([10, 11, 12])
      _ = try await engine.prefill(context: context)

      context.transcript = .tokens([10, 11, 12, 13, 14])
      let firstGeneration = try await generate(using: engine, context: context)
      expectNoDifference(firstGeneration.prefillMetrics.tokens, 2)

      context.transcript = .tokens([10, 11, 12, 13, 14, generatedTokenId, 15])
      let secondGeneration = try await generate(using: engine, context: context)

      expectNoDifference(secondGeneration.prefillMetrics.tokens, 1)
    }

    @Test
    func `Default Fused Sampler Seeds Penalties With Prompt Tokens`() async throws {
      let tokenizer = try testTokenizer()
      let engine = try MLXEngine<LLMPrefillTestProfile>(
        languageModel: PromptPenaltyLanguageModel(),
        tokenizer: tokenizer,
        vocabularySize: TestTokenizer.vocabularySize
      )
      let context = engine.context()

      let generation = try await engine.generate(
        prompt: EdgeToolsTranscript.Prompt(
          messages: EdgeToolsTranscript.tokens([1]).messages
        ),
        parameters: DefaultMLXGenerateParameters(
          sampling: EdgeToolsFusedSamplingParameters(
            temperature: 0,
            repetitionPenalty: 2
          ),
          maxTokens: 1,
          synchronizeStreamForMemorySnapshots: false
        ),
        context: context,
        channel: EdgeToolsGenerationChannel()
      ).value

      expectNoDifference(generation.tokens.map(\.id), [2])
    }
  }

  #if canImport(CoreImage) && canImport(MLXVLM)
    @Suite(.serialized, .enabledIfMLXTests())
    struct `MLXEngineVLMPrefill tests` {
      @Test
      func `Extending Prefill With Same Image Only Processes Suffix`() async throws {
        let engine = try makePrefillTestEngine(VLMPrefillTestProfile.self)
        let context = engine.context()
        context.transcript = .tokens([10, 11, 12], imageValue: 1)
        _ = try await engine.prefill(context: context)

        context.transcript = .tokens([
          ([10, 11, 12], 1),
          ([13, 14], nil)
        ])
        let extended = try await engine.prefill(context: context)

        expectNoDifference(extended.metrics.tokens, 2)
      }

      @Test
      func `Extending Prefill With Another Image Processes Full Input`() async throws {
        let engine = try makePrefillTestEngine(VLMPrefillTestProfile.self)
        let context = engine.context()
        context.transcript = .tokens([10, 11, 12], imageValue: 1)
        _ = try await engine.prefill(context: context)

        context.transcript = .tokens([
          ([10, 11, 12], 1),
          ([13, 14], 2)
        ])
        let extended = try await engine.prefill(context: context)

        expectNoDifference(extended.metrics.tokens, 5)
      }
    }
  #endif

  // MARK: - Test Profiles

  private struct LLMPrefillTestProfile: MLXLLMModelProfile {
    typealias Prompt = EdgeToolsTranscript
    typealias GenerationParser = TestGenerationParser
    typealias GenerateParameters = DefaultMLXGenerateParameters
    typealias GrammarEngine = XGrammarEngine

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
      LMInput(tokens: MLXArray(prompt.prefillTestTokenIds))
    }

    static func prefillInput(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput {
      try await self.input(
        prompt: prompt,
        tools: tools,
        tokenizer: tokenizer,
        processor: processor
      )
    }
  }

  #if canImport(CoreImage) && canImport(MLXVLM)
    private struct VLMPrefillTestProfile: MLXVLMModelProfile {
      typealias Prompt = EdgeToolsTranscript
      typealias GenerationParser = TestGenerationParser
      typealias GenerateParameters = DefaultMLXGenerateParameters
      typealias GrammarEngine = XGrammarEngine

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
        LMInput(
          text: LMInput.Text(
            tokens: MLXArray(prompt.prefillTestTokenIds).expandedDimensions(axis: 0),
            mask: MLXArray.ones([1, prompt.prefillTestTokenIds.count]).asType(.int8)
          ),
          image: prompt.prefillTestImageValue.map {
            LMInput.ProcessedImage(
              pixels: MLXArray([$0], [1, 1, 1]),
              frames: [THW(1, 1, 1)]
            )
          }
        )
      }
    }
  #endif

  extension EdgeToolsTranscript {
    fileprivate static func tokens(
      _ tokenIds: [EdgeToolsToken.ID],
      imageValue: UInt8? = nil
    ) -> Self {
      self.tokens([(tokenIds, imageValue)])
    }

    fileprivate static func tokens(
      _ messages: [([EdgeToolsToken.ID], UInt8?)]
    ) -> Self {
      Self(
        messages: messages.map { tokenIds, imageValue in
          .user(
            tokenIds.map(String.init).joined(separator: ","),
            images: imageValue.map { [Asset(bytes: [$0])] } ?? []
          )
        }
      )
    }

    fileprivate var prefillTestTokenIds: [EdgeToolsToken.ID] {
      self.messages.flatMap { message -> [EdgeToolsToken.ID] in
        guard case .user(let message) = message else {
          return []
        }
        return message.content.split(separator: ",").compactMap { EdgeToolsToken.ID($0) }
      }
    }

    fileprivate var prefillTestImageValue: Float? {
      guard
        case .user(let message) = self.messages.last,
        case .bytes(let bytes) = message.images.first?.content,
        let value = bytes.first
      else { return nil }
      return Float(value)
    }
  }

  private final class PrefillTestLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    let sampleTokenId: EdgeToolsToken.ID
    let vocabularySize: Int
    var kvHeads: [Int] { [1] }

    init(sampleTokenId: EdgeToolsToken.ID, vocabularySize: Int) {
      self.sampleTokenId = sampleTokenId
      self.vocabularySize = vocabularySize
      super.init()
    }

    func prepare(
      _ input: LMInput,
      cache: [any KVCache],
      windowSize: Int?
    ) throws -> PrepareResult {
      if input.image != nil {
        return .logits(LMOutput(logits: self(input.text.tokens, cache: cache)))
      }
      return .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
      let tokenCount = inputs.size
      let values = MLXArray.zeros([1, 1, tokenCount, 1])
      for cache in cache ?? [] {
        _ = cache.update(keys: values, values: values)
      }
      var logits = [Float](
        repeating: -100,
        count: tokenCount * self.vocabularySize
      )
      for index in 0..<tokenCount {
        logits[index * self.vocabularySize + self.sampleTokenId] = 100
      }
      return MLXArray(logits, [1, tokenCount, self.vocabularySize])
    }
  }

  private final class PromptPenaltyLanguageModel:
    Module, LanguageModel, KVCacheDimensionProvider
  {
    var vocabularySize: Int { TestTokenizer.vocabularySize }
    var kvHeads: [Int] { [1] }

    func prepare(
      _ input: LMInput,
      cache: [any KVCache],
      windowSize: Int?
    ) throws -> PrepareResult {
      .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
      let tokenCount = inputs.size
      let values = MLXArray.zeros([1, 1, tokenCount, 1])
      for cache in cache ?? [] {
        _ = cache.update(keys: values, values: values)
      }
      var logits = [Float](repeating: -100, count: tokenCount * self.vocabularySize)
      for index in 0..<tokenCount {
        logits[index * self.vocabularySize + 1] = 5
        logits[index * self.vocabularySize + 2] = 3
      }
      return MLXArray(logits, [1, tokenCount, self.vocabularySize])
    }
  }

  // MARK: - Test Helpers

  private func makePrefillTestEngine<Profile: MLXModelProfile>(
    _: Profile.Type,
    sampleTokenId: EdgeToolsToken.ID? = nil
  ) throws -> MLXEngine<Profile> where Profile.Prompt == EdgeToolsTranscript {
    let tokenizer = try testTokenizer()
    let eosTokenId = try requiredTestEOSToken(tokenizer: tokenizer)
    return try MLXEngine<Profile>(
      languageModel: PrefillTestLanguageModel(
        sampleTokenId: sampleTokenId ?? eosTokenId,
        vocabularySize: TestTokenizer.vocabularySize
      ),
      tokenizer: tokenizer,
      vocabularySize: TestTokenizer.vocabularySize
    )
  }

  private func generate<Profile: MLXModelProfile>(
    using engine: MLXEngine<Profile>,
    context: MLXContext<Profile>
  ) async throws -> EdgeToolsEngineGeneration
  where
    Profile.Prompt == EdgeToolsTranscript,
    Profile.GenerateParameters == DefaultMLXGenerateParameters
  {
    let task = try engine.generate(
      prompt: EdgeToolsTranscript.Prompt(messages: []),
      parameters: DefaultMLXGenerateParameters(
        sampler: ArgMaxSampler(),
        maxTokens: 1,
        synchronizeStreamForMemorySnapshots: false
      ),
      context: context,
      channel: EdgeToolsGenerationChannel()
    )
    return try await task.value
  }
#endif
