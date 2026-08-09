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

      let initial = try await engine.prefill(promptPrefix: .tokens([10, 11, 12]), tools: [])
      let extended = try await engine.prefill(
        promptPrefix: .tokens([10, 11, 12, 13, 14]),
        tools: []
      )

      expectNoDifference(initial.metrics.tokens, 3)
      expectNoDifference(extended.metrics.tokens, 2)
    }

    @Test
    func `Generation Only Processes Suffix After Prefill`() async throws {
      let engine = try makePrefillTestEngine(LLMPrefillTestProfile.self)
      _ = try await engine.prefill(promptPrefix: .tokens([10, 11, 12]), tools: [])

      let generation = try await generate(
        using: engine,
        prompt: .tokens([10, 11, 12, 13, 14])
      )

      expectNoDifference(generation.prefillMetrics.tokens, 2)
    }

    @Test
    func `Exact Prefill Avoids Reprocessing Prompt During Generation`() async throws {
      let engine = try makePrefillTestEngine(LLMPrefillTestProfile.self)
      let prompt = PrefillTestPrompt.tokens([10, 11, 12])
      _ = try await engine.prefill(promptPrefix: prompt, tools: [])

      let generation = try await generate(using: engine, prompt: prompt)

      expectNoDifference(generation.prefillMetrics.tokens, 0)
    }

    @Test
    func `Different Prefix Invalidates Prefill`() async throws {
      let engine = try makePrefillTestEngine(LLMPrefillTestProfile.self)
      _ = try await engine.prefill(promptPrefix: .tokens([10, 11, 12]), tools: [])

      let prefill = try await engine.prefill(
        promptPrefix: .tokens([10, 99, 12, 13]),
        tools: []
      )

      expectNoDifference(prefill.metrics.tokens, 4)
    }

    @Test
    func `Shorter Prompt Invalidates Prefill`() async throws {
      let engine = try makePrefillTestEngine(LLMPrefillTestProfile.self)
      _ = try await engine.prefill(promptPrefix: .tokens([10, 11, 12]), tools: [])

      let generation = try await generate(using: engine, prompt: .tokens([10, 11]))

      expectNoDifference(generation.prefillMetrics.tokens, 2)
    }
  }

  #if canImport(CoreImage) && canImport(MLXVLM)
    @Suite(.serialized, .enabledIfMLXTests())
    struct `MLXEngineVLMPrefill tests` {
      @Test
      func `Extending Prefill With Same Image Only Processes Suffix`() async throws {
        let engine = try makePrefillTestEngine(VLMPrefillTestProfile.self)
        _ = try await engine.prefill(
          promptPrefix: .tokens([10, 11, 12], imageValue: 1),
          tools: []
        )

        let extended = try await engine.prefill(
          promptPrefix: .tokens([10, 11, 12, 13, 14], imageValue: 1),
          tools: []
        )

        expectNoDifference(extended.metrics.tokens, 2)
      }

      @Test
      func `Generation Only Processes Suffix After Image Prefill`() async throws {
        let engine = try makePrefillTestEngine(VLMPrefillTestProfile.self)
        _ = try await engine.prefill(
          promptPrefix: .tokens([10, 11, 12], imageValue: 1),
          tools: []
        )

        let generation = try await generate(
          using: engine,
          prompt: .tokens([10, 11, 12, 13, 14], imageValue: 1)
        )

        expectNoDifference(generation.prefillMetrics.tokens, 2)
      }

      @Test
      func `Different Image In History Invalidates Prefill`() async throws {
        let engine = try makePrefillTestEngine(VLMPrefillTestProfile.self)
        _ = try await engine.prefill(
          promptPrefix: .tokens([10, 11, 12], imageValue: 1),
          tools: []
        )

        let prefill = try await engine.prefill(
          promptPrefix: .tokens([10, 11, 12, 13], imageValue: 2),
          tools: []
        )

        expectNoDifference(prefill.metrics.tokens, 4)
      }

      @Test
      func `Adding Or Removing Image Invalidates Prefill`() async throws {
        let engine = try makePrefillTestEngine(VLMPrefillTestProfile.self)
        _ = try await engine.prefill(promptPrefix: .tokens([10, 11, 12]), tools: [])

        let withImage = try await generate(
          using: engine,
          prompt: .tokens([10, 11, 12, 13], imageValue: 1)
        )
        expectNoDifference(withImage.prefillMetrics.tokens, 4)

        _ = try await engine.prefill(
          promptPrefix: .tokens([10, 11, 12], imageValue: 1),
          tools: []
        )
        let withoutImage = try await generate(
          using: engine,
          prompt: .tokens([10, 11, 12, 13])
        )
        expectNoDifference(withoutImage.prefillMetrics.tokens, 4)
      }
    }
  #endif

  // MARK: - Test Profiles

  private struct LLMPrefillTestProfile: MLXLLMModelProfile {
    typealias Prompt = PrefillTestPrompt
    typealias GenerationParser = NeedleGenerationParser
    typealias GenerateParameters = DefaultMLXGenerateParameters
    typealias GrammarCompiler = XGRCompiler
    typealias GrammarContext = XGRGrammarContext

    static func grammar(
      prompt _: PrefillTestPrompt,
      tools _: [EdgeToolDefinition],
      parameters _: DefaultMLXGenerateParameters,
      context _: XGRGrammarContext
    ) throws -> XGRGrammar {
      .universal
    }

    static func input(
      prompt: PrefillTestPrompt,
      tools _: [EdgeToolDefinition],
      tokenizer _: any EdgeToolsTokenizer,
      processor _: (any UserInputProcessor)?
    ) async throws -> LMInput {
      LMInput(tokens: MLXArray(prompt.tokenIds))
    }
  }

  #if canImport(CoreImage) && canImport(MLXVLM)
    private struct VLMPrefillTestProfile: MLXVLMModelProfile {
      typealias Prompt = PrefillTestPrompt
      typealias GenerationParser = NeedleGenerationParser
      typealias GenerateParameters = DefaultMLXGenerateParameters
      typealias GrammarCompiler = XGRCompiler
      typealias GrammarContext = XGRGrammarContext

      static func grammar(
        prompt _: PrefillTestPrompt,
        tools _: [EdgeToolDefinition],
        parameters _: DefaultMLXGenerateParameters,
        context _: XGRGrammarContext
      ) throws -> XGRGrammar {
        .universal
      }

      static func input(
        prompt: PrefillTestPrompt,
        tools _: [EdgeToolDefinition],
        tokenizer _: any EdgeToolsTokenizer,
        processor _: (any UserInputProcessor)?
      ) async throws -> LMInput {
        LMInput(
          text: LMInput.Text(
            tokens: MLXArray(prompt.tokenIds).expandedDimensions(axis: 0),
            mask: MLXArray.ones([1, prompt.tokenIds.count]).asType(.int8)
          ),
          image: prompt.imageValue.map {
            LMInput.ProcessedImage(
              pixels: MLXArray([$0], [1, 1, 1]),
              frames: [THW(1, 1, 1)]
            )
          }
        )
      }
    }
  #endif

  private struct PrefillTestPrompt: Sendable {
    var tokenIds: [EdgeToolsToken.ID]
    var imageValue: Float?

    static func tokens(
      _ tokenIds: [EdgeToolsToken.ID],
      imageValue: Float? = nil
    ) -> Self {
      Self(tokenIds: tokenIds, imageValue: imageValue)
    }
  }

  private final class PrefillTestLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
    let eosTokenId: EdgeToolsToken.ID
    let vocabularySize: Int
    var kvHeads: [Int] { [1] }

    init(eosTokenId: EdgeToolsToken.ID, vocabularySize: Int) {
      self.eosTokenId = eosTokenId
      self.vocabularySize = vocabularySize
      super.init()
    }

    func prepare(
      _ input: LMInput,
      cache: [any KVCache],
      windowSize _: Int?
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
        logits[index * self.vocabularySize + self.eosTokenId] = 100
      }
      return MLXArray(logits, [1, tokenCount, self.vocabularySize])
    }
  }

  // MARK: - Test Helpers

  private func makePrefillTestEngine<Profile: MLXModelProfile>(
    _: Profile.Type
  ) throws -> MLXEngine<Profile> {
    let tokenizer = try testTokenizer()
    let eosTokenId = try requiredTestEOSToken(tokenizer: tokenizer)
    return try MLXEngine<Profile>(
      languageModel: PrefillTestLanguageModel(
        eosTokenId: eosTokenId,
        vocabularySize: .needleVocabularySize
      ),
      tokenizer: tokenizer,
      vocabularySize: .needleVocabularySize
    )
  }

  private func generate<Profile: MLXModelProfile>(
    using engine: MLXEngine<Profile>,
    prompt: PrefillTestPrompt
  ) async throws -> EdgeToolsEngineGeneration
  where
    Profile.Prompt == PrefillTestPrompt,
    Profile.GenerateParameters == DefaultMLXGenerateParameters
  {
    let task = try engine.generate(
      prompt: prompt,
      tools: [],
      parameters: DefaultMLXGenerateParameters(
        sampler: ArgMaxSampler(),
        maxTokens: 1,
        synchronizeStreamForMemorySnapshots: false
      ),
      channel: EdgeToolsGenerationChannel()
    )
    return try await task.value
  }
#endif
