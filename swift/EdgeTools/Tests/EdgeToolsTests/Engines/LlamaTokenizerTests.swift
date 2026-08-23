#if HuggingFaceTokenizers && Llama && canImport(CLlama) && !os(WASI)
  import CustomDump
  import EdgeTools
  import EdgeToolsXGrammar
  import Testing

  @Suite(.serialized)
  struct `LlamaTokenizer tests` {
    @Test
    func `Round Trips Text Through The GGUF Vocabulary`() async throws {
      let tokenizer = try await qwen3LlamaTokenizer()
      let tokens = tokenizer.encode(text: "hello world", addSpecialTokens: false)

      expectNoDifference(tokens.isEmpty, false)
      expectNoDifference(tokenizer.decode(tokens: tokens.map(\.id)), "hello world")
    }

    @Test
    func `Marks The EOS Token As Ending A Generation`() async throws {
      let tokenizer = try await qwen3LlamaTokenizer()
      let eos = try #require(tokenizer.eos)

      expectNoDifference(tokenizer.endOfGenerationTokenIds().contains(eos.id), true)
      expectNoDifference(tokenizer.tokens(forTexts: [eos.stringValue]), [eos])
    }

    @Test
    func `Out Of Range IDs Resolve To No Token`() async throws {
      let tokenizer = try await qwen3LlamaTokenizer()

      expectNoDifference(
        tokenizer.tokens(forIds: [tokenizer.vocabularySize, -1]),
        [nil, nil]
      )
    }

    @Test
    func `IDs Outside The Native Range Decode To An Empty String`() async throws {
      let tokenizer = try await qwen3LlamaTokenizer()

      expectNoDifference(tokenizer.decode(tokens: [Int.max]), "")
    }

    @Test
    func `Renders The GGUF Embedded Chat Template`() async throws {
      let tokenizer = try await qwen3LlamaTokenizer()

      let rendered = try tokenizer.renderChatTemplate(
        messages: [["role": "user", "content": "hello"]],
        tools: nil,
        addGenerationPrompt: true,
        additionalContext: nil
      )

      expectNoDifference(rendered.contains("hello"), true)
      expectNoDifference(rendered.hasSuffix("<|im_start|>assistant\n"), true)
      expectNoDifference(
        try tokenizer.applyChatTemplate(
          messages: [["role": "user", "content": "hello"]],
          tools: nil,
          addGenerationPrompt: true,
          additionalContext: nil
        )
        .map(\.id),
        tokenizer.encode(text: rendered, addSpecialTokens: false).map(\.id)
      )
    }

    @Test
    func `Builds XGrammar Tokenizer Info From The Vocabulary`() async throws {
      let tokenizer = try await qwen3LlamaTokenizer()
      let eos = try #require(tokenizer.eos)
      let paddedSize = tokenizer.vocabularySize + 8

      let serialized = try tokenizer.tokenizerInfo(
        modelVocabularySize: paddedSize,
        extraStopTokenIds: []
      )
      .serializedJSON()

      expectNoDifference(serialized.contains("\"vocab_size\":\(paddedSize)"), true)
      expectNoDifference(serialized.contains("\"stop_token_ids\":[\(eos.id)]"), true)
    }
  }

  // MARK: - Helpers

  private func qwen3LlamaTokenizer() async throws -> LlamaTokenizer {
    let engine = try Qwen3LlamaModelEngine(
      modelPath: (try await downloadGGUFModel(id: .qwen3)).path()
    )
    return try #require(engine.tokenizer as? LlamaTokenizer)
  }
#endif
