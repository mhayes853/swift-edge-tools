#if LlamaCore
  import CustomDump
  import EdgeTools
  import Testing

  #if XGrammar
    import EdgeToolsXGrammar
  #endif

  @Suite
  struct `LlamaTokenizer tests` {
    private let tokenizer = LlamaTokenizer(
      api: mockLlamaApi(),
      model: LlamaModelRef(rawValue: OpaquePointer(bitPattern: 0x1)!)
    )

    @Test
    func `Encodes And Decodes Through The GGUF Vocabulary`() {
      let tokens = self.tokenizer.encode(text: "hello world")

      expectNoDifference(tokens.map(\.id), [1, 3, 4])
      expectNoDifference(
        self.tokenizer.decode(tokens: tokens.map(\.id)),
        "<bos> hello world"
      )
    }

    @Test
    func `Exposes Special Tokens From The Vocabulary`() {
      expectNoDifference(self.tokenizer.eos, EdgeToolsToken(id: 2, stringValue: "<eos>"))
      expectNoDifference(self.tokenizer.bos, EdgeToolsToken(id: 1, stringValue: "<bos>"))
      expectNoDifference(self.tokenizer.endOfGenerationTokenIds(), [2])
    }

    @Test
    func `Looks Up Tokens By Text Only For Exact Single Pieces`() {
      expectNoDifference(
        self.tokenizer.tokens(forTexts: ["<eos>", "hello world"]),
        [EdgeToolsToken(id: 2, stringValue: "<eos>"), nil]
      )
    }

    @Test
    func `Out Of Range IDs Resolve To No Token`() {
      expectNoDifference(
        self.tokenizer.tokens(forIds: [3, 99, -1]),
        [EdgeToolsToken(id: 3, stringValue: "▁hello"), nil, nil]
      )
    }

    @Test
    func `Renders The GGUF Embedded Chat Template`() throws {
      expectNoDifference(
        try self.tokenizer.renderChatTemplate(
          messages: [["role": "user", "content": "hello"]],
          tools: nil,
          addGenerationPrompt: true,
          additionalContext: nil
        ),
        "hello!"
      )
      expectNoDifference(
        try self.tokenizer.applyChatTemplate(
          messages: [["role": "user", "content": "hello"]],
          tools: nil,
          addGenerationPrompt: true,
          additionalContext: nil
        ),
        [
          EdgeToolsToken(id: 3, stringValue: "▁hello"),
          EdgeToolsToken(id: 5, stringValue: "!"),
        ]
      )
    }

    @Test
    func `Missing Chat Template Throws`() {
      let tokenizer = LlamaTokenizer(
        api: mockLlamaApi(chatTemplate: nil),
        model: LlamaModelRef(rawValue: OpaquePointer(bitPattern: 0x1)!)
      )
      #expect(throws: EdgeToolsTokenizerError.self) {
        try tokenizer.renderChatTemplate(
          messages: [],
          tools: nil,
          addGenerationPrompt: false,
          additionalContext: nil
        )
      }
    }

    #if XGrammar
      @Test
      func `Builds XGrammar Tokenizer Info From The Vocabulary`() throws {
        let info = try self.tokenizer.tokenizerInfo(
          modelVocabularySize: 8,
          extraStopTokenIds: [5]
        )

        let serialized = try info.serializedJSON()
        expectNoDifference(serialized.contains("\"vocab_size\":8"), true)
        expectNoDifference(serialized.contains("\"stop_token_ids\":[2,5]"), true)
      }
    #endif
  }
#endif
