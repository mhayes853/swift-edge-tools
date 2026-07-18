#if XGrammar
  import CustomDump
  import EdgeTools
  import Testing

  @Suite
  struct `XGR serialization tests` {
    @Test
    func `Tokenizer Info Round Trips Through JSON`() throws {
      let tokenizerInfo = try XGRTokenizerInfo(
        encodedVocabulary: ["a", ""],
        vocabularyType: .raw,
        stopTokenIDs: [1]
      )
      let restoredTokenizerInfo = try XGRTokenizerInfo(
        serializedJSON: try tokenizerInfo.serializedJSON()
      )

      let serializedJSON = try tokenizerInfo.serializedJSON()
      expectNoDifference(try restoredTokenizerInfo.serializedJSON(), serializedJSON)
    }

    @Test
    func `Grammar And Compiled Grammar Round Trip Through JSON`() throws {
      let tokenizerInfo = try XGRTokenizerInfo(
        encodedVocabulary: ["a", ""],
        vocabularyType: .raw,
        stopTokenIDs: [1]
      )
      let grammar = try XGRGrammar(literal: "a")
      let restoredGrammar = try XGRGrammar(serializedJSON: grammar.serializedJSON())
      let ebnf = grammar.ebnf
      expectNoDifference(restoredGrammar.ebnf, ebnf)

      let compiler = try XGRCompiler(tokenizerInfo: tokenizerInfo)
      let compiledGrammar = try compiler.compile(grammar)
      let restoredCompiledGrammar = try XGRCompiledGrammar(
        serializedJSON: try compiledGrammar.serializedJSON(),
        tokenizerInfo: tokenizerInfo
      )
      expectNoDifference(restoredCompiledGrammar.grammar.ebnf, ebnf)
    }

    @Test
    func `Hugging Face Metadata Detects Byte Fallback And Prefix Space`() throws {
      let backendJSON =
        #"{"decoder":{"type":"ByteFallback"},"normalizer":{"type":"Prepend","prepend":"▁"}}"#
      let metadata = try XGRTokenizerInfo.metadata(huggingFaceBackendJSON: backendJSON)
      let tokenizerInfo = try XGRTokenizerInfo.huggingFace(
        encodedVocabulary: ["a", ""],
        backendJSON: backendJSON
      )

      expectNoDifference(metadata.contains(#""vocab_type":1"#), true)
      expectNoDifference(metadata.contains(#""add_prefix_space":true"#), true)
      expectNoDifference(try tokenizerInfo.serializedJSON().isEmpty, false)
    }
  }
#endif
