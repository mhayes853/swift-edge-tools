#if HuggingFaceTokenizers
  import CustomDump
  import EdgeToolsTokenizers
  import Foundation
  import Testing

  @Suite
  struct `HuggingFaceTokenizer tests` {
    @Test
    func `Encodes Empty Text Without Failing`() throws {
      let tokenizer = try self.makeTokenizer()
      expectNoDifference(tokenizer.encode(text: ""), [])
      expectNoDifference(tokenizer.decode(tokens: []), "")
    }

    @Test
    func `Encodes Text Longer Than The Initial Output Buffer`() throws {
      let tokenizer = try self.makeTokenizer()
      let tokens = tokenizer.encode(text: String(repeating: "offline ", count: 4096))
      expectNoDifference(tokens.count, 4096)
      expectNoDifference(Set(tokens), [4])
    }

    @Test
    func `Reports Configured Special Tokens Through The Tokenizer Protocol`() throws {
      let tokenizer: any EdgeToolsTokenizer = try self.makeTokenizer()
      expectNoDifference(tokenizer.bosToken, "<bos>")
      expectNoDifference(tokenizer.eosToken, "<eos>")
      expectNoDifference(tokenizer.unknownToken, "<unk>")
      expectNoDifference(tokenizer.bosTokenId, 0)
      expectNoDifference(tokenizer.eosTokenId, 2)
      expectNoDifference(tokenizer.unknownTokenId, 3)
    }

    @Test
    func `Converts Unknown Tokens And Out Of Range Ids To Nil`() throws {
      let tokenizer = try self.makeTokenizer()
      expectNoDifference(tokenizer.convertTokensToIds(["offline", "absent"]), [4, nil])
      expectNoDifference(tokenizer.convertIdsToTokens([4, 9_999]), ["offline", nil])
    }

    #if XGrammar
      @Test
      func `Builds Tokenizer Info From A Sparse Vocabulary`() throws {
        let tokenizer = try HuggingFaceTokenizer(tokenizerJSON: Data(sparseTokenizerJSON.utf8))
        let info = try tokenizer.tokenizerInfo(modelVocabularySize: nil, extraStopTokenIds: [])
        let json = try info.serializedJSON()
        expectNoDifference(json.contains(#""vocab_size":41"#), true)
        expectNoDifference(json.contains("<|edge_tokenizer_unused_2|>"), true)
      }
    #endif

    private func makeTokenizer() throws -> HuggingFaceTokenizer {
      try HuggingFaceTokenizer(
        tokenizerJSON: try resource("test_tokenizer"),
        configuration: try resource("test_tokenizer_config")
      )
    }
  }

  private func resource(_ name: String) throws -> Data {
    try Data(contentsOf: Bundle.module.url(forResource: name, withExtension: "json")!)
  }

  private let sparseTokenizerJSON = """
    {
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "model": {
        "type": "BPE",
        "vocab": { "a": 0, "b": 1, "z": 40 },
        "merges": [],
        "continuing_subword_prefix": "",
        "end_of_word_suffix": "",
        "unk_token": null
      },
      "normalizer": { "type": "Lowercase" },
      "pre_tokenizer": { "type": "Whitespace" },
      "decoder": null
    }
    """
#endif
