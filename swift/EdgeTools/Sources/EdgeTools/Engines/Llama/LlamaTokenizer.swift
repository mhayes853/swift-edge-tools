#if LlamaCore
  import EdgeToolsCore
  import EdgeToolsTokenizers
  import OrderedCollections

  #if XGrammar
    import EdgeToolsXGrammar
  #endif

  #if canImport(EdgeToolsChatTemplates)
    import EdgeToolsChatTemplates
  #endif

  // MARK: - LlamaTokenizer

  /// A tokenizer over the GGUF-embedded vocabulary of a loaded llama.cpp model.
  ///
  /// The tokenizer shares the model handle with the engine that loaded it; the engine
  /// remains responsible for freeing the model.
  public final class LlamaTokenizer: EdgeToolsTokenizer, @unchecked Sendable {
    private let api: LlamaApi
    private let model: LlamaModelRef

    public let eos: EdgeToolsToken?
    public let bos: EdgeToolsToken?

    public init(api: LlamaApi, model: LlamaModelRef) {
      self.api = api
      self.model = model
      self.eos = api.model.eosToken(model)
        .flatMap { id in api.model.tokenText(model, id).map { EdgeToolsToken(id: id, stringValue: $0) } }
      self.bos = api.model.bosToken(model)
        .flatMap { id in api.model.tokenText(model, id).map { EdgeToolsToken(id: id, stringValue: $0) } }
    }

    public var vocabularySize: Int {
      self.api.model.vocabularySize(self.model)
    }

    public func encode(text: String) -> [EdgeToolsToken] {
      self.encode(text: text, addSpecialTokens: true)
    }

    public func encode(text: String, addSpecialTokens: Bool) -> [EdgeToolsToken] {
      let ids =
        (try? self.api.model.tokenize(self.model, text, addSpecialTokens, true)) ?? []
      return self.tokens(forIds: ids).compactMap { $0 }
    }

    public func decode(tokens: [EdgeToolsToken.ID]) -> String {
      (try? self.api.model.detokenize(self.model, tokens, true)) ?? ""
    }

    public func tokens(forIds ids: [EdgeToolsToken.ID]) -> [EdgeToolsToken?] {
      let vocabularySize = self.vocabularySize
      return ids.map { id in
        guard id >= 0 && id < vocabularySize else { return nil }
        return self.api.model.tokenText(self.model, id)
          .map { EdgeToolsToken(id: id, stringValue: $0) }
      }
    }

    public func tokens(forTexts texts: [String]) -> [EdgeToolsToken?] {
      texts.map { text in
        guard
          let ids = try? self.api.model.tokenize(self.model, text, false, true),
          ids.count == 1,
          let id = ids.first,
          self.api.model.tokenText(self.model, id) == text
        else {
          return nil
        }
        return EdgeToolsToken(id: id, stringValue: text)
      }
    }

    /// The token IDs the vocabulary marks as ending a generation, beyond the EOS token.
    public func endOfGenerationTokenIds() -> Set<EdgeToolsToken.ID> {
      var tokenIds = Set<EdgeToolsToken.ID>()
      for id in 0..<self.vocabularySize where self.api.model.isEndOfGeneration(self.model, id) {
        tokenIds.insert(id)
      }
      return tokenIds
    }
  }

  // MARK: - Chat Templates

  #if canImport(EdgeToolsChatTemplates)
    extension LlamaTokenizer: EdgeToolsChatTokenizer {
      public func renderChatTemplate(
        messages: [EdgeToolsValue],
        tools: [EdgeToolsValue]?,
        addGenerationPrompt: Bool,
        additionalContext: [String: EdgeToolsValue]?
      ) throws -> String {
        let toolTemplate =
          tools?.isEmpty == false ? self.api.model.chatTemplate(self.model, "tool_use") : nil
        guard let source = toolTemplate ?? self.api.model.chatTemplate(self.model, nil) else {
          throw EdgeToolsTokenizerError(
            code: .missingChatTemplate,
            message: "The GGUF model does not embed a chat template."
          )
        }
        var context = OrderedDictionary<String, EdgeToolsValue>()
        if let bos {
          context["bos_token"] = .string(bos.stringValue)
        }
        if let eos {
          context["eos_token"] = .string(eos.stringValue)
        }
        context["messages"] = .array(messages)
        context["add_generation_prompt"] = .boolean(addGenerationPrompt)
        if let tools {
          context["tools"] = .array(tools)
        }
        context.merge(additionalContext ?? [:]) { _, override in override }
        do {
          return try EdgeToolsChatTemplate(source: source).render(context: .object(context))
        } catch let error as EdgeToolsChatTemplateError {
          throw EdgeToolsTokenizerError(code: .nativeFailure, message: error.message)
        }
      }

      public func applyChatTemplate(
        messages: [EdgeToolsValue],
        tools: [EdgeToolsValue]?,
        addGenerationPrompt: Bool,
        additionalContext: [String: EdgeToolsValue]?
      ) throws -> [EdgeToolsToken] {
        let ids = try self.api.model.tokenize(
          self.model,
          self.renderChatTemplate(
            messages: messages,
            tools: tools,
            addGenerationPrompt: addGenerationPrompt,
            additionalContext: additionalContext
          ),
          false,
          true
        )
        return self.tokens(forIds: ids).compactMap { $0 }
      }
    }
  #endif

  // MARK: - XGrammar

  #if XGrammar
    extension LlamaTokenizer: XGRTokenizer {
      public func tokenizerInfo(
        modelVocabularySize: Int?,
        extraStopTokenIds: Set<EdgeToolsToken.ID>
      ) throws -> XGRTokenizerInfo {
        let vocabularySize = self.vocabularySize
        let vocabulary = (0..<vocabularySize)
          .map { self.api.model.tokenText(self.model, $0) ?? "<|edge_tokenizer_unused_\($0)|>" }
        var stopTokenIds = extraStopTokenIds
        if let eosTokenId = self.eos?.id {
          stopTokenIds.insert(eosTokenId)
        }
        let vocabularyType: XGRVocabularyType =
          switch self.api.model.vocabKind(self.model) {
          case .sentencePiece: .byteFallback
          case .bytePairEncoding: .byteLevel
          default: .raw
          }
        return try XGRTokenizerInfo(
          encodedVocabulary: vocabulary,
          vocabularyType: vocabularyType,
          vocabularySize: max(modelVocabularySize ?? 0, vocabularySize),
          stopTokenIDs: stopTokenIds.sorted(),
          addPrefixSpace: self.api.model.vocabKind(self.model) == .sentencePiece
        )
      }
    }
  #endif
#endif
