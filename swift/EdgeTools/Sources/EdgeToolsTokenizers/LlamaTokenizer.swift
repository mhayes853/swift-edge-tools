#if Llama && canImport(CLlama)
  import CLlama
  import EdgeToolsCore
  import OrderedCollections

  #if XGrammar
    import EdgeToolsXGrammar
  #endif

  // MARK: - LlamaTokenizer

  /// A tokenizer over the GGUF-embedded vocabulary of a loaded llama.cpp model.
  public final class LlamaTokenizer: EdgeToolsTokenizer, Sendable {
    package let model: LlamaModelBox

    public let eos: EdgeToolsToken?
    public let bos: EdgeToolsToken?

    public convenience init(model: consuming LlamaModel) {
      self.init(model: LlamaModelBox(model: consume model))
    }

    package init(model: LlamaModelBox) {
      self.model = model
      let vocabularyHandle = llama_model_get_vocab(model.model.handle)!
      self.eos = specialToken(
        model: model.model,
        tokenId: llama_vocab_eos(vocabularyHandle)
      )
      self.bos = specialToken(
        model: model.model,
        tokenId: llama_vocab_bos(vocabularyHandle)
      )
    }

    public var vocabularySize: Int {
      Int(llama_vocab_n_tokens(llama_model_get_vocab(self.model.model.handle)))
    }

    public func encode(text: String) -> [EdgeToolsToken] {
      self.encode(text: text, addSpecialTokens: true)
    }

    public func encode(text: String, addSpecialTokens: Bool) -> [EdgeToolsToken] {
      let ids = (try? self.tokenIds(text: text, addSpecialTokens: addSpecialTokens)) ?? []
      return self.tokens(forIds: ids).compactMap { $0 }
    }

    public func decode(tokens: [EdgeToolsToken.ID]) -> String {
      let ids = tokens.map { Int32($0) }
      let vocabularyHandle = llama_model_get_vocab(self.model.model.handle)!
      return ids.withUnsafeBufferPointer { ids in
        measuredCString(
          measure: {
            llama_detokenize(
              vocabularyHandle,
              ids.baseAddress,
              Int32(ids.count),
              nil,
              0,
              false,
              true
            )
          },
          fill: {
            llama_detokenize(
              vocabularyHandle,
              ids.baseAddress,
              Int32(ids.count),
              $0,
              Int32($1),
              false,
              true
            )
          }
        )
      } ?? ""
    }

    public func tokens(forIds ids: [EdgeToolsToken.ID]) -> [EdgeToolsToken?] {
      let vocabularyHandle = llama_model_get_vocab(self.model.model.handle)!
      let vocabularySize = Int(llama_vocab_n_tokens(vocabularyHandle))
      return ids.map { id in
        guard id >= 0 && id < vocabularySize else { return nil }
        return tokenText(model: self.model.model, tokenId: id)
          .map { EdgeToolsToken(id: id, stringValue: $0) }
      }
    }

    public func tokens(forTexts texts: [String]) -> [EdgeToolsToken?] {
      texts.map { text in
        guard
          let ids = try? self.tokenIds(text: text, addSpecialTokens: false),
          ids.count == 1,
          let id = ids.first,
          tokenText(model: self.model.model, tokenId: id) == text
        else {
          return nil
        }
        return EdgeToolsToken(id: id, stringValue: text)
      }
    }

    /// The token IDs the vocabulary marks as ending a generation, beyond the EOS token.
    public func endOfGenerationTokenIds() -> Set<EdgeToolsToken.ID> {
      let vocabularyHandle = llama_model_get_vocab(self.model.model.handle)!
      var tokenIds = Set<EdgeToolsToken.ID>()
      for id in 0..<Int(llama_vocab_n_tokens(vocabularyHandle))
      where llama_vocab_is_eog(vocabularyHandle, Int32(id)) {
        tokenIds.insert(id)
      }
      return tokenIds
    }

    fileprivate func tokenIds(
      text: String,
      addSpecialTokens: Bool
    ) throws -> [EdgeToolsToken.ID] {
      let utf8Count = Int32(text.utf8.count)
      let vocabularyHandle = llama_model_get_vocab(self.model.model.handle)!
      return try text.withCString { text in
        let count = -llama_tokenize(
          vocabularyHandle,
          text,
          utf8Count,
          nil,
          0,
          addSpecialTokens,
          true
        )
        guard count >= 0 else {
          throw LlamaRuntimeError(
            code: .tokenizationFailed,
            message: "The text could not be tokenized."
          )
        }
        var tokens = [Int32](repeating: 0, count: Int(count))
        let written = tokens.withUnsafeMutableBufferPointer { tokens in
          llama_tokenize(
            vocabularyHandle,
            text,
            utf8Count,
            tokens.baseAddress,
            Int32(tokens.count),
            addSpecialTokens,
            true
          )
        }
        guard written >= 0 else {
          throw LlamaRuntimeError(
            code: .tokenizationFailed,
            message: "The text could not be tokenized."
          )
        }
        return tokens.prefix(Int(written)).map { EdgeToolsToken.ID($0) }
      }
    }
  }

  // MARK: - Chat Templates

  #if ChatTemplates
    extension LlamaTokenizer: EdgeToolsChatTokenizer {
      public func renderChatTemplate(
        messages: [EdgeToolsValue],
        tools: [EdgeToolsValue]?,
        addGenerationPrompt: Bool,
        additionalContext: [String: EdgeToolsValue]?
      ) throws -> String {
        let toolTemplate =
          tools?.isEmpty == false ? self.chatTemplate(name: "tool_use") : nil
        guard let source = toolTemplate ?? self.chatTemplate(name: nil) else {
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
        let ids = try self.tokenIds(
          text: self.renderChatTemplate(
            messages: messages,
            tools: tools,
            addGenerationPrompt: addGenerationPrompt,
            additionalContext: additionalContext
          ),
          addSpecialTokens: false
        )
        return self.tokens(forIds: ids).compactMap { $0 }
      }

      private func chatTemplate(name: String?) -> String? {
        self.model.model.chatTemplate(named: name)
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
        var stopTokenIds = extraStopTokenIds
        if let eosTokenId = self.eos?.id {
          stopTokenIds.insert(eosTokenId)
        }
        let vocabularyHandle = llama_model_get_vocab(self.model.model.handle)!
        let vocabulary = try (0..<Int(llama_vocab_n_tokens(vocabularyHandle)))
          .map { tokenId in
            guard let text = tokenText(model: self.model.model, tokenId: tokenId) else {
              throw LlamaRuntimeError(
                code: .vocabularyUnavailable,
                message: "The vocabulary has no text for token \(tokenId)."
              )
            }
            return text
          }
        let vocabularyType: XGRVocabularyType =
          switch llama_vocab_type(vocabularyHandle) {
          case LLAMA_VOCAB_TYPE_SPM: .byteFallback
          case LLAMA_VOCAB_TYPE_BPE: .byteLevel
          default: .raw
          }
        return try XGRTokenizerInfo(
          encodedVocabulary: vocabulary,
          vocabularyType: vocabularyType,
          vocabularySize: max(modelVocabularySize ?? 0, vocabulary.count),
          stopTokenIDs: stopTokenIds.sorted(),
          addPrefixSpace: vocabularyType == .byteFallback
        )
      }
    }
  #endif

  // MARK: - Helpers

  private func tokenText(
    model: borrowing LlamaModel,
    tokenId: EdgeToolsToken.ID
  ) -> String? {
    let vocabularyHandle = llama_model_get_vocab(model.handle)!
    return llama_vocab_get_text(vocabularyHandle, Int32(tokenId)).map { String(cString: $0) }
  }

  private func specialToken(
    model: borrowing LlamaModel,
    tokenId: Int32
  ) -> EdgeToolsToken? {
    guard tokenId != -1 else { return nil }
    let id = EdgeToolsToken.ID(tokenId)
    return tokenText(model: model, tokenId: id)
      .map { EdgeToolsToken(id: id, stringValue: $0) }
  }

  private func measuredCString(
    measure: () -> Int32,
    fill: (UnsafeMutablePointer<CChar>?, Int) -> Int32
  ) -> String? {
    // Measuring calls report the required size either directly (`llama_model_meta_val_str`)
    // or negated (`llama_detokenize`); a fill that still fails reports the true error.
    let measured = measure()
    let count = measured >= 0 ? measured : -measured
    var storage = [CChar](repeating: 0, count: Int(count) + 1)
    let written = storage.withUnsafeMutableBufferPointer { fill($0.baseAddress, $0.count) }
    guard written >= 0 else { return nil }
    return storage.withUnsafeBufferPointer { buffer in
      String(decoding: buffer.prefix(Int(written)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
  }
#endif
