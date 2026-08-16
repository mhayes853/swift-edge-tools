#if LlamaCore
  import EdgeTools
  import EdgeToolsCore

  // MARK: - MockLlamaVocabulary

  struct MockLlamaVocabulary: Sendable {
    static let pieces = ["<unk>", "<bos>", "<eos>", "▁hello", "▁world", "!"]

    static func tokenIds(for text: String, parseSpecialTokens: Bool) -> [EdgeToolsToken.ID] {
      var remaining = Substring(text)
      var ids = [EdgeToolsToken.ID]()
      while !remaining.isEmpty {
        if parseSpecialTokens, remaining.hasPrefix("<bos>") {
          ids.append(1)
          remaining = remaining.dropFirst("<bos>".count)
        } else if parseSpecialTokens, remaining.hasPrefix("<eos>") {
          ids.append(2)
          remaining = remaining.dropFirst("<eos>".count)
        } else {
          let word = remaining.prefix { $0.isLetter }
          if word.isEmpty {
            let character = remaining.removeFirst()
            if character != " " {
              ids.append(Self.pieces.firstIndex(of: String(character)) ?? 0)
            }
            continue
          }
          ids.append(Self.pieces.firstIndex(of: "▁\(word)") ?? 0)
          remaining = remaining.dropFirst(word.count)
        }
      }
      return ids
    }
  }

  // MARK: - MockLlamaApi

  func mockLlamaApi(
    chatTemplate: String? = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
      + "{% if add_generation_prompt %}!{% endif %}",
    toolChatTemplate: String? = nil,
    decode: @escaping @Sendable (LlamaContextRef, LlamaDecodeBatch) throws -> Void = { _, _ in },
    lastLogits: @escaping @Sendable (LlamaContextRef) -> UnsafeMutablePointer<Float>? = { _ in
      nil
    },
    probeConfidence: (@Sendable (LlamaContextRef, Int) -> Float)? = nil,
    onMemoryCopy: (@Sendable (Int, Int) -> Void)? = nil,
    onCreateContext: (@Sendable () -> Void)? = nil
  ) -> LlamaApi {
    let hasProbe = probeConfidence != nil
    let tokenize: @Sendable (LlamaModelRef, String, Bool, Bool) throws -> [EdgeToolsToken.ID] = {
      model, text, addSpecialTokens, parseSpecialTokens in
      var ids = MockLlamaVocabulary.tokenIds(for: text, parseSpecialTokens: parseSpecialTokens)
      if addSpecialTokens {
        ids.insert(1, at: 0)
      }
      return ids
    }
    let detokenize: @Sendable (LlamaModelRef, [EdgeToolsToken.ID], Bool) throws -> String = {
      model, ids, renderSpecialTokens in
      ids.compactMap { id -> String? in
        guard MockLlamaVocabulary.pieces.indices.contains(id) else { return nil }
        if !renderSpecialTokens && [0, 1, 2].contains(id) {
          return nil
        }
        return MockLlamaVocabulary.pieces[id].replacingOccurrences(of: "▁", with: " ")
      }
      .joined()
    }
    let model = LlamaApi.Model(
      load: { path, parameters in LlamaModelRef(rawValue: OpaquePointer(bitPattern: 0x1)!) },
      free: { _ in },
      metadataValue: { model, key in key == "general.name" ? "mock" : nil },
      chatTemplate: { model, name in name == nil ? chatTemplate : toolChatTemplate },
      vocabularySize: { _ in MockLlamaVocabulary.pieces.count },
      vocabKind: { _ in .sentencePiece },
      tokenText: { model, id in
        MockLlamaVocabulary.pieces.indices.contains(id) ? MockLlamaVocabulary.pieces[id] : nil
      },
      eosToken: { _ in 2 },
      bosToken: { _ in 1 },
      addsBOSToken: { _ in true },
      isEndOfGeneration: { model, id in id == 2 },
      tokenize: tokenize,
      detokenize: detokenize,
      hasProbe: { _ in hasProbe }
    )
    var probeReset: (@Sendable (LlamaContextRef, Int) -> Void)?
    if hasProbe {
      probeReset = { _, _ in }
    }
    let create: @Sendable (LlamaModelRef, LlamaContextParameters) throws -> LlamaContextRef = {
      model, parameters in
      onCreateContext?()
      return LlamaContextRef(rawValue: OpaquePointer(bitPattern: 0x2)!)
    }
    let memoryRemove: @Sendable (LlamaContextRef, Int, Int, Int) -> Bool = { _, _, _, _ in
      true
    }
    let memoryCopy: @Sendable (LlamaContextRef, Int, Int, Int, Int) -> Void = {
      context, source, destination, from, to in
      onMemoryCopy?(source, destination)
    }
    let context = LlamaApi.Context(
      create: create,
      free: { _ in },
      decode: decode,
      lastLogits: lastLogits,
      memoryRemove: memoryRemove,
      memoryCopy: memoryCopy,
      probeConfidence: probeConfidence,
      probeReset: probeReset
    )
    return LlamaApi(
      backend: LlamaApi.Backend(initialize: {}, shutdown: {}),
      model: model,
      context: context
    )
  }
#endif
