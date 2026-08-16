#if LlamaCore
  import EdgeTools
  import EdgeToolsCore
  import Foundation

  // MARK: - MockLlamaVocabulary

  enum MockLlamaVocabulary {
    static let pieces = ["<unk>", "<bos>", "<eos>", "▁hello", "▁world", "!"]

    nonisolated(unsafe) static let modelPointer = OpaquePointer(bitPattern: 0x1)!
    nonisolated(unsafe) static let contextPointer = OpaquePointer(bitPattern: 0x2)!

    // The raw API returns C strings; these are interned for the process lifetime.
    nonisolated(unsafe) static let pieceCStrings: [UnsafePointer<CChar>] = pieces.map {
      UnsafePointer(strdup($0)!)
    }

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

    static func text(for ids: [EdgeToolsToken.ID], renderSpecialTokens: Bool) -> String {
      ids.compactMap { id -> String? in
        guard Self.pieces.indices.contains(id) else { return nil }
        if !renderSpecialTokens && [0, 1, 2].contains(id) {
          return nil
        }
        return Self.pieces[id].replacingOccurrences(of: "▁", with: " ")
      }
      .joined()
    }
  }

  // MARK: - MockLlamaApi

  func mockLlamaApi(
    chatTemplate: String? = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
      + "{% if add_generation_prompt %}!{% endif %}",
    toolChatTemplate: String? = nil,
    decode: @escaping @Sendable (OpaquePointer?, LlamaDecodeBatch) -> Int32 = { _, _ in 0 },
    lastLogits: @escaping @Sendable (OpaquePointer?) -> UnsafeMutablePointer<Float>? = { _ in
      nil
    },
    probeConfidence: (@Sendable (OpaquePointer?, Int32) -> Float)? = nil,
    onMemoryCopy: (@Sendable (Int32, Int32) -> Void)? = nil,
    onCreateContext: (@Sendable () -> Void)? = nil
  ) -> LlamaAPI {
    let hasProbe = probeConfidence != nil
    nonisolated(unsafe) let chatTemplateCString = chatTemplate.map { UnsafePointer(strdup($0)!) }
    nonisolated(unsafe) let toolTemplateCString = toolChatTemplate.map {
      UnsafePointer(strdup($0)!)
    }
    let tokenize:
      @Sendable (
        OpaquePointer?, UnsafePointer<CChar>?, Int32, UnsafeMutablePointer<Int32>?, Int32,
        Bool, Bool
      ) -> Int32 = { _, text, textCount, tokens, capacity, addSpecialTokens, parseSpecialTokens in
        guard let text else { return -1 }
        let bytes = UnsafeRawPointer(text).assumingMemoryBound(to: UInt8.self)
        let string = String(
          decoding: UnsafeBufferPointer(start: bytes, count: Int(textCount)),
          as: UTF8.self
        )
        var ids = MockLlamaVocabulary.tokenIds(for: string, parseSpecialTokens: parseSpecialTokens)
        if addSpecialTokens {
          ids.insert(1, at: 0)
        }
        guard Int(capacity) >= ids.count else { return Int32(-ids.count) }
        for (index, id) in ids.enumerated() {
          tokens?[index] = Int32(id)
        }
        return Int32(ids.count)
      }
    let detokenize:
      @Sendable (
        OpaquePointer?, UnsafePointer<Int32>?, Int32, UnsafeMutablePointer<CChar>?, Int32,
        Bool, Bool
      ) -> Int32 = { _, tokens, tokenCount, text, capacity, removeSpecial, unparseSpecial in
        let ids = UnsafeBufferPointer(start: tokens, count: Int(tokenCount))
          .map { EdgeToolsToken.ID($0) }
        let output = Array(
          MockLlamaVocabulary.text(for: ids, renderSpecialTokens: unparseSpecial).utf8
        )
        guard Int(capacity) >= output.count else { return Int32(-output.count) }
        for (index, byte) in output.enumerated() {
          text?[index] = CChar(bitPattern: byte)
        }
        return Int32(output.count)
      }
    var probeReset: (@Sendable (OpaquePointer?, Int32) -> Void)?
    if hasProbe {
      probeReset = { _, _ in }
    }
    return LlamaAPI(
      backendInit: {},
      backendFree: {},
      modelLoad: { path, parameters in MockLlamaVocabulary.modelPointer },
      modelFree: { _ in },
      modelGetVocab: { $0 },
      modelMetaValStr: { _, key, buffer, capacity in
        guard let key, String(cString: key) == "general.name" else { return -1 }
        let name = Array("mock".utf8)
        guard capacity >= name.count else { return Int32(name.count) }
        for (index, byte) in name.enumerated() {
          buffer?[index] = CChar(bitPattern: byte)
        }
        return Int32(name.count)
      },
      modelChatTemplate: { _, name in name == nil ? chatTemplateCString : toolTemplateCString },
      modelHasProbe: { _ in hasProbe },
      vocabNTokens: { _ in Int32(MockLlamaVocabulary.pieces.count) },
      vocabType: { _ in LlamaVocabKind.sentencePiece.rawValue },
      vocabGetText: { _, id in
        MockLlamaVocabulary.pieceCStrings.indices.contains(Int(id))
          ? MockLlamaVocabulary.pieceCStrings[Int(id)]
          : nil
      },
      vocabBOS: { _ in 1 },
      vocabEOS: { _ in 2 },
      vocabGetAddBOS: { _ in true },
      vocabIsEOG: { _, id in id == 2 },
      tokenize: tokenize,
      detokenize: detokenize,
      contextInit: { model, parameters in
        onCreateContext?()
        return MockLlamaVocabulary.contextPointer
      },
      contextFree: { _ in },
      decode: decode,
      getLogitsIth: { context, index in lastLogits(context) },
      getMemory: { $0 },
      memorySeqRm: { _, _, _, _ in true },
      memorySeqCp: { memory, source, destination, from, to in
        onMemoryCopy?(source, destination)
      },
      probeConfidence: probeConfidence,
      probeReset: probeReset
    )
  }
#endif
