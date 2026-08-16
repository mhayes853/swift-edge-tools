#if MLX && HuggingFaceTokenizers && canImport(CTokenizers) && canImport(MLX)
  import EdgeToolsCore
  import EdgeToolsTokenizers
  import MLXLMCommon
  import OrderedCollections
  import _EdgeToolsFoundation

  // MARK: - HuggingFaceTokenizer + MLXLMCommon
  //
  // `HuggingFaceTokenizer`'s `encode`/token lookup methods return `EdgeToolsToken` for
  // `EdgeToolsTokenizer`, not the raw `Int`/`String` that `MLXLMCommon.Tokenizer` requires, so the
  // conformance lives on a separate adapter rather than on `HuggingFaceTokenizer` itself.

  extension HuggingFaceTokenizer {
    package var mlxTokenizer: any MLXLMCommon.Tokenizer {
      MLXHuggingFaceTokenizerAdapter(tokenizer: self)
    }
  }

  private struct MLXHuggingFaceTokenizerAdapter: MLXLMCommon.Tokenizer {
    let tokenizer: HuggingFaceTokenizer

    var bosToken: String? { self.tokenizer.bos?.stringValue }
    var eosToken: String? { self.tokenizer.eos?.stringValue }
    var unknownToken: String? { self.tokenizer.unk?.stringValue }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
      self.tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens).map(\.id)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
      self.tokenizer.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
      self.tokenizer.token(forText: token)?.id
    }

    func convertIdToToken(_ id: Int) -> String? {
      self.tokenizer.token(forId: id)?.stringValue
    }

    func applyChatTemplate(
      messages: [[String: any Sendable]],
      tools: [[String: any Sendable]]?,
      additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
      try self.tokenizer.applyChatTemplate(
        messages: try messages.map { try EdgeToolsValue(sendable: $0) },
        tools: try tools?.map { try EdgeToolsValue(sendable: $0) },
        addGenerationPrompt: true,
        additionalContext: try additionalContext?.mapValues { try EdgeToolsValue(sendable: $0) }
      ).map(\.id)
    }
  }

  // MARK: - Untyped Conversion

  extension EdgeToolsValue {
    package init(sendable value: any Sendable) throws {
      switch value {
      case let value as EdgeToolsValue: self = value
      case let value as String: self = .string(value)
      case let value as Bool: self = .boolean(value)
      case let value as Int: self = .integer(value)
      case let value as any BinaryInteger: self = .integer(Int(value))
      case let value as Double: self = .number(value)
      case let value as any BinaryFloatingPoint: self = .number(Double(value))
      case let value as [any Sendable]:
        self = .array(try value.map { try Self(sendable: $0) })
      case let value as [String: any Sendable]:
        self = .object(
          OrderedDictionary(
            uniqueKeysWithValues: try value.keys.sorted().map { key in
              (key, try Self(sendable: value[key]!))
            }
          )
        )
      case is NSNull: self = .null
      default:
        throw EdgeToolsError(
          code: .unsupportedTokenizer,
          message: "A chat template value of type \(Swift.type(of: value)) is not representable."
        )
      }
    }
  }
#endif
