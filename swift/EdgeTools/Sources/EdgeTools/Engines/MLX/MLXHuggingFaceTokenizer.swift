#if MLX && HuggingFaceTokenizers && canImport(CTokenizers) && canImport(MLX)
  import MLXLMCommon
  import OrderedCollections
  import _EdgeToolsFoundation

  // MARK: - HuggingFaceTokenizer + MLXLMCommon

  extension HuggingFaceTokenizer: MLXLMCommon.Tokenizer {
    public func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
      self.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    public func applyChatTemplate(
      messages: [[String: any Sendable]],
      tools: [[String: any Sendable]]?,
      additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
      try self.applyChatTemplate(
        messages: try messages.map { try EdgeToolsValue(sendable: $0) },
        tools: try tools?.map { try EdgeToolsValue(sendable: $0) },
        addGenerationPrompt: true,
        additionalContext: try additionalContext?.mapValues { try EdgeToolsValue(sendable: $0) }
      )
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
