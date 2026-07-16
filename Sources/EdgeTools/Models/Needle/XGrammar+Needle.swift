#if XGrammar
  import Foundation

  // MARK: - Tokenizer Info

  extension XGrammarTokenizerInfo {
    public static func needle(
      tokenizer: borrowing some EdgeToolsTokenizer & ~Copyable
    ) throws -> XGrammarTokenizerInfo {
      try Self.needle(
        vocabulary: tokenizer.convertIdsToTokens(Array(0..<8192)),
        eosTokenID: tokenizer.eosTokenId
      )
    }

    static func needle(
      erasedTokenizer tokenizer: borrowing any EdgeToolsTokenizer & ~Copyable
    ) throws -> XGrammarTokenizerInfo {
      try Self.needle(
        vocabulary: tokenizer.convertIdsToTokens(Array(0..<8192)),
        eosTokenID: tokenizer.eosTokenId
      )
    }

    private static func needle(
      vocabulary: [String?],
      eosTokenID: EdgeToolsToken.ID?
    ) throws -> XGrammarTokenizerInfo {
      guard let eosTokenID, vocabulary.allSatisfy({ $0 != nil }) else {
        throw XGrammarError(
          message: "Needle requires a tokenizer with an EOS token and full vocabulary."
        )
      }
      return try XGrammarTokenizerInfo(
        encodedVocabulary: vocabulary.compactMap { $0 },
        vocabularyType: .byteFallback,
        vocabularySize: vocabulary.count,
        stopTokenIDs: [eosTokenID],
        addPrefixSpace: true
      )
    }
  }

  // MARK: - Grammar

  extension XGrammarGrammar {
    public static func needle(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGrammarGrammar {
      let calls = try XGrammarGrammar.needleCalls(
        tools: tools.map { $0.needleNormalized() },
        range: range
      )
      let prefix = try XGrammarGrammar(literal: "<tool_call> [")
      let prefixedCalls = try prefix.concatenate(calls)
      return try prefixedCalls.concatenate(XGrammarGrammar(literal: "]"))
    }

    private static func needleCalls(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGrammarGrammar {
      guard range.lowerBound >= 0 else { throw NeedleXGrammarError.invalidToolInvocationRange }
      guard let firstTool = tools.first else { return try XGrammarGrammar(literal: "") }

      var call = try XGrammarGrammar.needleCall(firstTool)
      for tool in tools.dropFirst() {
        call = try call.union(XGrammarGrammar.needleCall(tool))
      }
      return try Self.repeatingToolCall(call, separator: ",", range: range)
    }

    private static func needleCall(_ tool: EdgeToolDefinition) throws -> XGrammarGrammar {
      let schema = tool.arguments.orderedKeyEncoded()
      let arguments = try XGrammarGrammar(
        jsonSchema: schema,
        configuration: JSONSchemaConfiguration(
          anyWhitespace: false,
          separators: .init(comma: ",", colon: ":"),
          isStrict: true
        )
      )
      let namePrefix = try XGrammarGrammar(literal: "{\"name\":\"")
      let named = try namePrefix.concatenate(XGrammarGrammar(literal: tool.name))
      let argumentsPrefix = try named.concatenate(XGrammarGrammar(literal: "\",\"arguments\":"))
      let withArguments = try argumentsPrefix.concatenate(arguments)
      return try withArguments.concatenate(XGrammarGrammar(literal: "}"))
    }
  }

  // MARK: - Error

  public enum NeedleXGrammarError: Error, Hashable, Sendable {
    case invalidToolInvocationRange
  }

  // MARK: - Helpers

  extension ToolCallGrammarMatcherPool {
    static func needle(maxCount: Int = 8) -> ToolCallGrammarMatcherPool {
      ToolCallGrammarMatcherPool(
        maxCount: maxCount,
        normalizeTools: { $0.map { $0.needleNormalized() } },
        makeGrammar: { try XGrammarGrammar.needle(tools: $0, range: $1) }
      )
    }
  }
#endif
