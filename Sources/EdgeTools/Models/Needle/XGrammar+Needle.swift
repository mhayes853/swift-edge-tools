#if XGrammar
  import Foundation

  // MARK: - XGR Tokenizer Info

  extension XGRTokenizerInfo {
    public static func needle(
      tokenizer: some EdgeToolsTokenizer
    ) throws -> XGRTokenizerInfo {
      try Self.needle(
        vocabulary: tokenizer.convertIdsToTokens(Array(0..<8192)),
        eosTokenID: tokenizer.eosTokenId
      )
    }

    private static func needle(
      vocabulary: [String?],
      eosTokenID: EdgeToolsToken.ID?
    ) throws -> XGRTokenizerInfo {
      guard let eosTokenID, vocabulary.allSatisfy({ $0 != nil }) else {
        throw XGRError(
          message: "Needle requires a tokenizer with an EOS token and full vocabulary."
        )
      }
      return try XGRTokenizerInfo(
        encodedVocabulary: vocabulary.compactMap { $0 },
        vocabularyType: .byteFallback,
        vocabularySize: vocabulary.count,
        stopTokenIDs: [eosTokenID],
        addPrefixSpace: true
      )
    }
  }

  // MARK: - Grammar

  extension XGRGrammar {
    public static func needle(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      let calls = try XGRGrammar.needleCalls(
        tools: tools.map { $0.needleNormalized() },
        range: range
      )
      let prefix = try XGRGrammar(literal: "<tool_call> [")
      let prefixedCalls = try prefix.concatenate(calls)
      return try prefixedCalls.concatenate(XGRGrammar(literal: "]"))
    }

    private static func needleCalls(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      guard range.lowerBound >= 0 else { throw NeedleXGRError.invalidToolInvocationRange }
      guard let firstTool = tools.first else { return try XGRGrammar(literal: "") }

      var call = try XGRGrammar.needleCall(firstTool)
      for tool in tools.dropFirst() {
        call = try call.union(XGRGrammar.needleCall(tool))
      }
      return try Self.repeatingToolCall(call, separator: ",", range: range)
    }

    private static func needleCall(_ tool: EdgeToolDefinition) throws -> XGRGrammar {
      let schema = tool.arguments.orderedKeyEncoded()
      let arguments = try XGRGrammar(
        jsonSchema: schema,
        configuration: JSONSchemaConfiguration(
          anyWhitespace: false,
          separators: .init(comma: ",", colon: ":"),
          isStrict: true
        )
      )
      let namePrefix = try XGRGrammar(literal: "{\"name\":\"")
      let named = try namePrefix.concatenate(XGRGrammar(literal: tool.name))
      let argumentsPrefix = try named.concatenate(XGRGrammar(literal: "\",\"arguments\":"))
      let withArguments = try argumentsPrefix.concatenate(arguments)
      return try withArguments.concatenate(XGRGrammar(literal: "}"))
    }
  }

  // MARK: - Error

  public enum NeedleXGRError: Error, Hashable, Sendable {
    case invalidToolInvocationRange
  }

  // MARK: - Helpers

  extension XGRToolCallMatcherPool {
    static func needle(maxCount: Int = 8) -> XGRToolCallMatcherPool {
      XGRToolCallMatcherPool(
        maxCount: maxCount,
        normalizeTools: { $0.map { $0.needleNormalized() } },
        makeGrammar: { try XGRGrammar.needle(tools: $0, range: $1) }
      )
    }
  }
#endif
